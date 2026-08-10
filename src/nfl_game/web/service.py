from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.data.schedule import is_final_game
from nfl_game.market.compare import build_slate
from nfl_game.market.live import MarketSnapshot, MarketUnavailableError
from nfl_game.model.calibrate import Calibrator
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import (
    DEFAULT_ALPHA,
    ESTIMATORS,
    DegenerateFeatureError,
    GameModel,
)

DEFAULT_EDGE_THRESHOLD = 2.0
REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "spread_line",
    "total_line",
    "margin",
    "total_points",
    *FEATURE_COLS,
}
IDENTITY_COLUMNS = ("game_id", "away_team", "home_team")
SELECTOR_COLUMNS = ("season", "week")
LINE_TARGET_COLUMNS = ("spread_line", "total_line", "margin", "total_points")
MARKET_COLUMNS = {
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "spread_line",
    "total_line",
}
SCHEDULE_STATE_COLUMNS = {"kickoff_at", "result", "total"}


class SlateInputError(ValueError):
    """A requested option is invalid for the packaged dataset."""


class SlateUnavailableError(RuntimeError):
    """The selected slate cannot be modeled from the available prior data."""


class SlateNotFoundError(SlateUnavailableError):
    """The valid selection contains no games to return."""


@dataclass(frozen=True)
class ModelBundle:
    model: GameModel
    calibrator: Calibrator


def _is_real_number(value) -> bool:
    if isinstance(value, bool) or not pd.api.types.is_number(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _validate_dataset_values(features: pd.DataFrame) -> None:
    for column in IDENTITY_COLUMNS:
        values = features[column]
        blank = values.isna() | values.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if blank.any():
            raise ValueError(f"game features column {column!r} contains null or blank values")

    for column in SELECTOR_COLUMNS:
        values = features[column]
        if values.isna().any():
            raise ValueError(f"game features column {column!r} contains null values")
        if any(not _is_real_number(value) for value in values):
            raise ValueError(f"game features column {column!r} must contain numeric values")
        numbers = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numbers):
            raise ValueError(f"game features column {column!r} contains non-finite values")
        if any(not value.is_integer() for value in numbers):
            raise ValueError(f"game features column {column!r} contains fractional values")
        if any(value <= 0 for value in numbers):
            raise ValueError(f"game features column {column!r} contains non-positive values")

    for column in FEATURE_COLS:
        values = features[column]
        if values.isna().any():
            raise ValueError(f"game feature column {column!r} contains null values")
        if any(not _is_real_number(value) for value in values):
            raise ValueError(f"game feature column {column!r} must contain numeric values")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"game feature column {column!r} contains non-finite values")

    for column in LINE_TARGET_COLUMNS:
        values = features.loc[features[column].notna(), column]
        if any(not _is_real_number(value) for value in values):
            raise ValueError(
                f"game line/target column {column!r} must contain numeric values or null"
            )
        if any(math.isinf(float(value)) for value in values):
            raise ValueError(f"game line/target column {column!r} contains infinite values")


class SlateService:
    def __init__(
        self,
        features: pd.DataFrame,
        packaged_schedule: pd.DataFrame | None = None,
        market_provider=None,
        clock=lambda: datetime.now(UTC),
    ):
        missing = sorted(REQUIRED_COLUMNS - set(features.columns))
        if missing:
            raise ValueError(f"game features missing required columns: {missing}")
        if features.empty:
            raise ValueError("game features dataset is empty")
        _validate_dataset_values(features)
        if features["game_id"].duplicated().any():
            raise ValueError("game features contain duplicate game_id values")
        self._features = features.copy()
        schedule = features if packaged_schedule is None else packaged_schedule
        missing_market = sorted(MARKET_COLUMNS - set(schedule.columns))
        if missing_market:
            raise ValueError(f"packaged schedule missing required columns: {missing_market}")
        if schedule["game_id"].duplicated().any():
            raise ValueError("packaged schedule contains duplicate game_id values")
        self._packaged_schedule = schedule.copy()
        self._market_provider = market_provider
        self._clock = clock
        self._packaged_observed_at = clock()
        self._cache: dict[tuple[int, str], ModelBundle] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        packaged_schedule: pd.DataFrame | None = None,
        market_provider=None,
        clock=lambda: datetime.now(UTC),
    ) -> SlateService:
        return cls(pd.read_parquet(path), packaged_schedule, market_provider, clock)

    def weeks(self, season: int) -> list[int]:
        seasons = {int(value) for value in self._features["season"].unique()}
        if season not in seasons:
            raise SlateInputError(f"season {season} is not available")
        values = self._features.loc[self._features["season"] == season, "week"].unique()
        return sorted(int(value) for value in values)

    def options(self) -> dict:
        seasons = sorted(int(value) for value in self._features["season"].unique())
        latest_season = seasons[-1]
        weeks = self.weeks(latest_season)
        latest_week = weeks[-1]
        schedule = self._packaged_schedule.loc[self._packaged_schedule["season"].eq(latest_season)]
        if SCHEDULE_STATE_COLUMNS.issubset(schedule.columns):
            unplayed_weeks = sorted(
                {
                    int(row["week"])
                    for _, row in schedule.iterrows()
                    if int(row["week"]) in weeks and not is_final_game(row, self._clock())
                }
            )
            if unplayed_weeks:
                latest_week = unplayed_weeks[0]
        return {
            "seasons": seasons,
            "weeks": weeks,
            "estimators": sorted(ESTIMATORS),
            "default_estimator": "ridge",
            "default_edge_threshold": DEFAULT_EDGE_THRESHOLD,
            "latest": {"season": latest_season, "week": latest_week},
        }

    def _validate(self, season: int, week: int, estimator: str, edge_threshold: float) -> None:
        if estimator not in ESTIMATORS:
            raise SlateInputError(
                f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}"
            )
        if (
            isinstance(edge_threshold, bool)
            or not isinstance(edge_threshold, Real)
            or not math.isfinite(edge_threshold)
            or edge_threshold < 0
        ):
            raise SlateInputError("edge threshold must be a finite non-negative number")
        weeks = self.weeks(season)
        if week not in weeks:
            raise SlateInputError(f"week {week} is not available for season {season}")

    def _fit_bundle(self, season: int, estimator: str) -> ModelBundle:
        prior_seasons = sorted(
            int(value)
            for value in self._features.loc[self._features["season"] < season, "season"].unique()
        )
        if not prior_seasons:
            raise SlateUnavailableError(f"no calibration data is available before season {season}")
        oos = walk_forward(self._features, prior_seasons, estimator=estimator, alpha=DEFAULT_ALPHA)
        if oos.empty:
            raise SlateUnavailableError(f"no calibration data is available before season {season}")
        train = self._features[self._features["season"] < season]
        try:
            calibrator = Calibrator().fit(oos)
            model = GameModel(estimator=estimator, alpha=DEFAULT_ALPHA).fit(train)
        except (DegenerateFeatureError, ValueError) as exc:
            raise SlateUnavailableError(
                f"cannot train {estimator} for season {season}: {exc}"
            ) from exc
        return ModelBundle(model=model, calibrator=calibrator)

    def _bundle(self, season: int, estimator: str) -> ModelBundle:
        key = (season, estimator)
        with self._cache_lock:
            bundle = self._cache.get(key)
            if bundle is None:
                bundle = self._fit_bundle(season, estimator)
                self._cache[key] = bundle
            return bundle

    def _target(self, season: int, week: int) -> pd.DataFrame:
        target = self._features[
            (self._features["season"] == season) & (self._features["week"] == week)
        ].copy()
        if target.empty:
            raise SlateInputError(f"week {week} is not available for season {season}")
        return target

    def model_predictions(self, season: int, week: int, estimator: str = "ridge") -> pd.DataFrame:
        self._validate(season, week, estimator, DEFAULT_EDGE_THRESHOLD)
        predictions = self._bundle(season, estimator).model.predict(self._target(season, week))
        columns = ["game_id", "model_margin", "model_total"]
        return predictions.loc[:, columns].copy()

    def _market_snapshot(self, season: int) -> MarketSnapshot:
        if self._market_provider is not None:
            try:
                return self._market_provider.snapshot(season)
            except MarketUnavailableError:
                pass
        rows = self._packaged_schedule.loc[self._packaged_schedule["season"].eq(season)].copy()
        return MarketSnapshot(
            rows=rows,
            observed_at=self._packaged_observed_at,
            source="packaged",
            stale=True,
        )

    @staticmethod
    def _market_metadata(snapshot: MarketSnapshot) -> dict:
        observed_at = pd.Timestamp(snapshot.observed_at)
        if observed_at.tzinfo is None:
            observed_at = observed_at.tz_localize(UTC)
        else:
            observed_at = observed_at.tz_convert(UTC)
        return {
            "source": snapshot.source,
            "observed_at": observed_at.isoformat(),
            "stale": bool(snapshot.stale),
        }

    @staticmethod
    def _json_records(frame: pd.DataFrame) -> list[dict]:
        clean = frame.astype(object).where(pd.notna(frame), None)
        records = clean.to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if isinstance(value, (datetime, pd.Timestamp)):
                    timestamp = pd.Timestamp(value)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.tz_localize(UTC)
                    record[key] = timestamp.isoformat()
        return records

    def _overlay_market(self, target: pd.DataFrame, snapshot: MarketSnapshot) -> pd.DataFrame:
        missing = sorted(MARKET_COLUMNS - set(snapshot.rows.columns))
        if missing:
            raise SlateUnavailableError(f"market snapshot missing required columns: {missing}")
        market = snapshot.rows.loc[
            snapshot.rows["season"].eq(int(target["season"].iloc[0]))
            & snapshot.rows["week"].eq(int(target["week"].iloc[0])),
            list(MARKET_COLUMNS),
        ].copy()
        if market["game_id"].duplicated().any():
            raise SlateUnavailableError("market snapshot identity contains duplicate game_id")
        target_ids = set(target["game_id"])
        if set(market["game_id"]) != target_ids:
            raise SlateUnavailableError("market snapshot identity does not match slate games")

        market = market.set_index("game_id").loc[target["game_id"]]
        if market[["away_team", "home_team"]].isna().any(axis=None):
            raise SlateUnavailableError("market snapshot identity contains missing teams")
        if (market["away_team"].to_numpy() != target["away_team"].to_numpy()).any() or (
            market["home_team"].to_numpy() != target["home_team"].to_numpy()
        ).any():
            raise SlateUnavailableError("market snapshot identity does not match slate teams")

        overlaid = target.copy()
        try:
            overlaid["spread_line"] = pd.to_numeric(
                market["spread_line"], errors="raise"
            ).to_numpy()
            overlaid["total_line"] = pd.to_numeric(market["total_line"], errors="raise").to_numpy()
        except (TypeError, ValueError) as exc:
            raise SlateUnavailableError("market snapshot contains invalid lines") from exc
        return overlaid

    def _slate_result(
        self,
        season: int,
        week: int,
        estimator: str,
        edge_threshold: float,
    ) -> tuple[pd.DataFrame, dict]:
        self._validate(season, week, estimator, edge_threshold)
        target = self._target(season, week)
        snapshot = self._market_snapshot(season)
        target = self._overlay_market(target, snapshot)
        bundle = self._bundle(season, estimator)
        preds = bundle.model.predict(target)
        probs_input = target.merge(preds, on="game_id", validate="one_to_one")
        probs = bundle.calibrator.predict(probs_input)
        slate = build_slate(target, preds, probs, edge_threshold=edge_threshold)
        if slate.empty:
            raise SlateNotFoundError(f"no games are available for season {season} week {week}")

        available_status = "stale" if snapshot.stale else "live"
        slate["spread_market_status"] = slate["market_spread"].map(
            lambda value: "missing" if pd.isna(value) else available_status
        )
        slate["total_market_status"] = slate["market_total"].map(
            lambda value: "missing" if pd.isna(value) else available_status
        )
        return slate, self._market_metadata(snapshot)

    def slate(
        self,
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    ) -> pd.DataFrame:
        slate, _ = self._slate_result(season, week, estimator, edge_threshold)
        return slate

    def payload(
        self,
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    ) -> dict:
        slate, metadata = self._slate_result(season, week, estimator, edge_threshold)
        return {"games": self._json_records(slate), "market": metadata}

    def records(self, *args, **kwargs) -> list[dict]:
        slate, _ = self._slate_result(*args, **kwargs)
        return self._json_records(slate)

    def csv(self, *args, **kwargs) -> str:
        slate, _ = self._slate_result(*args, **kwargs)
        return slate.to_csv(index=False, na_rep="")

    def schedule_records(self, season: int) -> dict:
        snapshot = self._market_snapshot(season)
        rows = snapshot.rows.loc[snapshot.rows["season"].eq(season)].copy()
        if rows.empty:
            raise SlateInputError(f"season {season} is not available")
        return {
            "season": season,
            "games": self._json_records(rows),
            "market": self._market_metadata(snapshot),
        }
