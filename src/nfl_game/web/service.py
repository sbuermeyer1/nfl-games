from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.market.compare import build_slate
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
            raise ValueError(
                f"game features column {column!r} contains null or blank values"
            )

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
    def __init__(self, features: pd.DataFrame):
        missing = sorted(REQUIRED_COLUMNS - set(features.columns))
        if missing:
            raise ValueError(f"game features missing required columns: {missing}")
        if features.empty:
            raise ValueError("game features dataset is empty")
        _validate_dataset_values(features)
        if features["game_id"].duplicated().any():
            raise ValueError("game features contain duplicate game_id values")
        self._features = features.copy()
        self._cache: dict[tuple[int, str], ModelBundle] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_parquet(cls, path: str | Path) -> SlateService:
        return cls(pd.read_parquet(path))

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
        return {
            "seasons": seasons,
            "weeks": weeks,
            "estimators": sorted(ESTIMATORS),
            "default_estimator": "ridge",
            "default_edge_threshold": DEFAULT_EDGE_THRESHOLD,
            "latest": {"season": latest_season, "week": weeks[-1]},
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
            for value in self._features.loc[
                self._features["season"] < season, "season"
            ].unique()
        )
        if not prior_seasons:
            raise SlateUnavailableError(
                f"no calibration data is available before season {season}"
            )
        oos = walk_forward(
            self._features, prior_seasons, estimator=estimator, alpha=DEFAULT_ALPHA
        )
        if oos.empty:
            raise SlateUnavailableError(
                f"no calibration data is available before season {season}"
            )
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

    def slate(
        self,
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    ) -> pd.DataFrame:
        self._validate(season, week, estimator, edge_threshold)
        target = self._features[
            (self._features["season"] == season) & (self._features["week"] == week)
        ]
        if target.empty:
            raise SlateInputError(f"week {week} is not available for season {season}")
        bundle = self._bundle(season, estimator)
        preds = bundle.model.predict(target)
        probs_input = target.merge(preds, on="game_id", validate="one_to_one")
        probs = bundle.calibrator.predict(probs_input)
        slate = build_slate(target, preds, probs, edge_threshold=edge_threshold)
        if slate.empty:
            raise SlateNotFoundError(f"no games are available for season {season} week {week}")
        return slate

    def records(self, *args, **kwargs) -> list[dict]:
        slate = self.slate(*args, **kwargs).astype(object)
        return slate.where(pd.notna(slate), None).to_dict(orient="records")

    def csv(self, *args, **kwargs) -> str:
        return self.slate(*args, **kwargs).to_csv(index=False, na_rep="")
