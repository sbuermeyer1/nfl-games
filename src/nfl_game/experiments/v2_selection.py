"""Deterministic nested walk-forward selection for the Ridge-v2 challenger."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import DegenerateFeatureError, GameModel
from nfl_game.model.v2 import fit_target_ridge
from nfl_game.model.v2_config import (
    CANDIDATES,
    CandidateId,
    FeatureManifest,
    TargetConfig,
    target_tuning_grid,
)

MIN_INNER_SEASONS = 2
MIN_INNER_GAMES = 400
ONE_STANDARD_ERROR_TOLERANCE = 0.05

_TARGETS = ("margin", "total_points")
_C0_CONFIG = TargetConfig(
    candidate="C0",
    alpha=1.0,
    short_halflife=8,
    long_halflife=24,
    prior_season_weight=0.6,
)


class _TargetPredictor(Protocol):
    def predict(self, frame: pd.DataFrame) -> object: ...


TargetFitter = Callable[..., _TargetPredictor]


@dataclass(frozen=True)
class TargetSelection:
    target: str
    config: TargetConfig
    mean_inner_mae: float | None
    validation_seasons: tuple[int, ...]
    validation_games: int


@dataclass(frozen=True)
class OuterSelection:
    season: int
    margin: TargetSelection
    total: TargetSelection


@dataclass(frozen=True)
class NestedBacktestResult:
    predictions: pd.DataFrame
    selections: tuple[OuterSelection, ...]


def _candidate_number(candidate: str) -> int:
    try:
        return CANDIDATES.index(candidate)
    except ValueError as exc:
        raise ValueError(f"unsupported Ridge-v2 candidate {candidate!r}") from exc


def select_target_config(
    evaluations: pd.DataFrame,
    *,
    target: str,
) -> TargetSelection | None:
    """Select one target configuration from season-level inner-fold scores."""
    if target not in _TARGETS:
        raise ValueError(f"unsupported selection target {target!r}")
    required = ("config", "validation_season", "mae", "n_games")
    missing = sorted(set(required).difference(evaluations.columns))
    if missing:
        raise ValueError(f"selection evaluations are missing required column(s) {missing}")
    if evaluations.empty:
        return None

    grouped: dict[TargetConfig, list[tuple[int, float, int]]] = {}
    seen: set[tuple[str, int]] = set()
    for row in evaluations.loc[:, list(required)].itertuples(index=False):
        config = row.config
        if not isinstance(config, TargetConfig):
            raise TypeError("selection evaluation config values must be TargetConfig instances")
        season = int(row.validation_season)
        mae = float(row.mae)
        n_games = int(row.n_games)
        if not np.isfinite(mae) or mae < 0:
            raise ValueError("selection MAE values must be finite and non-negative")
        if n_games < 0 or float(row.n_games) != n_games:
            raise ValueError("selection game counts must be non-negative integers")
        identity = (config.key(), season)
        if identity in seen:
            raise ValueError(
                f"duplicate selection score for {config.key()} in validation season {season}"
            )
        seen.add(identity)
        grouped.setdefault(config, []).append((season, mae, n_games))

    eligible: list[TargetSelection] = []
    for config, rows in grouped.items():
        seasons = tuple(sorted(season for season, _, _ in rows))
        validation_games = sum(n_games for _, _, n_games in rows)
        if len(seasons) < MIN_INNER_SEASONS or validation_games < MIN_INNER_GAMES:
            continue
        eligible.append(
            TargetSelection(
                target=target,
                config=config,
                mean_inner_mae=float(np.mean([mae for _, mae, _ in rows])),
                validation_seasons=seasons,
                validation_games=validation_games,
            )
        )

    if not eligible:
        return None

    best_mae = min(cast(float, selection.mean_inner_mae) for selection in eligible)
    candidate_best: dict[str, float] = {}
    for selection in eligible:
        score = cast(float, selection.mean_inner_mae)
        candidate_best[selection.config.candidate] = min(
            candidate_best.get(selection.config.candidate, float("inf")), score
        )
    simple_candidates = [
        candidate
        for candidate, mae in candidate_best.items()
        if mae <= best_mae + ONE_STANDARD_ERROR_TOLERANCE + 1e-12
    ]
    selected_candidate = min(simple_candidates, key=_candidate_number)
    within_candidate = [
        selection for selection in eligible if selection.config.candidate == selected_candidate
    ]
    return min(
        within_candidate,
        key=lambda selection: (cast(float, selection.mean_inner_mae), selection.config.key()),
    )


def _target_manifest_name(target: str) -> str:
    return "margin" if target == "margin" else "total"


def _configs_for_target(manifest: FeatureManifest, target: str) -> tuple[TargetConfig, ...]:
    mapping = manifest.margin_by_candidate if target == "margin" else manifest.total_by_candidate
    unknown = sorted(set(mapping).difference(CANDIDATES))
    if unknown:
        raise ValueError(f"manifest contains unsupported candidate(s) {unknown}")

    configs: list[TargetConfig] = []
    for candidate in CANDIDATES:
        if candidate not in mapping:
            continue
        if candidate == "C5" and manifest.constants.get("c5_production_eligible") is False:
            continue
        if candidate == "C0":
            configs.append(_C0_CONFIG)
        else:
            configs.extend(target_tuning_grid(cast(CandidateId, candidate)))
    return tuple(configs)


def _materialize_target_features(
    frame: pd.DataFrame,
    target: str,
    config: TargetConfig,
    manifest: FeatureManifest,
) -> pd.DataFrame:
    if config.candidate == "C0":
        return frame
    mapping = manifest.rating_variant_columns(target, config)
    missing = sorted(set(mapping.values()).difference(frame.columns))
    if missing:
        raise ValueError(f"rating variant input is missing declared physical column(s) {missing}")
    manifested = set(manifest.columns(_target_manifest_name(target), config.candidate))
    undeclared = sorted(set(mapping).difference(manifested))
    if undeclared:
        raise ValueError(
            f"rating variant maps canonical column(s) outside the target schema: {undeclared}"
        )
    materialized = frame.copy()
    for canonical, physical in mapping.items():
        materialized[canonical] = frame[physical]
    return materialized


class _ConfiguredTargetPredictor:
    def __init__(self, fitted, target: str, config: TargetConfig, manifest: FeatureManifest):
        self._fitted = fitted
        self._target = target
        self._config = config
        self._manifest = manifest

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._config.candidate == "C0":
            predictions = self._fitted.predict(frame)
            column = "model_margin" if self._target == "margin" else "model_total"
            return predictions[column].to_numpy(dtype=float)
        materialized = _materialize_target_features(
            frame, self._target, self._config, self._manifest
        )
        columns = self._manifest.columns(
            _target_manifest_name(self._target), self._config.candidate
        )
        return self._fitted.predict(materialized.loc[:, list(columns)])


def _default_target_fitter(
    train: pd.DataFrame,
    target: str,
    config: TargetConfig,
    manifest: FeatureManifest,
    *,
    validation_season: int,
) -> _TargetPredictor:
    del validation_season
    if config.candidate == "C0":
        fitted = GameModel(estimator="ridge", alpha=1.0).fit(train)
    else:
        materialized = _materialize_target_features(train, target, config, manifest)
        fitted = fit_target_ridge(materialized, target, config, manifest)  # type: ignore[arg-type]
    return _ConfiguredTargetPredictor(fitted, target, config, manifest)


def _validate_feature_contract(features: pd.DataFrame, manifest: FeatureManifest) -> None:
    if features.columns.duplicated().any():
        duplicates = features.columns[features.columns.duplicated()].unique().tolist()
        raise ValueError(f"Ridge-v2 features contain duplicate column label(s) {duplicates}")

    required = {"game_id", "season", "margin", "total_points"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Ridge-v2 features are missing required column(s) {missing}")

    for target in _TARGETS:
        manifest_target = _target_manifest_name(target)
        mapping = (
            manifest.margin_by_candidate if target == "margin" else manifest.total_by_candidate
        )
        if "C0" not in mapping:
            raise ValueError(f"manifest has no exact C0 schema for {manifest_target}")
        if tuple(mapping["C0"]) != tuple(FEATURE_COLS):
            raise ValueError(
                f"manifest {manifest_target}/C0 schema must exactly match Ridge-v1 FEATURE_COLS"
            )
        for candidate, columns in mapping.items():
            canonical_missing = sorted(set(columns).difference(features.columns))
            if canonical_missing:
                raise ValueError(
                    f"Ridge-v2 features missing {manifest_target}/{candidate} column(s) "
                    f"{canonical_missing}"
                )
            if candidate == "C0":
                continue
            configs = target_tuning_grid(cast(CandidateId, candidate))
            seen_settings = set()
            for config in configs:
                if config.rating_key() in seen_settings:
                    continue
                seen_settings.add(config.rating_key())
                variants = manifest.rating_variant_columns(target, config)
                physical_missing = sorted(set(variants.values()).difference(features.columns))
                if physical_missing:
                    raise ValueError(
                        "rating variant input is missing declared physical column(s) "
                        f"{physical_missing}"
                    )
                undeclared = sorted(set(variants).difference(columns))
                if undeclared:
                    raise ValueError(
                        f"rating variant canonical column(s) absent from "
                        f"{manifest_target}/{candidate}: {undeclared}"
                    )


def _predict_target(
    fitter: TargetFitter,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    config: TargetConfig,
    manifest: FeatureManifest,
    *,
    validation_season: int,
) -> np.ndarray:
    fitted = fitter(train.copy(), target, config, manifest, validation_season=validation_season)
    predicted = np.asarray(fitted.predict(validation.copy()), dtype=float)
    if predicted.ndim != 1 or len(predicted) != len(validation):
        raise ValueError(
            f"{target} target fitter returned {predicted.shape!r} for {len(validation)} rows"
        )
    if not np.isfinite(predicted).all():
        raise ValueError(f"{target} target fitter returned non-finite predictions")
    return predicted


def _inner_evaluations(
    prior: pd.DataFrame,
    target: str,
    manifest: FeatureManifest,
    fitter: TargetFitter,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for validation_season in sorted(int(value) for value in prior["season"].unique()):
        train = prior.loc[prior["season"] < validation_season]
        validation = prior.loc[(prior["season"] == validation_season) & prior[target].notna()]
        if train.empty or validation.empty:
            continue
        for config in _configs_for_target(manifest, target):
            try:
                predicted = _predict_target(
                    fitter,
                    train,
                    validation,
                    target,
                    config,
                    manifest,
                    validation_season=validation_season,
                )
            except DegenerateFeatureError as exc:
                if config.candidate != "C0":
                    raise
                warnings.warn(
                    f"skipping inner validation season {validation_season} for {target}/C0: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            actual = validation[target].to_numpy(dtype=float)
            if not np.isfinite(actual).all():
                raise ValueError(f"inner validation target {target!r} contains non-finite values")
            rows.append(
                {
                    "config": config,
                    "validation_season": validation_season,
                    "mae": float(np.abs(predicted - actual).mean()),
                    "n_games": len(validation),
                }
            )
    return pd.DataFrame(rows, columns=["config", "validation_season", "mae", "n_games"])


def _fallback_selection(target: str, manifest: FeatureManifest) -> TargetSelection:
    mapping = manifest.margin_by_candidate if target == "margin" else manifest.total_by_candidate
    if "C0" not in mapping:
        raise ValueError(f"manifest has no C0 schema for {target!r} seed fallback")
    return TargetSelection(
        target=target,
        config=_C0_CONFIG,
        mean_inner_mae=None,
        validation_seasons=(),
        validation_games=0,
    )


def nested_walk_forward_v2(
    features: pd.DataFrame,
    test_seasons: list[int] | tuple[int, ...],
    manifest: FeatureManifest,
    *,
    target_fitter: TargetFitter | None = None,
) -> NestedBacktestResult:
    """Select, refit, and score each outer season without reading future rows."""
    _validate_feature_contract(features, manifest)
    if features["game_id"].isna().any() or features["game_id"].duplicated().any():
        raise ValueError("Ridge-v2 features require unique, non-null game_id values")
    if features["season"].isna().any():
        raise ValueError("Ridge-v2 features contain missing season values")

    fitter = target_fitter or _default_target_fitter
    prediction_frames: list[pd.DataFrame] = []
    selections: list[OuterSelection] = []
    for outer_season in sorted({int(season) for season in test_seasons}):
        prior = features.loc[features["season"] < outer_season]
        test = features.loc[features["season"] == outer_season]
        if prior.empty or test.empty:
            continue

        target_selections: dict[str, TargetSelection] = {}
        for target in _TARGETS:
            evaluations = _inner_evaluations(prior, target, manifest, fitter)
            selected = select_target_config(evaluations, target=target)
            target_selections[target] = selected or _fallback_selection(target, manifest)

        margin_selection = target_selections["margin"]
        total_selection = target_selections["total_points"]
        try:
            margin_predictions = _predict_target(
                fitter,
                prior,
                test,
                "margin",
                margin_selection.config,
                manifest,
                validation_season=outer_season,
            )
            total_predictions = _predict_target(
                fitter,
                prior,
                test,
                "total_points",
                total_selection.config,
                manifest,
                validation_season=outer_season,
            )
        except DegenerateFeatureError as exc:
            warnings.warn(
                f"skipping test season {outer_season}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        predicted = test.copy()
        predicted["model_margin"] = margin_predictions
        predicted["model_total"] = total_predictions
        prediction_frames.append(predicted)
        selections.append(
            OuterSelection(
                season=outer_season,
                margin=margin_selection,
                total=total_selection,
            )
        )

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions = pd.DataFrame(columns=[*features.columns, "model_margin", "model_total"])
    return NestedBacktestResult(predictions=predictions, selections=tuple(selections))
