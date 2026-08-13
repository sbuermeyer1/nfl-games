from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline

from nfl_game.model.predict import RobustStandardScaler
from nfl_game.model.v2_config import (
    MARKET_COLUMNS,
    MARKET_PROBABILITY_COLUMNS,
    FeatureManifest,
    TargetConfig,
    V2ModelConfig,
)

TargetName = Literal["margin", "total_points"]


def _manifest_columns(
    target: TargetName,
    config: TargetConfig,
    manifest: FeatureManifest,
) -> list[str]:
    manifest_target = "margin" if target == "margin" else "total"
    try:
        columns = list(manifest.columns(manifest_target, config.candidate))
    except KeyError as exc:
        raise ValueError(
            f"manifest has no {manifest_target} schema for candidate {config.candidate!r}"
        ) from exc
    if not columns:
        raise ValueError(
            f"manifest {manifest_target}/{config.candidate} schema has no candidate columns"
        )
    forbidden = sorted((MARKET_COLUMNS | MARKET_PROBABILITY_COLUMNS).intersection(columns))
    if forbidden:
        raise ValueError(
            f"market column(s) selected for {manifest_target}/{config.candidate}: {forbidden}"
        )
    return columns


def _select_model_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    target: TargetName,
    training: bool,
) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{target} model matrix is missing required column(s) {missing}")
    matrix = frame.loc[:, columns]
    validate_model_matrix(matrix, target=target, training=training)
    return matrix


def validate_model_matrix(
    matrix: pd.DataFrame,
    *,
    target: TargetName,
    training: bool = True,
) -> None:
    """Validate one target's manifested matrix without consulting the other target."""
    try:
        values = matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{target} model matrix must contain only numeric values") from exc

    bad_columns = [
        column
        for index, column in enumerate(matrix.columns)
        if not np.isfinite(values[:, index]).all()
    ]
    if bad_columns:
        raise ValueError(f"{target} model matrix contains non-finite value(s) in {bad_columns}")

    if not training:
        return
    degenerate = []
    for index, column in enumerate(matrix.columns):
        unique = np.unique(values[:, index])
        is_binary = set(unique).issubset({0.0, 1.0})
        if len(unique) < 2 and not is_binary:
            degenerate.append(column)
    if degenerate:
        raise ValueError(
            f"{target} model matrix contains degenerate non-binary feature(s) {degenerate}"
        )


def fit_target_ridge(
    train: pd.DataFrame,
    target: TargetName,
    config: TargetConfig,
    manifest: FeatureManifest,
) -> Pipeline:
    if target not in {"margin", "total_points"}:
        raise ValueError(f"unsupported Ridge-v2 target {target!r}")
    if target not in train.columns:
        raise ValueError(f"training frame is missing target column {target!r}")

    valid = train[target].notna()
    if not valid.any():
        raise ValueError(f"training target has no non-null rows for {target!r}")

    columns = _manifest_columns(target, config, manifest)
    valid_train = train.loc[valid]
    matrix = _select_model_matrix(
        valid_train,
        columns,
        target=target,
        training=True,
    )
    try:
        response = valid_train[target].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training target {target!r} must contain only numeric values") from exc
    if not np.isfinite(response).all():
        raise ValueError(f"training target contains non-finite value(s) for {target!r}")

    pipeline = make_pipeline(RobustStandardScaler(), Ridge(alpha=config.alpha))
    pipeline.fit(matrix, response)
    return pipeline


class RidgeV2Model:
    """Separate manifested Ridge pipelines for margin and game total."""

    def __init__(self, config: V2ModelConfig, manifest: FeatureManifest):
        self.config = config
        self.manifest = manifest
        self._margin: Pipeline | None = None
        self._total: Pipeline | None = None

    def fit(self, frame: pd.DataFrame) -> "RidgeV2Model":
        self._margin = None
        self._total = None
        margin = fit_target_ridge(frame, "margin", self.config.margin, self.manifest)
        total = fit_target_ridge(frame, "total_points", self.config.total, self.manifest)
        self._margin = margin
        self._total = total
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._margin is None or self._total is None:
            raise RuntimeError("call fit() before predict()")
        if "game_id" not in frame.columns:
            raise ValueError("predict() input is missing required column 'game_id'")
        if frame["game_id"].isna().any():
            raise ValueError("predict() input contains missing game_id values")
        if frame["game_id"].duplicated().any():
            raise ValueError("predict() input contains duplicate game_id values")

        margin_columns = _manifest_columns("margin", self.config.margin, self.manifest)
        total_columns = _manifest_columns("total_points", self.config.total, self.manifest)
        margin_matrix = _select_model_matrix(
            frame,
            margin_columns,
            target="margin",
            training=False,
        )
        total_matrix = _select_model_matrix(
            frame,
            total_columns,
            target="total_points",
            training=False,
        )
        return pd.DataFrame(
            {
                "game_id": frame["game_id"].to_numpy(),
                "model_margin": self._margin.predict(margin_matrix),
                "model_total": self._total.predict(total_matrix),
            }
        )
