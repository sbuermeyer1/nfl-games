"""Build and atomically publish prediction-ready 2026 artifacts."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.data.schedule import active_prediction_weeks
from nfl_game.model.features import FEATURE_COLS, build_game_features
from nfl_game.ratings.build import ratings_for_targets


@dataclass(frozen=True)
class RefreshArtifacts:
    """The paired feature and complete normalized schedule artifacts."""

    features: pd.DataFrame
    schedule: pd.DataFrame


def _cast_live_like_history(live: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """Keep appending live rows from changing the frozen history's column dtypes."""
    if historical.empty:
        return live
    cast = live.copy()
    for column, dtype in historical.dtypes.items():
        if column in cast.columns:
            cast[column] = cast[column].astype(dtype)
    return cast


def build_refresh_artifacts(
    historical_features: pd.DataFrame,
    schedules: pd.DataFrame,
    team_games: pd.DataFrame,
    ngs: pd.DataFrame,
    now: datetime,
) -> RefreshArtifacts:
    """Preserve frozen history and append only the first two active 2026 weeks."""
    weeks = active_prediction_weeks(schedules, now)
    targets = [(2026, week) for week in weeks]
    historical = historical_features.loc[historical_features["season"] <= 2025].copy()
    historical = historical.reset_index(drop=True)
    if not targets:
        return RefreshArtifacts(historical, schedules.copy())

    ratings = ratings_for_targets(team_games, targets)
    target_schedule = schedules.loc[schedules["season"].eq(2026) & schedules["week"].isin(weeks)]
    live = build_game_features(target_schedule, ratings, ngs)
    live = live.loc[live["season"].eq(2026)]
    live = _cast_live_like_history(live, historical)
    combined = pd.concat([historical, live], ignore_index=True)
    if combined["game_id"].duplicated().any():
        raise ValueError("refreshed game features contain duplicate game_id values")
    if combined[FEATURE_COLS].isna().any().any():
        raise ValueError("refreshed game features contain null model features")
    values = combined[FEATURE_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("refreshed game features contain non-finite model features")
    return RefreshArtifacts(combined, schedules.copy())


def sha256_file(path: Path) -> str | None:
    """Return a file's SHA-256 digest, or ``None`` when it does not exist."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


def _validate_artifacts(artifacts: RefreshArtifacts) -> None:
    features = artifacts.features
    schedule = artifacts.schedule
    required_features = {"game_id", "season", "week", *FEATURE_COLS}
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        raise ValueError(f"refreshed game features missing columns: {missing_features}")
    if features["game_id"].duplicated().any():
        raise ValueError("refreshed game features contain duplicate game_id values")
    if features[FEATURE_COLS].isna().any().any():
        raise ValueError("refreshed game features contain null model features")
    if not np.isfinite(features[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("refreshed game features contain non-finite model features")

    required_schedule = {
        "game_id",
        "season",
        "game_type",
        "week",
        "home_team",
        "away_team",
        "kickoff_at",
        "spread_line",
        "total_line",
    }
    missing_schedule = sorted(required_schedule - set(schedule.columns))
    if missing_schedule:
        raise ValueError(f"refreshed schedule missing columns: {missing_schedule}")
    if schedule.empty:
        raise ValueError("refreshed schedule contains no 2026 regular-season games")
    if schedule["game_id"].duplicated().any():
        raise ValueError("refreshed schedule contains duplicate game_id values")
    if not schedule["season"].eq(2026).all() or not schedule["game_type"].eq("REG").all():
        raise ValueError("refreshed schedule must contain only 2026 regular-season games")
    if schedule["kickoff_at"].isna().any():
        raise ValueError("refreshed schedule contains null kickoff_at values")
    for column in ("spread_line", "total_line"):
        values = pd.to_numeric(schedule[column], errors="coerce").dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"refreshed schedule contains non-finite {column} values")


def _restore_destination(destination: Path, backup: Path | None, replaced: bool) -> None:
    if not replaced:
        return
    if backup is None:
        destination.unlink(missing_ok=True)
    else:
        backup.replace(destination)


def _temporary_sibling_copy(source: Path, destination: Path, label: str) -> Path:
    """Copy validated bytes beside the destination so Windows inherits its ACL."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.{label}-",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def write_artifacts_atomic(
    artifacts: RefreshArtifacts,
    feature_path: Path,
    schedule_path: Path,
) -> None:
    """Validate and atomically replace the paired artifacts, rolling back on failure."""
    feature_path = Path(feature_path)
    schedule_path = Path(schedule_path)
    if feature_path.parent.resolve() != schedule_path.parent.resolve():
        raise ValueError("paired artifacts must use the same destination directory")
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="refresh-2026-", dir=feature_path.parent) as temp:
        temp_dir = Path(temp)
        staged_schedule = temp_dir / schedule_path.name
        staged_features = temp_dir / feature_path.name
        _write_parquet(artifacts.schedule, staged_schedule)
        _write_parquet(artifacts.features, staged_features)

        staged = RefreshArtifacts(
            features=pd.read_parquet(staged_features),
            schedule=pd.read_parquet(staged_schedule),
        )
        _validate_artifacts(staged)

        schedule_changed = sha256_file(staged_schedule) != sha256_file(schedule_path)
        features_changed = sha256_file(staged_features) != sha256_file(feature_path)
        if not schedule_changed and not features_changed:
            return

        publication_features = None
        publication_schedule = None
        feature_backup = None
        schedule_backup = None
        schedule_replaced = False
        features_replaced = False
        try:
            if schedule_changed:
                publication_schedule = _temporary_sibling_copy(
                    staged_schedule, schedule_path, "publish"
                )
            if features_changed:
                publication_features = _temporary_sibling_copy(
                    staged_features, feature_path, "publish"
                )
            if features_changed and feature_path.exists():
                feature_backup = _temporary_sibling_copy(feature_path, feature_path, "backup")
            if schedule_changed and schedule_path.exists():
                schedule_backup = _temporary_sibling_copy(schedule_path, schedule_path, "backup")

            if publication_schedule is not None:
                publication_schedule.replace(schedule_path)
                schedule_replaced = True
            if publication_features is not None:
                publication_features.replace(feature_path)
                features_replaced = True
        except Exception:
            _restore_destination(feature_path, feature_backup, features_replaced)
            _restore_destination(schedule_path, schedule_backup, schedule_replaced)
            raise
        finally:
            for temporary_path in (
                publication_features,
                publication_schedule,
                feature_backup,
                schedule_backup,
            ):
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
