"""Reproducible historical feature and manifest builder for the Ridge-v2 challenger."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.data.source_manifest import SourceSnapshot, schema_fingerprint
from nfl_game.data.teams import normalize_team_codes
from nfl_game.experiments.v2_selection import nested_walk_forward_v2
from nfl_game.model.features import FEATURE_COLS, TARGET_COLS
from nfl_game.model.v2_config import (
    PRIOR_SEASON_WEIGHTS,
    RATING_WINDOWS,
    FeatureManifest,
    TargetConfig,
)
from nfl_game.model.v2_features import build_v2_game_features, team_block_to_game_features
from nfl_game.ratings.personnel import PERSONNEL_FEATURE_COLS, personnel_features_for_targets
from nfl_game.ratings.pfr import (
    PFR_OUTPUT_COLS,
    team_week_pfr,
    trailing_pfr_features,
)
from nfl_game.ratings.qb import QB_FEATURE_COLS, qb_features_for_targets, qb_week_stats
from nfl_game.ratings.style import STYLE_FEATURE_COLS, style_features_for_targets, team_game_style
from nfl_game.ratings.v2_team import V2_RATING_TARGETS, team_game_v2, v2_team_ratings

V2_HISTORICAL_SEASONS = tuple(range(2015, 2026))
V2_EVALUATION_SEASONS = tuple(range(2021, 2026))

# Frozen Ridge-v1 artifacts. The plan's global constraint is that v1, its calibration and
# its recorded baseline stay untouched, so no v2 destination may resolve to one of these
# names -- the writer replaces whatever path it is handed.
PROTECTED_V1_ARTIFACT_NAMES = frozenset(
    {"game_features.parquet", "tracker_ledger.parquet", "schedule_2026.parquet"}
)

_DEFAULT_RATING_SETTING = (8, 24, 0.6)
_GAME_KEYS = ("game_id", "season", "week")
_TEAM_WEEK_KEYS = ("season", "week", "team")
_SOURCE_NAMES = (
    "schedules",
    "pbp",
    "ngs",
    "player_stats",
    "players",
    "rosters",
    "depth_charts",
    "snap_counts",
    "pfr_pass",
    "pfr_rush",
    "pfr_rec",
    "pfr_def",
    "base_features",
)


@dataclass(frozen=True)
class V2BuildInputs:
    schedules: pd.DataFrame
    pbp: pd.DataFrame
    ngs: pd.DataFrame
    player_stats: pd.DataFrame
    players: pd.DataFrame
    rosters: pd.DataFrame
    depth_charts: pd.DataFrame
    snap_counts: pd.DataFrame
    pfr: Mapping[str, pd.DataFrame]
    base_features: pd.DataFrame


@dataclass(frozen=True)
class V2BuildArtifacts:
    features: pd.DataFrame
    manifest: dict[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _semantic_json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def semantic_frame_digest(frame: pd.DataFrame) -> str:
    """Digest dataframe values, labels, order, and dtypes without relying on Parquet bytes."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(_canonical_json_bytes(schema))
    if len(frame):
        hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != pd.Timedelta(0):
        raise ValueError("retrieved_at must be UTC")


def _latest_event_at(frame: pd.DataFrame) -> datetime | None:
    for column in ("kickoff_at", "dt", "gameday", "game_date"):
        if column not in frame:
            continue
        values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
        if not values.empty:
            return values.max().to_pydatetime()
    return None


def _actual_seasons(frame: pd.DataFrame) -> tuple[int, ...]:
    if "season" not in frame:
        return ()
    numeric = pd.to_numeric(frame["season"], errors="coerce").dropna()
    return tuple(sorted({int(value) for value in numeric}))


def _source_snapshot(name: str, frame: pd.DataFrame, retrieved_at: datetime) -> SourceSnapshot:
    cells = frame.size
    coverage = float(frame.notna().to_numpy().sum() / cells) if cells else 0.0
    return SourceSnapshot(
        name=name,
        seasons=_actual_seasons(frame),
        retrieved_at=retrieved_at,
        schema_sha256=schema_fingerprint(frame),
        rows=len(frame),
        coverage={"non_null_cells": coverage},
        latest_event_at=_latest_event_at(frame),
    )


def _snapshot_payload(snapshot: SourceSnapshot) -> dict[str, object]:
    return {
        "name": snapshot.name,
        "seasons": list(snapshot.seasons),
        "retrieved_at": snapshot.retrieved_at.isoformat(),
        "schema_sha256": snapshot.schema_sha256,
        "rows": snapshot.rows,
        "coverage": snapshot.coverage,
        "latest_event_at": (
            snapshot.latest_event_at.isoformat() if snapshot.latest_event_at is not None else None
        ),
    }


def _source_frames(inputs: V2BuildInputs) -> dict[str, pd.DataFrame]:
    missing_pfr = sorted({"pass", "rush", "rec", "def"}.difference(inputs.pfr))
    if missing_pfr:
        raise ValueError(f"missing PFR source frame(s): {missing_pfr}")
    return {
        "schedules": inputs.schedules,
        "pbp": inputs.pbp,
        "ngs": inputs.ngs,
        "player_stats": inputs.player_stats,
        "players": inputs.players,
        "rosters": inputs.rosters,
        "depth_charts": inputs.depth_charts,
        "snap_counts": inputs.snap_counts,
        "pfr_pass": inputs.pfr["pass"],
        "pfr_rush": inputs.pfr["rush"],
        "pfr_rec": inputs.pfr["rec"],
        "pfr_def": inputs.pfr["def"],
        "base_features": inputs.base_features,
    }


def _historical_regular_base(inputs: V2BuildInputs) -> pd.DataFrame:
    required = {"game_id", "season", "week", "home_team", "away_team", *FEATURE_COLS, *TARGET_COLS}
    missing = sorted(required.difference(inputs.base_features.columns))
    if missing:
        raise ValueError(f"C0 base features missing column(s): {missing}")
    if inputs.base_features.columns.duplicated().any():
        raise ValueError("C0 base features contain duplicate column labels")

    schedules = inputs.schedules
    schedule_required = {"game_id", "season", "week", "game_type"}
    schedule_missing = sorted(schedule_required.difference(schedules.columns))
    if schedule_missing:
        raise ValueError(f"schedules missing column(s): {schedule_missing}")
    regular_ids = set(
        schedules.loc[
            schedules["game_type"].eq("REG") & schedules["season"].isin(V2_HISTORICAL_SEASONS),
            "game_id",
        ]
    )
    base = inputs.base_features.loc[
        inputs.base_features["season"].isin(V2_HISTORICAL_SEASONS)
    ].copy()
    unknown = sorted(set(base["game_id"]).difference(regular_ids))
    if unknown:
        raise ValueError(
            f"C0 contains game(s) absent from the regular-season schedule: {unknown[:5]}"
        )
    if base[list(_GAME_KEYS)].isna().any(axis=None) or base["game_id"].duplicated().any():
        raise ValueError("C0 requires one unique non-null row per regular-season game")
    return base.reset_index(drop=True)


def _adapt_depth_identifiers(depth_charts: pd.DataFrame) -> pd.DataFrame:
    out = depth_charts.copy()
    if "player_id" not in out and "gsis_id" in out:
        out["player_id"] = out["gsis_id"]
    if "gsis_id" not in out and "player_id" in out:
        out["gsis_id"] = out["player_id"]
    return out


def _target_team_weeks(base: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for side in ("home_team", "away_team"):
        pieces.append(base[["season", "week", side]].rename(columns={side: "team"}))
    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(list(_TEAM_WEEK_KEYS))
        .sort_values(list(_TEAM_WEEK_KEYS))
        .reset_index(drop=True)
    )


def _block_coverage(
    name: str,
    block: pd.DataFrame,
    expected: pd.DataFrame,
    numeric_columns: Sequence[str],
    *,
    production_eligible: bool,
    evaluation_seasons: Sequence[int],
) -> dict[str, object]:
    missing = sorted({*_TEAM_WEEK_KEYS, *numeric_columns}.difference(block.columns))
    if missing:
        raise ValueError(f"{name} block missing coverage column(s): {missing}")
    if block.duplicated(list(_TEAM_WEEK_KEYS)).any():
        raise ValueError(f"{name} block contains duplicate team-week identities")
    joined = expected.merge(
        block[[*_TEAM_WEEK_KEYS, *numeric_columns]],
        on=list(_TEAM_WEEK_KEYS),
        how="left",
        validate="one_to_one",
    )
    numeric = joined[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)), index=numeric.index, columns=numeric.columns
    )
    complete = numeric.notna().all(axis=1) & finite.all(axis=1)
    season_values: dict[str, float] = {}
    for season in evaluation_seasons:
        mask = joined["season"].eq(season)
        if not mask.any():
            raise ValueError(f"{name} block has no rows for required evaluation season {season}")
        coverage = float(complete.loc[mask].mean())
        season_values[str(season)] = coverage
        if production_eligible and coverage < 0.90:
            raise ValueError(
                f"{name} production block coverage below 0.9000 for season {season}: {coverage:.4f}"
            )
    return {
        "production_eligible": production_eligible,
        "minimum_required": 0.90 if production_eligible else None,
        "seasons": season_values,
    }


def _c1_numeric_columns() -> tuple[str, ...]:
    return tuple(
        f"{window}_{unit}_{target}"
        for window in ("short", "long")
        for unit in ("off", "def")
        for target in V2_RATING_TARGETS
    )


def _assemble_blocks(
    inputs: V2BuildInputs,
    base: pd.DataFrame,
    evaluation_seasons: Sequence[int],
) -> tuple[
    dict[tuple[int, int, float], pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, object],
]:
    targets = sorted({(int(row.season), int(row.week)) for row in base.itertuples(index=False)})
    normalized_base = normalize_team_codes(base, ["home_team", "away_team"])
    normalized_schedules = normalize_team_codes(inputs.schedules, ["home_team", "away_team"])
    depth = _adapt_depth_identifiers(inputs.depth_charts)

    team_games = team_game_v2(inputs.pbp)
    ratings_by_setting = {
        (short, long, prior): v2_team_ratings(team_games, targets, short, long, prior)
        for short, long in RATING_WINDOWS
        for prior in PRIOR_SEASON_WEIGHTS
    }
    qb = qb_features_for_targets(
        qb_week_stats(inputs.player_stats), depth, normalized_schedules, targets
    )
    style = style_features_for_targets(team_game_style(inputs.pbp), targets)
    personnel = personnel_features_for_targets(
        inputs.snap_counts,
        inputs.rosters,
        depth,
        inputs.players,
        normalized_schedules,
        targets,
    )

    # C5 stays on the established PFR formula path. Only the final production gate is
    # intentionally omitted here because this block is persisted for research while the
    # frozen 69.12% receiving-drop coverage keeps it selection-ineligible.
    pfr_team_weeks = team_week_pfr(inputs.pfr)
    pfr = trailing_pfr_features(pfr_team_weeks, targets, PFR_OUTPUT_COLS, halflife=8.0)
    default_ratings = ratings_by_setting[_DEFAULT_RATING_SETTING]
    blocks = {"C1": default_ratings, "C2": qb, "C3": style, "C4": personnel, "C5": pfr}

    expected = _target_team_weeks(normalized_base)
    gated = {"production_eligible": True, "evaluation_seasons": evaluation_seasons}
    coverage: dict[str, object] = {
        "C1": _block_coverage("C1", default_ratings, expected, _c1_numeric_columns(), **gated),
        "C2": _block_coverage("C2", qb, expected, QB_FEATURE_COLS, **gated),
        "C3": _block_coverage("C3", style, expected, STYLE_FEATURE_COLS, **gated),
        "C4": _block_coverage("C4", personnel, expected, PERSONNEL_FEATURE_COLS, **gated),
        "C5": _block_coverage(
            "C5",
            pfr,
            expected,
            (*PFR_OUTPUT_COLS, "pfr_imputed"),
            production_eligible=False,
            evaluation_seasons=evaluation_seasons,
        ),
    }
    coverage["C5"]["pfr_rec_drop_rate_2025"] = 0.6912  # type: ignore[index]
    return ratings_by_setting, blocks, coverage


def _add_rating_variants(
    frame: pd.DataFrame,
    base: pd.DataFrame,
    ratings_by_setting: Mapping[tuple[int, int, float], pd.DataFrame],
    manifest: FeatureManifest,
) -> pd.DataFrame:
    normalized_base = normalize_team_codes(base, ["home_team", "away_team"])
    columns: dict[str, pd.Series] = {}
    for (short, long, prior), ratings in ratings_by_setting.items():
        variant = team_block_to_game_features(normalized_base, ratings, name="C1").set_index(
            "game_id"
        )
        for target in ("margin", "total"):
            config = TargetConfig("C1", 1.0, short, long, prior)
            mapping = manifest.rating_variant_columns(target, config)
            for canonical, physical in mapping.items():
                columns[physical] = frame["game_id"].map(variant[canonical])
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(columns)], axis=1)


def _restore_exact_c0(assembled: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    extension_columns = [column for column in assembled.columns if column not in base.columns]
    indexed = assembled.set_index("game_id", drop=False)
    extensions = indexed.loc[base["game_id"], extension_columns].reset_index(drop=True)
    out = pd.concat([base.reset_index(drop=True), extensions], axis=1)
    if out.columns.duplicated().any():
        raise ValueError("Ridge-v2 assembly produced duplicate column labels")
    pd.testing.assert_frame_equal(out[base.columns], base, check_exact=True)
    return out


def _manifest_without_own_digest(manifest: Mapping[str, object]) -> dict[str, object]:
    payload = json.loads(json.dumps(manifest))
    output = payload.get("output")
    if isinstance(output, dict):
        output.pop("manifest_semantic_sha256", None)
    return payload


def _build_manifest(
    features: pd.DataFrame,
    feature_manifest: FeatureManifest,
    source_snapshots: Sequence[SourceSnapshot],
    coverage: Mapping[str, object],
    retrieved_at: datetime,
    evaluation_seasons: Sequence[int],
) -> dict[str, object]:
    snapshots = [_snapshot_payload(snapshot) for snapshot in source_snapshots]
    payload: dict[str, object] = {
        "model_version": "ridge-v2",
        "feature_schema_version": feature_manifest.version,
        "build_timestamp": retrieved_at.isoformat(),
        "historical_seasons": list(V2_HISTORICAL_SEASONS),
        "evaluation_seasons": list(evaluation_seasons),
        "feature_manifest": feature_manifest.to_dict(),
        "source_snapshots": snapshots,
        "source_manifest_sha256": _semantic_json_digest(snapshots),
        "source_row_counts": {snapshot.name: snapshot.rows for snapshot in source_snapshots},
        "block_coverage": dict(coverage),
        "selection_contract_valid": True,
        "output": {
            "rows": len(features),
            "columns": len(features.columns),
            "schema_sha256": schema_fingerprint(features),
            "features_semantic_sha256": semantic_frame_digest(features),
        },
    }
    payload["output"]["manifest_semantic_sha256"] = _semantic_json_digest(  # type: ignore[index]
        _manifest_without_own_digest(payload)
    )
    return payload


def build_v2_artifacts(
    inputs: V2BuildInputs,
    *,
    retrieved_at: datetime,
    evaluation_seasons: Sequence[int] = V2_EVALUATION_SEASONS,
) -> V2BuildArtifacts:
    """Build deterministic Ridge-v2 payloads without writing to disk."""
    _require_utc(retrieved_at)
    source_frames = _source_frames(inputs)
    if tuple(source_frames) != _SOURCE_NAMES:
        raise AssertionError("internal Ridge-v2 source order changed")
    snapshots = tuple(
        _source_snapshot(name, frame, retrieved_at) for name, frame in source_frames.items()
    )

    base = _historical_regular_base(inputs)
    ratings_by_setting, blocks, coverage = _assemble_blocks(inputs, base, evaluation_seasons)
    bundle = build_v2_game_features(normalize_team_codes(base, ["home_team", "away_team"]), blocks)
    bundle.manifest.validate_selection_contract()
    with_variants = _add_rating_variants(bundle.frame, base, ratings_by_setting, bundle.manifest)
    features = _restore_exact_c0(with_variants, base)

    # Exercise the exact selection boundary against the completed physical schema while
    # keeping this task independent of any result-bearing outer fold.
    nested = nested_walk_forward_v2(features.iloc[0:0].copy(), (), bundle.manifest)
    if not nested.predictions.empty or nested.selections:
        raise AssertionError("empty selection-contract validation produced experiment output")

    manifest = _build_manifest(
        features, bundle.manifest, snapshots, coverage, retrieved_at, evaluation_seasons
    )
    return V2BuildArtifacts(features=features, manifest=manifest)


def _validate_artifacts(artifacts: V2BuildArtifacts) -> None:
    features = artifacts.features
    manifest = artifacts.manifest
    if features.empty or features["game_id"].isna().any() or features["game_id"].duplicated().any():
        raise ValueError("Ridge-v2 feature payload requires unique non-null game rows")
    try:
        feature_manifest = FeatureManifest.from_dict(manifest["feature_manifest"])
        output = manifest["output"]
        timestamp = datetime.fromisoformat(str(manifest["build_timestamp"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed Ridge-v2 manifest payload") from exc
    _require_utc(timestamp)
    feature_manifest.validate_selection_contract()
    if output["features_semantic_sha256"] != semantic_frame_digest(features):
        raise ValueError("Ridge-v2 feature semantic digest mismatch")
    if output["schema_sha256"] != schema_fingerprint(features):
        raise ValueError("Ridge-v2 feature schema digest mismatch")
    expected_manifest_digest = _semantic_json_digest(_manifest_without_own_digest(manifest))
    if output["manifest_semantic_sha256"] != expected_manifest_digest:
        raise ValueError("Ridge-v2 manifest semantic digest mismatch")


def _temporary_sibling_copy(source: Path, destination: Path, label: str) -> Path:
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


def _restore_destination(destination: Path, backup: Path | None, replaced: bool) -> None:
    if not replaced:
        return
    if backup is None:
        destination.unlink(missing_ok=True)
    else:
        backup.replace(destination)


def write_v2_artifacts_atomic(
    artifacts: V2BuildArtifacts,
    feature_path: Path,
    manifest_path: Path,
) -> None:
    """Validate, publish, and if needed roll back the Ridge-v2 feature/manifest pair."""
    feature_path = Path(feature_path)
    manifest_path = Path(manifest_path)
    for label, path in (("feature", feature_path), ("manifest", manifest_path)):
        if path.name in PROTECTED_V1_ARTIFACT_NAMES:
            raise ValueError(
                f"refusing to write the Ridge-v2 {label} artifact over "
                f"frozen Ridge-v1 file {path.name!r}"
            )
    _validate_artifacts(artifacts)
    if feature_path.resolve() == manifest_path.resolve():
        raise ValueError("Ridge-v2 feature and manifest destinations must differ")
    if feature_path.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError("paired Ridge-v2 artifacts must use the same destination directory")
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ridge-v2-build-", dir=feature_path.parent) as temp:
        temp_dir = Path(temp)
        staged_features = temp_dir / feature_path.name
        staged_manifest = temp_dir / manifest_path.name
        artifacts.features.to_parquet(staged_features, index=False)
        staged_manifest.write_bytes(_canonical_json_bytes(artifacts.manifest))
        staged = V2BuildArtifacts(
            features=pd.read_parquet(staged_features),
            manifest=json.loads(staged_manifest.read_text(encoding="utf-8")),
        )
        _validate_artifacts(staged)

        publication_features = _temporary_sibling_copy(staged_features, feature_path, "publish")
        publication_manifest = _temporary_sibling_copy(staged_manifest, manifest_path, "publish")
        feature_backup = (
            _temporary_sibling_copy(feature_path, feature_path, "backup")
            if feature_path.exists()
            else None
        )
        manifest_backup = (
            _temporary_sibling_copy(manifest_path, manifest_path, "backup")
            if manifest_path.exists()
            else None
        )
        feature_replaced = False
        manifest_replaced = False
        try:
            publication_features.replace(feature_path)
            feature_replaced = True
            publication_manifest.replace(manifest_path)
            manifest_replaced = True
            published = V2BuildArtifacts(
                features=pd.read_parquet(feature_path),
                manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            )
            _validate_artifacts(published)
        except Exception:
            _restore_destination(manifest_path, manifest_backup, manifest_replaced)
            _restore_destination(feature_path, feature_backup, feature_replaced)
            raise
        finally:
            for path in (
                publication_features,
                publication_manifest,
                feature_backup,
                manifest_backup,
            ):
                if path is not None:
                    path.unlink(missing_ok=True)
