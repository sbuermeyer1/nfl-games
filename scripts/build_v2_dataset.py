"""Dry-run or atomically write the locked historical Ridge-v2 feature corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_game.data.nfl import (
    load_depth_charts,
    load_ngs,
    load_pbp,
    load_pfr_advstats,
    load_player_stats,
    load_players,
    load_rosters_weekly,
    load_schedules,
    load_snap_counts,
)
from nfl_game.paths import PROCESSED_DIR, V2_FEATURES_PATH, V2_MANIFEST_PATH
from nfl_game.pipeline.build_v2 import (
    V2_HISTORICAL_SEASONS,
    V2BuildArtifacts,
    V2BuildInputs,
    build_v2_artifacts,
    write_v2_artifacts_atomic,
)

V2_CORE_SEASONS = tuple(range(2016, 2026))
V2_PFR_SEASONS = tuple(range(2018, 2026))


def _dependency(
    dependencies: Mapping[str, Callable] | None,
    name: str,
    default: Callable,
) -> Callable:
    return default if dependencies is None else dependencies.get(name, default)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reproducible 2015-2025 source manifests and Ridge-v2 features."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    mode.add_argument("--write", action="store_true", help="atomically replace both v2 artifacts")
    parser.add_argument(
        "--base-features",
        type=Path,
        default=PROCESSED_DIR / "game_features.parquet",
        help="frozen Ridge-v1 C0 artifact (read-only)",
    )
    parser.add_argument("--features", type=Path, default=V2_FEATURES_PATH)
    parser.add_argument("--manifest", type=Path, default=V2_MANIFEST_PATH)
    return parser


def _load_inputs(args, dependencies: Mapping[str, Callable] | None) -> V2BuildInputs:
    schedules_loader = _dependency(dependencies, "load_schedules", load_schedules)
    pbp_loader = _dependency(dependencies, "load_pbp", load_pbp)
    ngs_loader = _dependency(dependencies, "load_ngs", load_ngs)
    player_stats_loader = _dependency(dependencies, "load_player_stats", load_player_stats)
    players_loader = _dependency(dependencies, "load_players", load_players)
    rosters_loader = _dependency(dependencies, "load_rosters_weekly", load_rosters_weekly)
    depth_loader = _dependency(dependencies, "load_depth_charts", load_depth_charts)
    snaps_loader = _dependency(dependencies, "load_snap_counts", load_snap_counts)
    pfr_loader = _dependency(dependencies, "load_pfr_advstats", load_pfr_advstats)
    read_parquet = _dependency(dependencies, "read_parquet", pd.read_parquet)

    historical = list(V2_HISTORICAL_SEASONS)
    core = list(V2_CORE_SEASONS)
    pfr_seasons = list(V2_PFR_SEASONS)
    ngs_frames = []
    for stat_type in ("passing", "rushing", "receiving"):
        frame = ngs_loader(core, stat_type, save=False).copy()
        frame["stat_type"] = stat_type
        ngs_frames.append(frame)
    ngs = pd.concat(ngs_frames, ignore_index=True, sort=False)

    return V2BuildInputs(
        schedules=schedules_loader(historical, save=False),
        pbp=pbp_loader(historical, save=False),
        ngs=ngs,
        player_stats=player_stats_loader(core, save=False),
        players=players_loader(save=False),
        rosters=rosters_loader(core, save=False),
        depth_charts=depth_loader(core, save=False),
        snap_counts=snaps_loader(core, save=False),
        pfr={
            stat_type: pfr_loader(pfr_seasons, stat_type, save=False)
            for stat_type in ("pass", "rush", "rec", "def")
        },
        base_features=read_parquet(args.base_features),
    )


def _print_report(artifacts: V2BuildArtifacts, args) -> None:
    manifest = artifacts.manifest
    for snapshot in manifest["source_snapshots"]:
        print(
            f"source {snapshot['name']}: rows={snapshot['rows']} "
            f"schema_sha256={snapshot['schema_sha256']}"
        )
        for metric, value in snapshot["coverage"].items():
            print(f"source {snapshot['name']} coverage {metric}={value:.6f}")
    for block, report in manifest["block_coverage"].items():
        for season, value in report["seasons"].items():
            print(f"block {block} coverage season={season} value={value:.6f}")
        if block == "C5":
            print(
                "block C5 research-only production_eligible=false "
                f"pfr_rec_drop_rate_2025={report['pfr_rec_drop_rate_2025']:.4f}"
            )
    output = manifest["output"]
    print(f"output rows={output['rows']} columns={output['columns']}")
    print(f"output schema_sha256={output['schema_sha256']}")
    print(f"output features_semantic_sha256={output['features_semantic_sha256']}")
    print(f"output manifest_semantic_sha256={output['manifest_semantic_sha256']}")
    print(f"feature destination={args.features}")
    print(f"manifest destination={args.manifest}")


def main(argv=None, loaders=None, retrieved_at=None) -> int:
    """Build with injectable loaders and clock; default to a non-mutating dry run."""
    args = _parser().parse_args(argv)
    retrieved_at = datetime.now(UTC) if retrieved_at is None else retrieved_at
    try:
        inputs = _load_inputs(args, loaders)
        builder = _dependency(loaders, "build_v2_artifacts", build_v2_artifacts)
        artifacts = builder(inputs, retrieved_at=retrieved_at)
        _print_report(artifacts, args)
        if args.write:
            writer = _dependency(loaders, "write_v2_artifacts_atomic", write_v2_artifacts_atomic)
            writer(artifacts, args.features, args.manifest)
            print("write complete")
        else:
            print("dry-run: no artifacts changed")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary converts failures to exit status
        print(f"ridge-v2 build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
