"""Dry-run or atomically refresh the packaged 2026 schedule and features."""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_game.data.nfl import load_ngs, load_pbp, load_schedules
from nfl_game.data.schedule import is_final_game, normalize_schedule
from nfl_game.paths import PROCESSED_DIR
from nfl_game.pipeline.refresh_2026 import (
    build_refresh_artifacts,
    sha256_file,
    write_artifacts_atomic,
)
from nfl_game.ratings.epa import team_game_epa
from nfl_game.ratings.ngs import NGS_METRICS, team_week_ngs

if __package__:
    from scripts.build_tracker import (
        EXPECTED_BASELINE,
        assert_acceptance_baseline,
        build_historical_ledger,
    )
else:
    from build_tracker import (
        EXPECTED_BASELINE,
        assert_acceptance_baseline,
        build_historical_ledger,
    )

HISTORICAL_PBP_SEASONS = tuple(range(2015, 2026))


def empty_ngs_frame() -> pd.DataFrame:
    """Return the exact typed team-week NGS schema with no observations."""
    columns: dict[str, pd.Series] = {
        "season": pd.Series(dtype="int64"),
        "week": pd.Series(dtype="int64"),
        "team": pd.Series(dtype="string"),
    }
    columns.update({metric: pd.Series(dtype="float64") for metric in NGS_METRICS})
    columns.update({f"{metric}_imputed": pd.Series(dtype="int64") for metric in NGS_METRICS})
    return pd.DataFrame(columns)


def _dependency(
    loaders: Mapping[str, Callable] | None,
    name: str,
    default: Callable,
) -> Callable:
    return default if loaders is None else loaders.get(name, default)


def _frame_digest(frame: pd.DataFrame, filename: str) -> str:
    with tempfile.TemporaryDirectory(prefix="refresh-2026-digest-") as temp:
        path = Path(temp) / filename
        frame.to_parquet(path, index=False)
        digest = sha256_file(path)
    if digest is None:  # pragma: no cover - the write above guarantees the file exists
        raise RuntimeError("failed to digest staged parquet")
    return digest


def _has_completed_regular_season_game(schedule: pd.DataFrame, now: datetime) -> bool:
    return any(is_final_game(row, now) for _, row in schedule.iterrows())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic 2026 schedule and prediction-week artifacts."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    mode.add_argument("--write", action="store_true", help="atomically replace changed artifacts")
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "game_features.parquet")
    parser.add_argument("--schedule", type=Path, default=PROCESSED_DIR / "schedule_2026.parquet")
    return parser


def main(argv=None, loaders=None, now=None) -> None:
    """Build with injectable data loaders and clock; dry-run unless ``--write`` is set."""
    args = _parser().parse_args(argv)
    now = datetime.now(UTC) if now is None else now
    if now.tzinfo is None:
        raise ValueError("refresh clock must be timezone-aware")

    read_parquet = _dependency(loaders, "read_parquet", pd.read_parquet)
    schedules_loader = _dependency(loaders, "load_schedules", load_schedules)
    pbp_loader = _dependency(loaders, "load_pbp", load_pbp)
    ngs_loader = _dependency(loaders, "load_ngs", load_ngs)

    historical_features = read_parquet(args.features)
    raw_schedule = schedules_loader([2026], save=False)
    schedule = normalize_schedule(raw_schedule, 2026)
    completed_2026 = _has_completed_regular_season_game(schedule, now)

    pbp_seasons = [*HISTORICAL_PBP_SEASONS]
    if completed_2026:
        pbp_seasons.append(2026)
    team_games = team_game_epa(pbp_loader(pbp_seasons, save=False))

    if completed_2026:
        ngs = team_week_ngs(
            ngs_loader([2026], "passing", save=False),
            ngs_loader([2026], "rushing", save=False),
            ngs_loader([2026], "receiving", save=False),
        )
        if ngs.empty:
            ngs = empty_ngs_frame()
    else:
        ngs = empty_ngs_frame()

    artifacts = build_refresh_artifacts(
        historical_features=historical_features,
        schedules=schedule,
        team_games=team_games,
        ngs=ngs,
        now=now,
    )

    ledger = build_historical_ledger(artifacts.features)
    assert_acceptance_baseline(ledger, EXPECTED_BASELINE)

    old_feature_digest = sha256_file(args.features)
    old_schedule_digest = sha256_file(args.schedule)
    new_feature_digest = _frame_digest(artifacts.features, args.features.name)
    new_schedule_digest = _frame_digest(artifacts.schedule, args.schedule.name)
    print(
        f"features: {len(historical_features)} -> {len(artifacts.features)} rows; "
        f"sha256 {old_feature_digest or 'missing'} -> {new_feature_digest}"
    )
    print(
        f"schedule: {0 if old_schedule_digest is None else len(read_parquet(args.schedule))} "
        f"-> {len(artifacts.schedule)} rows; "
        f"sha256 {old_schedule_digest or 'missing'} -> {new_schedule_digest}"
    )

    if args.write:
        write_artifacts_atomic(artifacts, args.features, args.schedule)
        print("write complete")
    else:
        print("dry-run: no artifacts changed")


if __name__ == "__main__":
    main()
