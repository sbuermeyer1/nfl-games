"""Dry-run or atomically rebuild the packaged QB starter advisory artifact.

This precomputes `nfl_game.ratings.starters.starter_advisory(...)` -- the same
function `NflverseStarterProvider` (market/live_starters.py) uses -- for the active
prediction weeks, and writes `data/processed/starter_advisory.parquet`.

Why this exists: the live provider downloads and parses the full nflverse depth-chart
feed (1,059,637 rows) inside a web request. Measured on the packaged app, one advisory
snapshot peaked at 944.4 MB against a 512 MB Render dyno, which OOM-killed the worker
and took the whole dashboard down (see the comment on STARTER_ADVISORY_PATH in
src/nfl_game/web/runtime.py). This script runs on a GitHub Actions runner with
gigabytes of headroom, so the heavy feed load belongs HERE, not in the web tier --
`PackagedStarterProvider` (market/packaged_starters.py) only ever reads the small
parquet file this writes.

Mirrors scripts/refresh_2026.py and scripts/update_live_tracker.py: dry-run by
default, `--write` to actually replace the file, and an atomic write (temp file in the
same directory, then `os.replace`) so a crashed run cannot leave a half-written
artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_game.data.nfl import load_depth_charts, load_player_stats, load_players, load_schedules
from nfl_game.data.schedule import active_prediction_weeks, normalize_schedule
from nfl_game.paths import PROCESSED_DIR
from nfl_game.ratings.qb import qb_week_stats
from nfl_game.ratings.starters import ADVISORY_COLS, starter_advisory

SEASON = 2026
ARTIFACT_COLUMNS = [*ADVISORY_COLS, "season", "week", "computed_at"]


def _empty_artifact() -> pd.DataFrame:
    """Typed empty artifact for when there are no active prediction weeks left."""
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype="string"),
            "home_qb": pd.Series(dtype="string"),
            "away_qb": pd.Series(dtype="string"),
            "qb_change_epa_home": pd.Series(dtype="float64"),
            "qb_change_epa_away": pd.Series(dtype="float64"),
            "qb_watch": pd.Series(dtype="Int64"),
            "qb_inferred": pd.Series(dtype="Int64"),
            "season": pd.Series(dtype="int64"),
            "week": pd.Series(dtype="int64"),
            "computed_at": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )[ARTIFACT_COLUMNS]


def _dependency(loaders, name, default):
    return default if loaders is None else loaders.get(name, default)


def _load_stats_tolerant(loader, seasons: list[int]) -> pd.DataFrame:
    """Load player stats one season at a time, tolerating an unpublished current season.

    Duplicated (not imported) from `NflverseStarterProvider._load_stats_per_season` in
    market/live_starters.py -- that module is shaped for a bounded-cache web request
    and this script must not depend on it, but the reason is identical: nflreadpy does
    not publish `stats_player_week_YYYY.parquet` until that season's games have been
    played, so a combined `[season - 1, season]` request 404s for the whole preseason
    and week 1 -- exactly when a starter advisory matters most. Loading per season and
    skipping only the one that 404s keeps the advisory alive on whichever season DID
    publish.
    """
    frames = []
    error: ConnectionError | None = None
    for season in seasons:
        try:
            frames.append(loader([season], save=False))
        except ConnectionError as exc:
            error = exc
    if not frames:
        raise error
    return pd.concat(frames, ignore_index=True)


def build_advisory_artifact(
    packaged_schedule: pd.DataFrame,
    weeks: list[int],
    now: datetime,
    depth_loader=load_depth_charts,
    stats_loader=load_player_stats,
    players_loader=load_players,
    schedule_loader=load_schedules,
) -> pd.DataFrame:
    """Build the packaged advisory frame for `weeks` of SEASON, stamped with `now`.

    Loads depth/stats/players/schedules for `[SEASON - 1, SEASON]`, mirroring
    `NflverseStarterProvider._season_frames` in market/live_starters.py, so the
    resulting advisory rows match what the live provider would have produced. The
    `schedules` frame passed to `starter_advisory` is the RAW loader output (not run
    through `normalize_schedule`), exactly as the live provider passes it -- both
    `starter_advisory` and `qb_features_for_targets` tolerate the raw feed's own
    `gameday`/`gametime` columns directly. `packaged_schedule` (normalized, 2026-only)
    is used only to attach `season`/`week` to each advisory row by `game_id`.
    """
    if not weeks:
        return _empty_artifact()

    seasons = sorted({SEASON - 1, SEASON})
    live_schedules = schedule_loader(seasons, save=False)
    depth = depth_loader(seasons, save=False)
    stats = _load_stats_tolerant(stats_loader, seasons)
    players = players_loader(save=False)

    targets = [(SEASON, week) for week in weeks]
    rows = starter_advisory(qb_week_stats(stats), depth, live_schedules, players, targets)
    if set(rows.columns) != set(ADVISORY_COLS):
        raise ValueError(f"advisory rows have unexpected columns: {sorted(rows.columns)}")

    games = packaged_schedule.loc[
        packaged_schedule["season"].eq(SEASON) & packaged_schedule["week"].isin(weeks),
        ["game_id", "season", "week"],
    ].drop_duplicates("game_id")
    merged = rows.merge(games, on="game_id", how="left", validate="one_to_one")
    if merged[["season", "week"]].isna().any(axis=None):
        raise ValueError("advisory rows do not match a scheduled game in the packaged schedule")
    merged["season"] = merged["season"].astype("int64")
    merged["week"] = merged["week"].astype("int64")
    merged["computed_at"] = pd.Timestamp(now).tz_convert(UTC)
    return merged[ARTIFACT_COLUMNS].reset_index(drop=True)


def sha256_file(path: Path) -> str | None:
    """Return a file's SHA-256 digest, or ``None`` when it does not exist."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False)
    payload = stream.getvalue()
    persisted = pd.read_parquet(io.BytesIO(payload))
    if list(persisted.columns) != ARTIFACT_COLUMNS:
        raise ValueError("staged starter advisory parquet has unexpected columns")
    return payload


def _atomic_replace(payload: bytes, destination: Path) -> None:
    """Write bytes to a temp file beside `destination`, fsync, then atomically replace."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.update-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute the packaged QB starter advisory artifact for the "
        "active prediction weeks."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    mode.add_argument("--write", action="store_true", help="atomically replace the artifact")
    parser.add_argument("--schedule", type=Path, default=PROCESSED_DIR / "schedule_2026.parquet")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "starter_advisory.parquet")
    return parser


def main(argv=None, loaders=None, now=None) -> None:
    """Build with injectable data loaders and clock; dry-run unless ``--write`` is set."""
    args = _parser().parse_args(argv)
    now = datetime.now(UTC) if now is None else now
    if now.tzinfo is None:
        raise ValueError("build clock must be timezone-aware")

    read_parquet = _dependency(loaders, "read_parquet", pd.read_parquet)
    depth_loader = _dependency(loaders, "load_depth_charts", load_depth_charts)
    stats_loader = _dependency(loaders, "load_player_stats", load_player_stats)
    players_loader = _dependency(loaders, "load_players", load_players)
    schedule_loader = _dependency(loaders, "load_schedules", load_schedules)

    packaged_schedule = normalize_schedule(read_parquet(args.schedule), SEASON)
    weeks = active_prediction_weeks(packaged_schedule, now)

    advisory = build_advisory_artifact(
        packaged_schedule,
        weeks,
        now,
        depth_loader=depth_loader,
        stats_loader=stats_loader,
        players_loader=players_loader,
        schedule_loader=schedule_loader,
    )

    old_digest = sha256_file(args.out)
    payload = _validated_parquet_bytes(advisory)
    new_digest = _digest(payload)
    changed = old_digest != new_digest
    print(
        f"starter advisory: weeks {weeks} -> {len(advisory)} rows; "
        f"sha256 {old_digest or 'missing'} -> {new_digest}"
    )

    if args.write and changed:
        _atomic_replace(payload, args.out)
        print("write complete")
    elif args.write:
        print("write skipped: artifact unchanged")
    else:
        print("dry-run: no artifact changed")


if __name__ == "__main__":
    main()
