"""Backfill early betting lines for 2021-2025 from the nflverse/nfldata git history.

One snapshot per (season, week), taken `--days-before` ahead of that week's first kickoff, so
the snapshot precedes every game in the week. Results are cached per week under
`data/raw/line_history/`, so the run is resumable and re-runs cost nothing.

The GitHub commits API allows 60 requests/hour unauthenticated and 5,000 with a token. Set
GITHUB_TOKEN to make this take minutes instead of hours; without one the script paces itself
against the published limit rather than getting itself rate-limited.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from nfl_game.data.line_history import collect_line_history
from nfl_game.data.nfl import load_schedules
from nfl_game.data.schedule import normalize_schedule
from nfl_game.paths import RAW_DIR

CACHE_DIR = RAW_DIR / "line_history"
DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025)
RATE_LIMIT_URL = "https://api.github.com/rate_limit"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2021-2025")
    parser.add_argument(
        "--days-before",
        type=int,
        default=4,
        help="snapshot lead time; 4 gave full coverage where 6 lost unposted games",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    return parser


def _rate_limit(token: str | None) -> tuple[int, float]:
    """Remaining core-API calls and the reset epoch. Querying this does not consume quota."""
    headers = {"User-Agent": "nfl-game-model-line-history"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(RATE_LIMIT_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    core = payload["resources"]["core"]
    return int(core["remaining"]), float(core["reset"])


def _wait_for_quota(token: str | None, *, need: int = 2) -> None:
    remaining, reset = _rate_limit(token)
    if remaining >= need:
        return
    delay = max(0.0, reset - time.time()) + 5
    print(f"  rate limit exhausted ({remaining} left); sleeping {delay / 60:.1f} min", flush=True)
    time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    low, _, high = args.seasons.partition("-")
    seasons = list(range(int(low), int(high or low) + 1))
    token = os.environ.get("GITHUB_TOKEN") or None
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"seasons {seasons[0]}-{seasons[-1]}, days_before={args.days_before}, "
        f"token={'yes' if token else 'no'}",
        flush=True,
    )

    raw = load_schedules()
    schedule = pd.concat(
        [normalize_schedule(raw.loc[raw["season"].eq(season)], season) for season in seasons],
        ignore_index=True,
    )
    weeks = sorted({(int(s), int(w)) for s, w in zip(schedule["season"], schedule["week"])})
    print(f"{len(weeks)} (season, week) snapshots to collect", flush=True)

    done = 0
    for season, week in weeks:
        target = cache_dir / f"line_history_{season}_wk{week:02d}_d{args.days_before:02d}.parquet"
        if target.exists():
            done += 1
            continue
        _wait_for_quota(token)
        subset = schedule.loc[schedule["season"].eq(season) & schedule["week"].eq(week)]
        frame = collect_line_history(subset, days_before=args.days_before, token=token)
        frame.to_parquet(target, index=False)
        done += 1
        priced = int(frame["early_spread_line"].notna().sum())
        print(
            f"  [{done}/{len(weeks)}] {season} wk{week:02d}: {priced}/{len(frame)} priced "
            f"@ {frame['snapshot_at'].iloc[0]:%Y-%m-%d %H:%M} UTC",
            flush=True,
        )

    # Only this run's lead time, and never the combined file, which lives in this directory
    # and would otherwise be concatenated into its own successor. A snapshot taken 6 days out
    # is a different measurement from one taken 4 days out, so the two never share a cache.
    parts = sorted(cache_dir.glob(f"line_history_*_wk*_d{args.days_before:02d}.parquet"))
    combined = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    out_path = cache_dir / f"line_history_combined_d{args.days_before:02d}.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\ncombined {len(parts)} weeks -> {len(combined)} games at {out_path}")
    print(f"priced: {int(combined['early_spread_line'].notna().sum())}/{len(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
