"""Reconstruct betting-line history from the nflverse/nfldata git history.

`nflreadpy` exposes only one row per game, holding the *current* line -- which for a finished
game is the closing number. There is no opening line in it. But `data/games.csv` in the
nflverse/nfldata repository is committed roughly every 10-30 minutes, and each commit captures
the line as it stood at that moment. Fetching that file at a past commit therefore recovers the
line that was actually available to bet, days before kickoff.

Two rules keep this honest, because both failure modes would manufacture an edge rather than
measure one:

* **A game that had already been played at snapshot time is refused.** Its `spread_line` is the
  closing number, so treating it as an early line would report zero movement and bias any
  closing-line-value estimate toward zero.
* **A game with no line posted yet comes back NaN, never 0.** A zero would read as a pick'em and
  fabricate a large edge on precisely the games the market had not priced.
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

NFLDATA_REPO = "nflverse/nfldata"
GAMES_CSV_PATH = "data/games.csv"
_COMMITS_API = "https://api.github.com/repos/{repo}/commits"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"

_NUMERIC_COLUMNS = (
    "spread_line",
    "total_line",
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
    "under_odds",
    "over_odds",
    "result",
    "total",
)

Fetcher = Callable[[str], bytes]


def _default_fetch(token: str | None = None) -> Fetcher:
    def fetch(url: str) -> bytes:
        headers = {"User-Agent": "nfl-game-model-line-history"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    return fetch


def commit_at(
    timestamp: pd.Timestamp,
    *,
    token: str | None = None,
    fetch: Fetcher | None = None,
) -> str:
    """The newest commit to games.csv at or before `timestamp`."""
    fetch = fetch or _default_fetch(token)
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    query = urllib.parse.urlencode(
        {
            "until": stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "per_page": 1,
        }
    )
    url = f"{_COMMITS_API.format(repo=NFLDATA_REPO)}?{query}"
    payload: Any = json.loads(fetch(url))
    if not payload:
        raise ValueError(f"no nflverse commit at or before {stamp.isoformat()}")
    return str(payload[0]["sha"])


def games_at(sha: str, *, token: str | None = None, fetch: Fetcher | None = None) -> pd.DataFrame:
    """`data/games.csv` exactly as it stood at one commit."""
    fetch = fetch or _default_fetch(token)
    url = _RAW_URL.format(repo=NFLDATA_REPO, sha=sha, path=GAMES_CSV_PATH)
    frame = pd.read_csv(io.BytesIO(fetch(url)), low_memory=False)
    for column in _NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def snapshot_timestamps(
    schedule: pd.DataFrame,
    *,
    days_before: int = 5,
) -> dict[tuple[int, int], pd.Timestamp]:
    """One snapshot time per (season, week), `days_before` ahead of that week's first kickoff.

    Anchoring to the week's earliest kickoff guarantees the snapshot precedes every game in the
    week, so no game in it can already have been played.
    """
    required = {"season", "week", "kickoff_at"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError(f"schedule is missing required column(s) {missing}")
    frame = schedule.copy()
    frame["_kickoff"] = pd.to_datetime(frame["kickoff_at"], utc=True, errors="coerce")
    if frame["_kickoff"].isna().any():
        raise ValueError("schedule contains unparseable kickoff_at values")
    offset = pd.Timedelta(days=days_before)
    grouped = frame.groupby(["season", "week"])["_kickoff"].min()
    return {
        (int(season), int(week)): kickoff - offset for (season, week), kickoff in grouped.items()
    }


def line_snapshot(games: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    """The line for every game of one week that had NOT yet been played at this snapshot."""
    required = {"game_id", "season", "week", "spread_line", "total_line", "result"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise ValueError(f"games snapshot is missing required column(s) {missing}")

    frame = games.copy()
    for column in ("spread_line", "total_line", "result"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    selected = frame.loc[
        frame["season"].astype("Int64").eq(season) & frame["week"].astype("Int64").eq(week)
    ]
    # A game with a result was already played: its line is the close, not an early number.
    unplayed = selected.loc[selected["result"].isna()]
    return pd.DataFrame(
        {
            "game_id": unplayed["game_id"].to_numpy(),
            "early_spread_line": unplayed["spread_line"].to_numpy(dtype=float),
            "early_total_line": unplayed["total_line"].to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


def collect_line_history(
    schedule: pd.DataFrame,
    *,
    days_before: int = 5,
    token: str | None = None,
    fetch: Fetcher | None = None,
    observer: Callable[[Mapping[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Early lines for every (season, week) in `schedule`, one snapshot per week."""
    stamps = snapshot_timestamps(schedule, days_before=days_before)
    frames: list[pd.DataFrame] = []
    for (season, week), stamp in sorted(stamps.items()):
        sha = commit_at(stamp, token=token, fetch=fetch)
        snapshot = line_snapshot(games_at(sha, token=token, fetch=fetch), season=season, week=week)
        snapshot["season"] = season
        snapshot["week"] = week
        snapshot["snapshot_at"] = stamp
        snapshot["snapshot_sha"] = sha
        frames.append(snapshot)
        if observer is not None:
            observer(
                {
                    "season": season,
                    "week": week,
                    "snapshot_at": stamp,
                    "sha": sha,
                    "games": len(snapshot),
                    "with_line": int(snapshot["early_spread_line"].notna().sum()),
                }
            )
    if not frames:
        return pd.DataFrame(
            columns=[
                "game_id",
                "early_spread_line",
                "early_total_line",
                "season",
                "week",
                "snapshot_at",
                "snapshot_sha",
            ]
        )
    return pd.concat(frames, ignore_index=True)
