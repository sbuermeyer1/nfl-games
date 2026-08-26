from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_game.data.line_history import (
    GAMES_CSV_PATH,
    commit_at,
    games_at,
    line_snapshot,
    snapshot_timestamps,
)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2024_10_ATL_NO",
                "season": 2024,
                "week": 10,
                "kickoff_at": "2024-11-10T18:00:00Z",
            },
            {
                "game_id": "2024_10_DEN_KC",
                "season": 2024,
                "week": 10,
                "kickoff_at": "2024-11-10T21:25:00Z",
            },
            {
                "game_id": "2024_10_MIA_LA",
                "season": 2024,
                "week": 10,
                "kickoff_at": "2024-11-11T01:20:00Z",
            },
            {
                "game_id": "2024_11_BUF_IND",
                "season": 2024,
                "week": 11,
                "kickoff_at": "2024-11-17T18:00:00Z",
            },
        ]
    )


def _games_csv(rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False)


def _snapshot_rows() -> list[dict[str, object]]:
    return [
        # Already final at snapshot time: its "line" is the closing number.
        {
            "game_id": "2024_09_OLD_GAME",
            "season": 2024,
            "week": 9,
            "spread_line": -3.0,
            "total_line": 44.0,
            "result": -7.0,
        },
        {
            "game_id": "2024_10_ATL_NO",
            "season": 2024,
            "week": 10,
            "spread_line": -1.0,
            "total_line": 43.5,
            "result": "",
        },
        {
            "game_id": "2024_10_DEN_KC",
            "season": 2024,
            "week": 10,
            "spread_line": 9.5,
            "total_line": 41.0,
            "result": "",
        },
        # In the requested week but no line posted yet.
        {
            "game_id": "2024_10_MIA_LA",
            "season": 2024,
            "week": 10,
            "spread_line": "",
            "total_line": "",
            "result": "",
        },
    ]


def test_snapshot_timestamp_precedes_every_kickoff_in_its_week():
    stamps = snapshot_timestamps(_schedule(), days_before=5)

    assert set(stamps) == {(2024, 10), (2024, 11)}
    schedule = _schedule()
    for (season, week), stamp in stamps.items():
        kickoffs = pd.to_datetime(
            schedule.loc[schedule["season"].eq(season) & schedule["week"].eq(week), "kickoff_at"],
            utc=True,
        )
        assert stamp < kickoffs.min()


def test_a_larger_days_before_moves_the_snapshot_earlier():
    early = snapshot_timestamps(_schedule(), days_before=7)[(2024, 10)]
    late = snapshot_timestamps(_schedule(), days_before=2)[(2024, 10)]

    assert early < late
    assert (late - early) == pd.Timedelta(days=5)


def test_games_already_played_at_snapshot_time_are_refused():
    """A finished game's spread IS the closing line; taking it as an early line is a leak."""
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())

    out = line_snapshot(frame, season=2024, week=9)

    assert out.empty


def test_unplayed_games_yield_their_line_and_an_unposted_line_is_nan_not_zero():
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())

    out = line_snapshot(frame, season=2024, week=10)

    assert list(out["game_id"]) == ["2024_10_ATL_NO", "2024_10_DEN_KC", "2024_10_MIA_LA"]
    assert out.loc[out["game_id"].eq("2024_10_ATL_NO"), "early_spread_line"].iloc[0] == -1.0
    assert out.loc[out["game_id"].eq("2024_10_DEN_KC"), "early_total_line"].iloc[0] == 41.0
    # No line posted yet must stay missing: a zero would read as a pick'em.
    assert np.isnan(out.loc[out["game_id"].eq("2024_10_MIA_LA"), "early_spread_line"].iloc[0])


def test_games_at_requests_the_versioned_path_and_parses_numerics():
    seen: dict[str, str] = {}

    def fetch(url: str) -> bytes:
        seen["url"] = url
        return _games_csv(_snapshot_rows()).encode()

    frame = games_at("abc123", fetch=fetch)

    assert "abc123" in seen["url"]
    assert GAMES_CSV_PATH in seen["url"]
    assert frame["spread_line"].dtype.kind == "f"


def test_commit_at_asks_for_the_newest_commit_at_or_before_the_timestamp():
    seen: dict[str, str] = {}

    def fetch(url: str) -> bytes:
        seen["url"] = url
        return b'[{"sha": "feedface", "commit": {"committer": {"date": "2024-11-05T12:00:00Z"}}}]'

    sha = commit_at(pd.Timestamp("2024-11-05T12:00:00Z"), fetch=fetch)

    assert sha == "feedface"
    assert (
        "until=2024-11-05T12%3A00%3A00Z" in seen["url"]
        or "until=2024-11-05T12:00:00Z" in seen["url"]
    )
    assert "per_page=1" in seen["url"]


def test_commit_at_raises_when_no_commit_exists_before_the_timestamp():
    with pytest.raises(ValueError, match="no nflverse commit"):
        commit_at(pd.Timestamp("2015-01-01T00:00:00Z"), fetch=lambda url: b"[]")
