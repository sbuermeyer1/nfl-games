from datetime import UTC, datetime

import pandas as pd
import pytest

from nfl_game.market.live_starters import (
    NflverseStarterProvider,
    StarterSnapshot,
    StartersUnavailableError,
)


def _loaders():
    schedules = pd.DataFrame(
        {
            "game_id": ["2025_05_CIN_BAL"],
            "season": [2025],
            "week": [5],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2025-10-05 17:00", tz="UTC")],
        }
    )
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "CIN"],
            "gsis_id": ["HUNT", "BURROW"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1.0, 1.0],
            "dt": pd.to_datetime(["2025-10-04 12:00"] * 2, utc=True),
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [4, 4],
            "team": ["BAL", "CIN"],
            "player_id": ["LAMAR", "BURROW"],
            "position": ["QB", "QB"],
            "season_type": ["REG", "REG"],
            "attempts": [30.0, 30.0],
            "sacks_suffered": [2.0, 2.0],
            "passing_epa": [12.0, 8.0],
            "passing_cpoe": [3.0, 1.0],
            "passing_interceptions": [0.0, 1.0],
        }
    )
    players = pd.DataFrame(
        {
            "gsis_id": ["LAMAR", "HUNT", "BURROW"],
            "display_name": ["Lamar Jackson", "Tyler Huntley", "Joe Burrow"],
        }
    )
    return schedules, depth, stats, players


def _provider(clock=None, **overrides):
    schedules, depth, stats, players = _loaders()
    kwargs = {
        "depth_loader": lambda seasons, save=False: depth,
        "stats_loader": lambda seasons, save=False: stats,
        "players_loader": lambda save=False: players,
        "schedule_loader": lambda seasons=None, save=False: schedules,
        "clock": clock or (lambda: datetime(2025, 10, 4, 13, tzinfo=UTC)),
    }
    kwargs.update(overrides)
    return NflverseStarterProvider(**kwargs)


def test_snapshot_returns_the_advisory_rows():
    snap = _provider().snapshot(2025, 5)
    assert isinstance(snap, StarterSnapshot)
    assert snap.stale is False
    assert snap.rows.iloc[0]["home_qb"] == "Tyler Huntley"
    assert snap.rows.iloc[0]["qb_watch"] == 1


def test_second_call_within_the_ttl_does_not_reload():
    calls = []

    def counting_depth(seasons, save=False):
        calls.append(seasons)
        return _loaders()[1]

    provider = _provider(depth_loader=counting_depth)
    provider.snapshot(2025, 5)
    provider.snapshot(2025, 5)
    assert len(calls) == 1


def test_a_failing_load_with_no_cache_raises():
    def boom(seasons, save=False):
        raise RuntimeError("feed down")

    with pytest.raises(StartersUnavailableError):
        _provider(depth_loader=boom).snapshot(2025, 5)


def test_a_failing_load_after_a_good_one_returns_the_cache_marked_stale():
    state = {"fail": False}

    def flaky(seasons, save=False):
        if state["fail"]:
            raise RuntimeError("feed down")
        return _loaders()[1]

    times = iter(
        [
            datetime(2025, 10, 4, 13, tzinfo=UTC),
            datetime(2025, 10, 4, 13, tzinfo=UTC),
            datetime(2025, 10, 4, 23, tzinfo=UTC),
        ]
    )
    last = {"t": datetime(2025, 10, 4, 13, tzinfo=UTC)}

    def clock():
        last["t"] = next(times, last["t"])
        return last["t"]

    provider = _provider(clock=clock, depth_loader=flaky)
    provider.snapshot(2025, 5)
    state["fail"] = True
    snap = provider.snapshot(2025, 5)
    assert snap.stale is True
    assert snap.rows.iloc[0]["home_qb"] == "Tyler Huntley"


def test_stats_loader_is_asked_for_the_prior_and_current_season_only():
    seen = []

    def recording_stats(seasons, save=False):
        seen.append(list(seasons))
        return _loaders()[2]

    _provider(stats_loader=recording_stats).snapshot(2025, 5)
    assert seen == [[2024, 2025]]
