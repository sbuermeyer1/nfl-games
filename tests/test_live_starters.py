from datetime import UTC, datetime
from threading import Event

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


def test_default_timeout_covers_the_measured_four_feed_cold_load():
    """I5(a) found the original 20.0s could let a presentation-only column add real
    latency ahead of the model; that was fixed by moving the fetch after `_bundle(...)`
    in web/service.py, not by shrinking the timeout to the market provider's 5.0 --
    this provider loads four feeds (depth, stats, players, schedules) where the market
    provider loads one schedule, and a measured cold load came in at ~6.98s, already
    over a 5.0s budget. 15.0 clears that measurement with headroom while staying well
    under the original 20.0."""
    provider = _provider()
    assert provider._timeout_seconds == 15.0


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


def test_stepping_through_weeks_in_a_season_reuses_the_loaded_frames():
    """I5(b): every loaded input in _load_snapshot is season-scoped (depth, stats,
    players, schedules for [season-1, season]) -- only `targets` differs per week.
    Stepping through weeks in the same season must not trigger a full four-feed
    reload per week. Counts all four loaders (Minor 4): counting only depth would
    still pass if stats/players/schedules caching were ever reverted while depth
    stayed cached."""
    schedules, depth, stats, players = _loaders()
    extra_week = pd.concat(
        [
            schedules,
            pd.DataFrame(
                {
                    "game_id": ["2025_06_CIN_BAL"],
                    "season": [2025],
                    "week": [6],
                    "home_team": ["BAL"],
                    "away_team": ["CIN"],
                    "kickoff_at": [pd.Timestamp("2025-10-12 17:00", tz="UTC")],
                }
            ),
        ],
        ignore_index=True,
    )
    depth_calls = []
    stats_calls = []
    players_calls = []
    schedule_calls = []

    def counting_depth(seasons, save=False):
        depth_calls.append(list(seasons))
        return depth

    def counting_stats(seasons, save=False):
        stats_calls.append(list(seasons))
        return stats

    def counting_players(save=False):
        players_calls.append(True)
        return players

    def counting_schedules(seasons=None, save=False):
        schedule_calls.append(True)
        return extra_week

    provider = _provider(
        depth_loader=counting_depth,
        stats_loader=counting_stats,
        players_loader=counting_players,
        schedule_loader=counting_schedules,
    )
    provider.snapshot(2025, 5)
    provider.snapshot(2025, 6)
    assert depth_calls == [[2024, 2025]], "the second week must reuse the cached season load"
    assert stats_calls == [[2024], [2025]], (
        "stats must also be cached across weeks, not just depth -- one call per season "
        "from the first week's load, and none added by the second week's cache hit"
    )
    assert len(players_calls) == 1, "players must also be cached across weeks, not just depth"
    assert len(schedule_calls) == 1, "schedules must also be cached across weeks, not just depth"


def test_a_second_weeks_snapshot_reports_the_season_frames_actual_load_time():
    """I5b's season cache (_season_frames) can serve depth/stats/players/schedules
    loaded up to one TTL earlier than a later week's snapshot build. `observed_at`
    must report when those feeds were actually fetched, not this build's clock --
    otherwise a cross-week request can serve a ~59-minute-old depth chart while
    reporting `stale=False` and an `observed_at` up to 30 minutes newer than the
    data it describes. A snapshot must never report a timestamp newer than the
    frames it was built from."""
    schedules, _depth, _stats, _players = _loaders()
    extra_week = pd.concat(
        [
            schedules,
            pd.DataFrame(
                {
                    "game_id": ["2025_06_CIN_BAL"],
                    "season": [2025],
                    "week": [6],
                    "home_team": ["BAL"],
                    "away_team": ["CIN"],
                    "kickoff_at": [pd.Timestamp("2025-10-12 17:00", tz="UTC")],
                }
            ),
        ],
        ignore_index=True,
    )
    loaded_at = datetime(2025, 10, 4, 13, 0, tzinfo=UTC)
    built_at = datetime(2025, 10, 4, 13, 29, tzinfo=UTC)  # within the season cache's TTL
    box = {"now": loaded_at}

    provider = _provider(
        clock=lambda: box["now"],
        schedule_loader=lambda seasons=None, save=False: extra_week,
    )
    provider.snapshot(2025, 5)

    box["now"] = built_at
    snap = provider.snapshot(2025, 6)

    assert snap.observed_at == loaded_at, (
        "a second week's snapshot must report when the season frames were actually "
        "fetched, not the later build time"
    )
    assert snap.stale is False


def test_stats_loader_is_asked_for_the_prior_and_current_season_only():
    """Stats are loaded one season at a time (see `_load_stats_per_season`), not as a
    single combined [season-1, season] request -- so both seasons must still show up
    as separate calls. Asserting each season individually (not just `len(seen) == 2`)
    means a future change that silently drops the current season once a 2026 stats
    file exists -- a regression invisible today because there is no such file to
    drop -- would still fail this test."""
    seen = []

    def recording_stats(seasons, save=False):
        seen.append(list(seasons))
        return _loaders()[2]

    _provider(stats_loader=recording_stats).snapshot(2025, 5)
    assert seen == [[2024], [2025]]


def _loaders_2026():
    """Season-2026 week-1 schedule/depth paired with season-2025 stats as the prior
    season's history -- the shape live data actually has for the whole preseason and
    week 1, before stats_player_week_2026.parquet is published."""
    schedules = pd.DataFrame(
        {
            "game_id": ["2026_01_CIN_BAL"],
            "season": [2026],
            "week": [1],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2026-09-10 17:00", tz="UTC")],
        }
    )
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "CIN"],
            "gsis_id": ["HUNT", "BURROW"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1.0, 1.0],
            "dt": pd.to_datetime(["2026-09-04 12:00"] * 2, utc=True),
        }
    )
    _schedules_2025, _depth_2025, stats, players = _loaders()
    return schedules, depth, stats, players


def test_stats_load_tolerates_an_unpublished_current_season():
    """Root-cause fix: nflreadpy only publishes stats_player_week_YYYY.parquet once
    games are played, so a combined [season-1, season] request 404s for the whole
    preseason and week 1 -- exactly when a starter advisory matters most. Loading
    per season and skipping only the missing one must still produce a real advisory
    from the prior season's rows."""
    schedules, depth, stats, _players = _loaders_2026()

    def per_season_stats(seasons, save=False):
        if 2026 in seasons:
            raise ConnectionError(
                "Failed to download stats_player_week_2026.parquet: "
                "404 Client Error: Not Found"
            )
        return stats

    provider = _provider(
        clock=lambda: datetime(2026, 9, 8, 13, tzinfo=UTC),
        depth_loader=lambda seasons, save=False: depth,
        stats_loader=per_season_stats,
        schedule_loader=lambda seasons=None, save=False: schedules,
    )
    snap = provider.snapshot(2026, 1)
    assert snap.rows.iloc[0]["home_qb"] == "Tyler Huntley"
    assert snap.rows.iloc[0]["away_qb"] == "Joe Burrow"


def test_stats_load_raises_when_no_season_is_available():
    """Fail-soft must survive: if every season's stats load fails, the advisory is
    still unavailable rather than silently serving depth charts with no history."""

    def always_fails(seasons, save=False):
        raise ConnectionError("404 Client Error: Not Found")

    with pytest.raises(StartersUnavailableError):
        _provider(stats_loader=always_fails).snapshot(2025, 5)


def test_season_cache_holds_the_qb_filtered_depth_frame_not_the_raw_feed():
    """The advisory only ever reads QB rows (see `qb_features_for_targets`), but the
    raw depth feed carries every position -- 1,059,637 rows / 158.6 MB measured on
    live 2025+2026 data, vs 39,969 rows / 7.9 MB once reduced to QB rows. Caching the
    raw frame for the full 30-minute TTL is exactly the ~150 MB this fix removes; this
    test fails if a later change reverts `_season_frames` to caching the raw feed."""
    _schedules, depth, _stats, _players = _loaders()
    non_qb_row = pd.DataFrame(
        {
            "club_code": ["BAL"],
            "gsis_id": ["HENRY"],
            "pos_abb": ["RB"],
            "pos_rank": [1.0],
            "dt": pd.to_datetime(["2025-10-04 12:00"], utc=True),
        }
    )
    depth_with_rb = pd.concat([depth, non_qb_row], ignore_index=True)
    provider = _provider(depth_loader=lambda seasons, save=False: depth_with_rb)
    frames = provider._season_frames(2025, [2024, 2025], provider._clock())
    assert list(frames.depth.columns) == ["season", "week", "team", "player_id", "rank", "dt"], (
        "the cached depth frame must be the QB-normalized shape, not the raw feed's "
        "columns (which include `position`)"
    )
    assert set(frames.depth["player_id"]) == {"HUNT", "BURROW"}, (
        "the cached depth frame must be filtered to QB rows only -- the RB row must "
        "not survive into the cache"
    )


def test_stats_load_uses_both_seasons_when_both_are_available():
    """The direction most likely to regress unnoticed: tolerating a missing season
    must not turn into silently dropping a season that IS present. Assert per-season
    calls are actually made and both years' rows land in the combined frame."""
    stats_2024 = pd.DataFrame(
        {
            "season": [2024],
            "week": [10],
            "team": ["BAL"],
            "player_id": ["LAMAR"],
            "position": ["QB"],
            "season_type": ["REG"],
            "attempts": [25.0],
            "sacks_suffered": [1.0],
            "passing_epa": [5.0],
            "passing_cpoe": [2.0],
            "passing_interceptions": [0.0],
        }
    )
    stats_2025 = pd.DataFrame(
        {
            "season": [2025],
            "week": [4],
            "team": ["BAL"],
            "player_id": ["LAMAR"],
            "position": ["QB"],
            "season_type": ["REG"],
            "attempts": [30.0],
            "sacks_suffered": [2.0],
            "passing_epa": [12.0],
            "passing_cpoe": [3.0],
            "passing_interceptions": [0.0],
        }
    )
    calls = []

    def per_season(seasons, save=False):
        calls.append(list(seasons))
        return stats_2024 if seasons == [2024] else stats_2025

    provider = _provider(stats_loader=per_season)
    frames = provider._season_frames(2025, [2024, 2025], provider._clock())
    assert calls == [[2024], [2025]], "the fix must load stats per season, not batched"
    assert sorted(frames.stats["season"].unique().tolist()) == [2024, 2025], (
        "both seasons' rows must survive once both are available"
    )


def test_cold_timeout_keeps_future_registered_for_later_consumption():
    started = Event()
    release = Event()
    completed = Event()
    calls = []

    def slow_depth(seasons, save=False):
        calls.append(list(seasons))
        started.set()
        # Bounded block: released explicitly below, with a timeout as a backstop
        # so this thread cannot hang the suite even if the assertions fail first.
        release.wait(timeout=5)
        return _loaders()[1]

    provider = _provider(depth_loader=slow_depth, timeout_seconds=0.01)
    try:
        with pytest.raises(StartersUnavailableError, match="expected-starter feed unavailable"):
            provider.snapshot(2025, 5)
        assert started.wait(timeout=1)

        key = (2025, 5)
        registered = provider._futures.get(key)
        assert registered is not None, "still-running future must stay registered on timeout"
        assert not registered.done()
        registered.add_done_callback(lambda _future: completed.set())
    finally:
        release.set()

    assert completed.wait(timeout=1)

    snapshot = provider.snapshot(2025, 5)
    assert calls == [[2024, 2025]], "a follow-up call must not submit a second load"
    assert snapshot.stale is False
