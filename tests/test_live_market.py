from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event, current_thread

import numpy as np
import pandas as pd
import pytest

from nfl_game.market.live import MarketSnapshot, MarketUnavailableError, NflverseMarketProvider


class FakeClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


@pytest.fixture
def schedule_fixture():
    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_LA_SF",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "away_team": "LA",
                "home_team": "SF",
                "result": np.nan,
                "total": np.nan,
                "spread_line": 2.5,
                "total_line": 45.5,
            },
            {
                "game_id": "2026_01_BUF_NYJ",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "away_team": "BUF",
                "home_team": "NYJ",
                "result": np.nan,
                "total": np.nan,
                "spread_line": -3.0,
                "total_line": 46.0,
            },
        ]
    )


def test_snapshot_reuses_one_observation_for_exactly_five_minutes(schedule_fixture):
    calls = []
    clock = FakeClock(datetime(2026, 9, 1, tzinfo=UTC))
    provider = NflverseMarketProvider(
        loader=lambda seasons, save=False: calls.append(seasons) or schedule_fixture,
        clock=clock,
        ttl=timedelta(minutes=5),
        timeout_seconds=0.2,
    )

    first = provider.snapshot(2026)
    clock.advance(minutes=4, seconds=59)
    second = provider.snapshot(2026)
    clock.advance(seconds=1)
    third = provider.snapshot(2026)

    assert calls == [[2026], [2026]]
    assert second is first
    assert third is not first
    assert third.observed_at == datetime(2026, 9, 1, 0, 5, tzinfo=UTC)


def test_concurrent_cold_requests_share_one_loader_call(schedule_fixture):
    started = Event()
    release = Event()
    calls = []

    def loader(seasons, save=False):
        calls.append(seasons)
        started.set()
        release.wait(timeout=1)
        return schedule_fixture

    provider = NflverseMarketProvider(loader=loader, timeout_seconds=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(provider.snapshot, 2026) for _ in range(8)]
        assert started.wait(timeout=1)
        release.set()
        snapshots = [future.result() for future in futures]

    assert calls == [[2026]]
    assert all(snapshot.observed_at == snapshots[0].observed_at for snapshot in snapshots)


def test_delayed_old_waiter_cannot_overwrite_a_newer_refresh(schedule_fixture):
    clock = FakeClock(datetime(2026, 9, 1, tzinfo=UTC))
    old_store_ready = Event()
    release_old_store = Event()
    calls = []

    def loader(seasons, save=False):
        calls.append(seasons)
        return schedule_fixture.assign(spread_line=float(len(calls)))

    class InterleavingProvider(NflverseMarketProvider):
        def _store_refresh(self, season, future, refreshed):
            if current_thread().name.startswith("delayed-old"):
                old_store_ready.set()
                release_old_store.wait(timeout=1)
            return super()._store_refresh(season, future, refreshed)

    provider = InterleavingProvider(loader=loader, clock=clock, timeout_seconds=1)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="delayed-old") as pool:
        delayed_old = pool.submit(provider.snapshot, 2026)
        assert old_store_ready.wait(timeout=1)

        same_old = provider.snapshot(2026)
        clock.advance(minutes=6)
        new_result = provider.snapshot(2026)
        release_old_store.set()
        assert delayed_old.result().rows.loc[0, "spread_line"] == 1.0

    final_cache = provider.snapshot(2026)
    assert same_old.rows.loc[0, "spread_line"] == 1.0
    assert new_result.rows.loc[0, "spread_line"] == 2.0
    assert final_cache.rows.loc[0, "spread_line"] == 2.0
    assert calls == [[2026], [2026]]


def test_cold_timeout_keeps_future_registered_for_later_consumption(schedule_fixture):
    started = Event()
    completed = Event()
    release = Event()
    calls = []

    def loader(seasons, save=False):
        calls.append(seasons)
        started.set()
        release.wait(timeout=1)
        return schedule_fixture

    provider = NflverseMarketProvider(loader=loader, timeout_seconds=0.01)
    with pytest.raises(MarketUnavailableError, match="market feed unavailable"):
        provider.snapshot(2026)
    assert started.wait(timeout=1)
    registered = provider._futures[2026]
    registered.add_done_callback(lambda _future: completed.set())

    release.set()
    assert completed.wait(timeout=1)
    assert provider._futures[2026] is registered
    snapshot = provider.snapshot(2026)

    assert calls == [[2026]]
    assert snapshot.stale is False


def test_future_completing_at_timeout_boundary_is_consumed(schedule_fixture):
    snapshot = MarketSnapshot(
        rows=schedule_fixture,
        observed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    class BoundaryFuture:
        done_calls = 0

        def result(self, timeout=None):
            if timeout is not None:
                raise TimeoutError
            return snapshot

        def done(self):
            self.done_calls += 1
            return self.done_calls >= 2

    future = BoundaryFuture()

    class OneFutureExecutor:
        def submit(self, function, season):
            return future

    provider = NflverseMarketProvider(loader=lambda seasons, save=False: schedule_fixture)
    provider._executor = OneFutureExecutor()

    assert provider.snapshot(2026) is snapshot
    assert provider.snapshot(2026) is snapshot


def test_timeout_returns_stale_cache_without_overwriting_it(schedule_fixture):
    clock = FakeClock(datetime(2026, 9, 1, tzinfo=UTC))
    refresh_started = Event()
    refresh_completed = Event()
    release = Event()
    calls = []

    def loader(seasons, save=False):
        calls.append(seasons)
        if len(calls) == 1:
            return schedule_fixture
        refresh_started.set()
        release.wait(timeout=1)
        return schedule_fixture.assign(spread_line=7.5)

    provider = NflverseMarketProvider(
        loader=loader,
        clock=clock,
        timeout_seconds=1,
    )
    first = provider.snapshot(2026)
    clock.advance(minutes=6)
    provider._timeout_seconds = 0.01

    stale = provider.snapshot(2026)
    assert refresh_started.is_set()
    provider._futures[2026].add_done_callback(lambda _future: refresh_completed.set())
    release.set()
    assert refresh_completed.wait(timeout=1)
    refreshed = provider.snapshot(2026)

    assert stale.stale is True
    assert stale is not first
    assert first.stale is False
    assert stale.rows is not first.rows
    assert stale.rows["spread_line"].tolist() == [2.5, -3.0]
    assert refreshed.stale is False
    assert refreshed.rows["spread_line"].tolist() == [7.5, 7.5]
    assert calls == [[2026], [2026]]


def test_failed_refresh_never_replaces_valid_cache(schedule_fixture):
    clock = FakeClock(datetime(2026, 9, 1, tzinfo=UTC))
    outcomes = [schedule_fixture, RuntimeError("upstream failed"), schedule_fixture]

    def loader(seasons, save=False):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    provider = NflverseMarketProvider(loader=loader, clock=clock, timeout_seconds=0.2)
    first = provider.snapshot(2026)
    clock.advance(minutes=6)

    stale = provider.snapshot(2026)
    recovered = provider.snapshot(2026)

    assert stale.stale is True
    assert stale.rows.equals(first.rows)
    assert recovered.stale is False
    assert recovered.observed_at == clock.now
    assert outcomes == []


def test_successful_snapshot_normalizes_teams_and_kickoffs(schedule_fixture):
    observed_at = datetime(2026, 9, 1, tzinfo=UTC)
    snapshot = NflverseMarketProvider(
        loader=lambda seasons, save=False: schedule_fixture,
        clock=lambda: observed_at,
    ).snapshot(2026)

    assert snapshot.observed_at == observed_at
    assert snapshot.source == "nflverse"
    assert snapshot.stale is False
    assert snapshot.rows.loc[0, "away_team"] == "LAR"
    assert str(snapshot.rows["kickoff_at"].dtype) == "datetime64[ns, UTC]"
    assert snapshot.rows.loc[1, "kickoff_at"] == pd.Timestamp("2026-09-13T17:00:00Z")


@pytest.mark.parametrize(
    ("mutate", "expected_cause"),
    [
        (
            lambda rows: pd.concat([rows, rows.iloc[[0]]], ignore_index=True),
            "duplicate game_id",
        ),
        (
            lambda rows: rows.assign(home_team=rows["home_team"].where(rows.index != 0, "DAL")),
            "team mismatch",
        ),
        (
            lambda rows: rows.assign(
                gametime=rows["gametime"].where(rows.index != 1, "not-a-time")
            ),
            "invalid kickoff",
        ),
        (
            lambda rows: rows.assign(
                spread_line=rows["spread_line"].where(rows.index != 0, float("inf"))
            ),
            "infinite values",
        ),
    ],
)
def test_invalid_feed_rejects_the_refresh(schedule_fixture, mutate, expected_cause):
    provider = NflverseMarketProvider(
        loader=lambda seasons, save=False: mutate(schedule_fixture.copy()),
    )

    with pytest.raises(MarketUnavailableError, match="market feed unavailable") as error:
        provider.snapshot(2026)

    assert expected_cause in str(error.value.__cause__)


def test_feed_with_no_regular_season_rows_rejects_the_refresh(schedule_fixture):
    preseason = schedule_fixture.assign(game_type="PRE")
    provider = NflverseMarketProvider(loader=lambda seasons, save=False: preseason)

    with pytest.raises(MarketUnavailableError, match="market feed unavailable") as error:
        provider.snapshot(2026)

    assert "no regular-season games" in str(error.value.__cause__)


def test_null_spread_does_not_suppress_valid_total(schedule_fixture):
    rows = schedule_fixture.copy()
    rows.loc[0, "spread_line"] = np.nan
    snapshot = NflverseMarketProvider(
        loader=lambda seasons, save=False: rows,
    ).snapshot(2026)

    game = snapshot.rows.loc[snapshot.rows["game_id"].eq("2026_01_LA_SF")].iloc[0]
    assert pd.isna(game["spread_line"])
    assert game["total_line"] == 45.5
