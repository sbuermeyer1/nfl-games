"""Bounded-cache live expected-starter snapshots.

Mirrors `market/live.py`'s snapshot/TTL/stale/fallback contract deliberately: the two
overlays have the same failure modes and should behave the same way under them. The TTL
is longer because depth charts publish daily at most.

This is presentation, not a model input: nothing here reaches `FEATURE_COLS`, and a
failed refresh degrades to a missing or stale advisory, never to a failed prediction.

Player stats are loaded for `[season - 1, season]` only, not the full 2016-onward
corpus a research block would train on. `qb_epa_per_db` shrinks toward the league prior
at 200 dropbacks (see `nfl_game.ratings.qb.QB_PRIOR_DROPBACKS`), so a quarterback with
little recent history lands near league average rather than at a wild value from a thin
sample. This makes `qb_change_epa` here NOT numerically identical to the Ridge-v2 C2
research block's, which trains on the full history -- that divergence is an accepted
trade-off for keeping a live request cheap, and advisory numbers must never be quoted
as research numbers.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pandas as pd

from nfl_game.data.nfl import (
    load_depth_charts,
    load_player_stats,
    load_players,
    load_schedules,
)
from nfl_game.ratings.qb import qb_week_stats
from nfl_game.ratings.starters import starter_advisory


class StartersUnavailableError(RuntimeError):
    """No advisory could be produced and no cached one exists."""


@dataclass(frozen=True)
class StarterSnapshot:
    rows: pd.DataFrame
    observed_at: datetime
    source: str = "nflverse"
    stale: bool = False


class NflverseStarterProvider:
    def __init__(
        self,
        depth_loader=load_depth_charts,
        stats_loader=load_player_stats,
        players_loader=load_players,
        schedule_loader=load_schedules,
        clock=lambda: datetime.now(UTC),
        ttl=timedelta(minutes=30),
        timeout_seconds=20.0,
    ):
        self._depth_loader = depth_loader
        self._stats_loader = stats_loader
        self._players_loader = players_loader
        self._schedule_loader = schedule_loader
        self._clock = clock
        self._ttl = ttl
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._snapshots = {}
        self._futures = {}
        self._latest_futures = {}

    def snapshot(self, season: int, week: int) -> StarterSnapshot:
        key = (int(season), int(week))
        now = self._clock()
        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None and now - cached.observed_at < self._ttl:
                return cached
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(self._load_snapshot, key)
                self._futures[key] = future
                self._latest_futures[key] = future
        try:
            refreshed = future.result(timeout=self._timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - the injected upstream may fail arbitrarily
            # FutureTimeoutError is itself an Exception, so this one clause covers both a
            # slow feed and a broken one.
            return self._stale_or_raise(key, future, exc)
        return self._store_refresh(key, future, refreshed)

    def _load_snapshot(self, key) -> StarterSnapshot:
        season, week = key
        # The prior season carries the "recent starter" for an early-season week; the
        # full corpus is deliberately not loaded behind a live request. See the plan.
        seasons = [season - 1, season]
        depth = self._depth_loader(seasons, save=False)
        stats = self._stats_loader(seasons, save=False)
        players = self._players_loader(save=False)
        schedules = self._schedule_loader(seasons, save=False)
        # A single clock read, reused for both the depth-chart cutoff and the
        # snapshot's own `observed_at`. The brief's own draft read the clock twice
        # here (once for each); besides letting the two values disagree under a
        # fast-moving clock, it also breaks against an iterator-based fake clock in
        # tests: the flaky-reload test's fixed sequence of clock values only lines up
        # with "one now-read per snapshot() call, plus one more only when an actual
        # load is attempted" -- two reads per load consumes values meant for the next
        # call's TTL check.
        observed_at = self._clock()
        rows = starter_advisory(
            qb_week_stats(stats),
            depth,
            schedules,
            players,
            [(season, week)],
            cutoff=observed_at,
        )
        return StarterSnapshot(rows=rows.copy(deep=True), observed_at=observed_at)

    def _store_refresh(self, key, future, refreshed):
        with self._lock:
            if self._latest_futures.get(key) is not future:
                return refreshed
            if self._futures.get(key) is future:
                self._futures.pop(key)
            self._snapshots[key] = refreshed
        return refreshed

    def _stale_or_raise(self, key, future, exc):
        # Deliberate divergence from market/live.py: that provider has a
        # `_consume_completed` path that adopts a future which finished just after the
        # timeout. Here the TTL is 30 minutes against a 20-second timeout, so discarding
        # a late result costs at most one refresh cycle and is not worth the extra state.
        with self._lock:
            if self._futures.get(key) is future:
                self._futures.pop(key)
            cached = self._snapshots.get(key)
        if cached is not None:
            return replace(cached, rows=cached.rows.copy(deep=True), stale=True)
        raise StartersUnavailableError("expected-starter feed unavailable") from exc
