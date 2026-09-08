"""Bounded-cache live expected-starter snapshots.

Mirrors `market/live.py`'s snapshot/TTL/stale/fallback contract deliberately: the two
overlays have the same failure modes and should behave the same way under them. The TTL
is longer because depth charts publish daily at most. The one deliberate divergence is
late-result adoption on timeout -- see `_stale_or_raise` below; in-flight de-duplication
itself is mirrored, not diverged.

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

Those four season-scoped feeds (depth, stats, players, schedules) are cached ONCE PER
SEASON, separately from the per-(season, week) snapshot cache above -- see
`_SeasonFrames`/`_season_frames`. Only the advisory's `targets` argument differs per
week, so without this, stepping through a season's weeks paid a full four-feed reload
on every week, serialized behind this provider's single-worker executor.
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


@dataclass(frozen=True)
class _SeasonFrames:
    """The four season-scoped feeds a snapshot is built from, cached once per season.

    Every one of depth/stats/players/schedules is loaded for `[season - 1, season]`
    (or unconditionally for players) -- only the advisory's `targets` argument differs
    per week. Without this, stepping through a season's weeks paid a full four-feed
    reload on every week, serialized behind the provider's single-worker executor.
    """

    depth: pd.DataFrame
    stats: pd.DataFrame
    players: pd.DataFrame
    schedules: pd.DataFrame
    loaded_at: datetime


class NflverseStarterProvider:
    def __init__(
        self,
        depth_loader=load_depth_charts,
        stats_loader=load_player_stats,
        players_loader=load_players,
        schedule_loader=load_schedules,
        clock=lambda: datetime.now(UTC),
        ttl=timedelta(minutes=30),
        timeout_seconds=5.0,
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
        # Season-scoped input cache, keyed by season alone -- see _SeasonFrames.
        # Touched only from the single executor worker thread (max_workers=1), so
        # concurrent _load_snapshot calls never race on it.
        self._season_cache: dict[int, _SeasonFrames] = {}

    def snapshot(self, season: int, week: int) -> StarterSnapshot:
        key = (int(season), int(week))
        now = self._clock()
        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None and now - cached.observed_at < self._ttl:
                return cached
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(self._load_snapshot, key, now)
                self._futures[key] = future
                self._latest_futures[key] = future
        try:
            refreshed = future.result(timeout=self._timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - the injected upstream may fail arbitrarily
            # FutureTimeoutError is itself an Exception, so this one clause covers both a
            # slow feed and a broken one.
            return self._stale_or_raise(key, future, exc)
        return self._store_refresh(key, future, refreshed)

    def _season_frames(self, season: int, seasons: list[int], now: datetime) -> _SeasonFrames:
        """Load (or reuse) the season-scoped feeds, cached once per season.

        Reuses the `now` already read by `snapshot()` for its own TTL check, rather
        than reading the clock again here -- see the single-clock-read note in
        `_load_snapshot` below, which this must not break.
        """
        cached = self._season_cache.get(season)
        # Deliberately reuses self._ttl -- the same TTL that bounds the per-(season,
        # week) snapshot cache in snapshot() above. test_a_failing_load_after_a_good_
        # one_returns_the_cache_marked_stale relies on this: its single 10-hour clock
        # jump must expire BOTH caches for the flaky reload to even be attempted. If
        # this season cache is ever given its own (e.g. longer) TTL, that test would
        # pass vacuously -- the stale season frames would short-circuit the reload
        # and nobody would be told the flaky-loader path went untested.
        if cached is not None and now - cached.loaded_at < self._ttl:
            return cached
        frames = _SeasonFrames(
            depth=self._depth_loader(seasons, save=False),
            stats=self._stats_loader(seasons, save=False),
            players=self._players_loader(save=False),
            schedules=self._schedule_loader(seasons, save=False),
            loaded_at=now,
        )
        self._season_cache[season] = frames
        return frames

    def _load_snapshot(self, key, now: datetime) -> StarterSnapshot:
        season, week = key
        # The prior season carries the "recent starter" for an early-season week; the
        # full corpus is deliberately not loaded behind a live request. See the plan.
        # Only `targets` below differs per week -- every one of these four feeds is
        # season-scoped, so it is loaded once per season and reused across weeks.
        seasons = [season - 1, season]
        frames = self._season_frames(season, seasons, now)
        # `observed_at` must be the season frames' OWN load time, not this build's
        # clock reading. `_season_frames` can return frames loaded up to one TTL
        # earlier than this call (a cross-week request reusing the season cache --
        # see `_SeasonFrames`/`_season_frames` above), and stamping the build clock
        # here would report the snapshot as fresher than the depth chart it was
        # actually built from -- up to two TTLs stale while `stale` still reads
        # False. See
        # test_a_second_weeks_snapshot_reports_the_season_frames_actual_load_time.
        #
        # We still read the clock exactly once here (now for the depth-chart cutoff
        # only), matching the read count from before this fix: `_season_frames`
        # reuses the `now` passed in above rather than reading the clock again, so
        # this remains the only extra read per load. That count still matters -- the
        # flaky-reload test (`test_a_failing_load_after_a_good_one_returns_the_cache_
        # marked_stale`) drives an iterator-based fake clock whose fixed sequence of
        # values only lines up with "one now-read per snapshot() call, plus one more
        # only when an actual load is attempted"; a second or missing read here
        # would desync it. An earlier draft of this fix dropped this read (using
        # `frames.loaded_at` for the cutoff too) and broke exactly that test.
        cutoff = self._clock()
        observed_at = frames.loaded_at
        rows = starter_advisory(
            qb_week_stats(frames.stats),
            frames.depth,
            frames.schedules,
            frames.players,
            [(season, week)],
            cutoff=cutoff,
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
        # timeout. Here the TTL is 30 minutes against a 5-second timeout, so discarding
        # a late result costs at most one refresh cycle and is not worth the extra state.
        # De-duplication of in-flight work is NOT part of that divergence: a future that
        # is still running stays registered so concurrent callers for the same key
        # rendezvous on it instead of each queueing a redundant reload behind it.
        with self._lock:
            if future.done() and self._futures.get(key) is future:
                self._futures.pop(key)
            cached = self._snapshots.get(key)
        if cached is not None:
            return replace(cached, rows=cached.rows.copy(deep=True), stale=True)
        raise StartersUnavailableError("expected-starter feed unavailable") from exc
