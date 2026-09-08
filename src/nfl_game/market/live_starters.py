"""Bounded-cache live expected-starter snapshots.

Mirrors `market/live.py`'s snapshot/TTL/stale/fallback contract deliberately: the two
overlays have the same failure modes and should behave the same way under them. The TTL
is longer because depth charts publish daily at most. The one deliberate divergence is
late-result adoption on timeout -- see `_stale_or_raise` below; in-flight de-duplication
itself is mirrored, not diverged.

This is presentation, not a model input: nothing here reaches `FEATURE_COLS`, and a
failed refresh degrades to a missing or stale advisory, never to a failed prediction.

Player stats are loaded for `[season - 1, season]` only, not the full 2016-onward
corpus a research block would train on -- one request PER SEASON in that pair (see
`_load_stats_per_season`), not a single combined request, because `nflreadpy` does not
publish the current season's stats file until games have been played; an unpublished
season is skipped rather than failing the whole load. `qb_epa_per_db` shrinks toward the
league prior at 200 dropbacks (see `nfl_game.ratings.qb.QB_PRIOR_DROPBACKS`), so a
quarterback with little recent history lands near league average rather than at a wild
value from a thin sample. This makes `qb_change_epa` here NOT numerically identical to
the Ridge-v2 C2 research block's, which trains on the full history -- that divergence is
an accepted trade-off for keeping a live request cheap, and advisory numbers must never
be quoted as research numbers.

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
from nfl_game.ratings.qb import normalize_depth_chart_history, qb_week_stats
from nfl_game.ratings.starters import ADVISORY_COLS, starter_advisory


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

    `depth` is NOT the raw depth-chart feed. The advisory only ever reads quarterback
    rows (`qb_features_for_targets` immediately reduces its `depth_history` argument to
    QB rows via `normalize_depth_chart_history`), but the raw feed carries every
    position -- measured at 1,059,637 rows / 158.6 MB on live 2025+2026 data, against
    39,969 rows / 7.9 MB once reduced to QB rows. Caching the raw frame held that extra
    ~151 MB for the full 30-minute TTL on a 512 MB dyno, so it is normalized and
    filtered to QB rows here, once per season load, and the raw frame is dropped
    immediately after. `qb_features_for_targets` re-normalizes whatever it is handed
    regardless (its `depth_history` parameter is documented as the raw feed), but
    `normalize_depth_charts` tolerates an already-normalized, already-filtered frame --
    see `_coalesce_optional` and the comment on `normalize_depth_chart_history` -- and
    re-deriving costs single-digit milliseconds on this already-small frame, not the
    36 seconds the module docstring measures on the full raw feed.
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
        # Must return a timezone-aware datetime. A naive one reaches `_cutoff_for` in
        # qb.py as the depth-chart cutoff, which raises there -- a failure this
        # provider still degrades soft on, but from deep and unobviously inside qb.py
        # rather than here.
        clock=lambda: datetime.now(UTC),
        ttl=timedelta(minutes=30),
        # Measured cold load on a dev machine: depth charts (2025+2026) 4.85s, stats
        # 2026 (404 path) 0.02s, stats 2025 1.46s, players 0.49s, schedules 0.16s --
        # total ~6.98s. This is NOT the market provider's 5.0s: that provider loads
        # one schedule feed, this one loads four, and depth-chart parsing alone is
        # ~4.85s of that -- already over budget at 5.0. 15.0 clears the measured
        # figure with headroom for a slower network or a cold Render dyno, while
        # staying well under the original 20.0 that a review found could block the
        # web dashboard's primary content (fixed instead by fetching this AFTER
        # `_bundle(...)` in web/service.py -- see that fix; do not revert it).
        timeout_seconds=15.0,
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
        schedules = self._schedule_loader(seasons, save=False)
        # Reduce the raw depth feed to QB rows immediately, before it is cached -- see
        # the `_SeasonFrames.depth` docstring. The raw frame is not kept anywhere past
        # this line, so it is released once this call returns rather than held for the
        # full TTL.
        depth_raw = self._depth_loader(seasons, save=False)
        frames = _SeasonFrames(
            depth=normalize_depth_chart_history(depth_raw, schedules),
            stats=self._load_stats_per_season(seasons),
            players=self._players_loader(save=False),
            schedules=schedules,
            loaded_at=now,
        )
        self._season_cache[season] = frames
        return frames

    def _load_stats_per_season(self, seasons: list[int]) -> pd.DataFrame:
        """Load player stats one season at a time, tolerating an unpublished season.

        `nflreadpy` publishes `stats_player_week_YYYY.parquet` only once that
        season's games have been played, so a single combined `[season - 1,
        season]` request 404s for the whole preseason and week 1 -- exactly when a
        starter advisory matters most (see the module docstring). Loading per
        season and skipping only the one that 404s keeps the advisory alive on
        whichever seasons DID publish.

        Only `ConnectionError` -- the measured 404 shape -- is tolerated per
        season. Anything else propagates so a genuine outage still fails rather
        than silently serving a half-loaded frame. If every season fails, the last
        `ConnectionError` propagates, which `snapshot()`'s existing handling turns
        into a stale cache or `StartersUnavailableError`, same as any other feed
        failure.
        """
        frames = []
        error: ConnectionError | None = None
        for season in seasons:
            try:
                frames.append(self._stats_loader([season], save=False))
            except ConnectionError as exc:
                error = exc
        if not frames:
            raise error
        return pd.concat(frames, ignore_index=True)

    def _load_snapshot(self, key, now: datetime) -> StarterSnapshot:
        season, week = key
        # The prior season carries the "recent starter" for an early-season week; the
        # full corpus is deliberately not loaded behind a live request. See the
        # module docstring above.
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
        # Cheap shape assertion: a malformed advisory frame (e.g. from a cached row
        # source with a stale schema) would otherwise surface as a KeyError inside a
        # request rather than degrading softly. Raising here puts it inside this
        # method's own error path -- caught by snapshot()'s broad except -- so it
        # degrades to a stale cache or a missing (n/a) advisory like any other
        # provider failure, instead of 500ing the slate request.
        if set(rows.columns) != set(ADVISORY_COLS):
            raise ValueError(f"advisory rows have unexpected columns: {sorted(rows.columns)}")
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
