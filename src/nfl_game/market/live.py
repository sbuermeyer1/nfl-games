import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pandas as pd

from nfl_game.data.nfl import load_schedules
from nfl_game.data.schedule import ScheduleSchemaError, normalize_schedule
from nfl_game.data.teams import normalize_team_codes


class MarketUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketSnapshot:
    rows: pd.DataFrame
    observed_at: datetime
    source: str = "nflverse"
    stale: bool = False


def _validate_market_rows(rows: pd.DataFrame) -> None:
    if rows.empty:
        raise ScheduleSchemaError("schedule contains no regular-season games for season")

    identity = rows["game_id"].astype("string").str.extract(
        r"^(?P<season>\d{4})_(?P<week>\d{2})_(?P<away_team>[A-Z]+)_(?P<home_team>[A-Z]+)$"
    )
    if identity.isna().any(axis=None):
        raise ScheduleSchemaError("schedule contains a team mismatch with game_id")
    identity = normalize_team_codes(identity, ["away_team", "home_team"])
    mismatched = (
        pd.to_numeric(identity["season"]).ne(rows["season"].to_numpy())
        | pd.to_numeric(identity["week"]).ne(rows["week"].to_numpy())
        | identity["away_team"].ne(rows["away_team"].to_numpy())
        | identity["home_team"].ne(rows["home_team"].to_numpy())
    )
    if mismatched.any() or rows["away_team"].eq(rows["home_team"]).any():
        raise ScheduleSchemaError("schedule contains a team mismatch with game_id")

    appearances = pd.concat(
        [
            rows[["week", "away_team"]].rename(columns={"away_team": "team"}),
            rows[["week", "home_team"]].rename(columns={"home_team": "team"}),
        ],
        ignore_index=True,
    )
    if appearances.duplicated(["week", "team"]).any():
        raise ScheduleSchemaError("schedule contains a team mismatch within a week")


class NflverseMarketProvider:
    def __init__(
        self,
        loader=load_schedules,
        clock=lambda: datetime.now(UTC),
        ttl=timedelta(minutes=5),
        timeout_seconds=5.0,
    ):
        self._loader = loader
        self._clock = clock
        self._ttl = ttl
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._latest_futures = {}
        self._snapshots = {}
        self._futures = {}

    def snapshot(self, season: int) -> MarketSnapshot:
        now = self._clock()
        with self._lock:
            cached = self._snapshots.get(season)
            if cached is not None and now - cached.observed_at < self._ttl:
                return cached
            future = self._futures.get(season)
            if future is None:
                future = self._executor.submit(self._load_snapshot, season)
                self._futures[season] = future
                self._latest_futures[season] = future

        try:
            refreshed = future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError as exc:
            return self._stale_or_raise(season, future, exc)
        except Exception as exc:  # noqa: BLE001 - the injected upstream may fail arbitrarily
            return self._stale_or_raise(season, future, exc)

        return self._store_refresh(season, future, refreshed)

    def _load_snapshot(self, season: int) -> MarketSnapshot:
        raw = self._loader([season], save=False)
        rows = normalize_schedule(raw, season)
        _validate_market_rows(rows)
        return MarketSnapshot(
            rows=rows.copy(deep=True),
            observed_at=self._clock(),
        )

    def _store_refresh(self, season, future, refreshed):
        with self._lock:
            if self._latest_futures.get(season) is not future:
                return refreshed
            if self._futures.get(season) is future:
                self._futures.pop(season)
            self._snapshots[season] = refreshed
        return refreshed

    def _stale_or_raise(self, season, future, exc):
        if future.done():
            return self._consume_completed(season, future, exc)
        with self._lock:
            completed = future.done()
            cached = self._snapshots.get(season)
        if completed:
            return self._consume_completed(season, future, exc)
        return self._fallback(cached, exc)

    def _consume_completed(self, season, future, exc):
        try:
            refreshed = future.result()
        except Exception as completed_exc:  # noqa: BLE001 - consume the upstream future
            exc = completed_exc
        else:
            return self._store_refresh(season, future, refreshed)

        with self._lock:
            if self._futures.get(season) is future:
                self._futures.pop(season)
            cached = self._snapshots.get(season)
        return self._fallback(cached, exc)

    @staticmethod
    def _fallback(cached, exc):
        if cached is not None:
            return replace(cached, rows=cached.rows.copy(deep=True), stale=True)
        raise MarketUnavailableError("market feed unavailable") from exc
