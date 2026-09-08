"""File-backed QB starter advisory provider, reading a small packaged artifact.

Satisfies the same interface `NflverseStarterProvider` (`market/live_starters.py`)
does -- `.snapshot(season, week) -> StarterSnapshot` -- so `SlateService` needs no
change to consume it; see `web/runtime.py` for how it is wired in.

This exists because the live provider downloads and parses the full nflverse
depth-chart feed (1,059,637 rows) inside the request that builds a snapshot. Measured
on the packaged app, one such snapshot peaked at 944.4 MB against a 512 MB Render
dyno, which OOM-killed the worker -- see the comment on `STARTER_ADVISORY_PATH` in
`web/runtime.py` for the full story. This provider never downloads or parses anything:
`scripts/build_starter_advisory.py` does that heavy work OFFLINE (on a GitHub Actions
runner with gigabytes of headroom, refreshed daily) and writes a tiny precomputed
parquet file; this class only ever reads that file, once, at construction.

This is presentation, not a model input, exactly like the live provider it replaces:
nothing here reaches `FEATURE_COLS`, and a failed or missing snapshot degrades to no
advisory, never to a failed prediction.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from nfl_game.market.live_starters import StarterSnapshot, StartersUnavailableError
from nfl_game.ratings.starters import ADVISORY_COLS

#: Columns the packaged artifact must carry, beyond ADVISORY_COLS itself. `season` and
#: `week` are the lookup key `.snapshot()` filters on; `computed_at` is the artifact's
#: own build time, identical on every row (see scripts/build_starter_advisory.py).
_ARTIFACT_EXTRA_COLS = ("season", "week", "computed_at")

#: The artifact rebuilds daily (.github/workflows/refresh-2026-model.yml, same cadence
#: as game_features.parquet/schedule_2026.parquet). 36 hours clears one full day's
#: cadence with room for the run to land a bit late, but flags a second consecutive
#: missed rebuild -- e.g. a snapshot from yesterday's normal run is ~24h old and stays
#: fresh; one that is two days old (a skipped run) is flagged stale rather than served
#: silently as if it were current.
STALE_AFTER = timedelta(hours=36)


class PackagedStarterProvider:
    """Serve the precomputed starter-advisory artifact from memory.

    The file is read exactly once, here in `__init__` -- not per request. The artifact
    is tiny (a handful of rows per active week), so this costs nothing at startup and
    keeps `.snapshot()` free of any I/O, matching the packaged dataset/tracker/schedule
    services this mirrors.
    """

    def __init__(self, path: str | Path, clock=lambda: datetime.now(UTC)) -> None:
        self._clock = clock
        self._rows = self._read(Path(path))

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise StartersUnavailableError(f"packaged starter advisory not found: {path}")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise StartersUnavailableError(
                f"cannot read packaged starter advisory: {path}"
            ) from exc
        expected = set(ADVISORY_COLS) | set(_ARTIFACT_EXTRA_COLS)
        if set(frame.columns) != expected:
            raise StartersUnavailableError(
                f"packaged starter advisory has unexpected columns: {sorted(frame.columns)}"
            )
        return frame

    def snapshot(self, season: int, week: int) -> StarterSnapshot:
        matching = self._rows.loc[
            self._rows["season"].eq(int(season)) & self._rows["week"].eq(int(week))
        ]
        if matching.empty:
            raise StartersUnavailableError(
                f"no packaged starter advisory for season={season} week={week}"
            )
        observed_at = self._computed_at(matching)
        stale = (self._clock() - observed_at) > STALE_AFTER
        return StarterSnapshot(
            rows=matching[list(ADVISORY_COLS)].reset_index(drop=True).copy(deep=True),
            observed_at=observed_at,
            source="packaged",
            stale=stale,
        )

    @staticmethod
    def _computed_at(rows: pd.DataFrame) -> datetime:
        stamp = pd.Timestamp(rows["computed_at"].iloc[0])
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(UTC)
        else:
            stamp = stamp.tz_convert(UTC)
        return stamp.to_pydatetime()
