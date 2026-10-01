"""Read-only web boundary for the immutable performance tracker ledger."""

from numbers import Integral, Number
from pathlib import Path

import pandas as pd

from nfl_game.tracking.ledger import HISTORICAL_MODEL_VERSION, RECORD_TYPES, validate_ledger
from nfl_game.tracking.summary import (
    QUALIFIED_EDGE,
    SPREAD_EDGE_THRESHOLDS,
    TOTAL_EDGE_THRESHOLDS,
    audit_rows,
    summarize_selection,
)

#: What the official ledger may contain. Reconstructed records come from their own artifact
#: (scripts/reconstruct_outage_2026.py) and are never mixed into it.
OFFICIAL_RECORD_TYPES = frozenset({"backtest", "live"})


class TrackerInputError(ValueError):
    """A tracker record type or season selection is invalid."""


class TrackerService:
    """Validate and expose the packaged tracker ledger without mutating it."""

    def __init__(self, ledger: pd.DataFrame, reconstructed: pd.DataFrame | None = None):
        validate_ledger(ledger)
        if not ledger["record_type"].isin(OFFICIAL_RECORD_TYPES).all():
            raise ValueError("the official ledger may hold only backtest and live records")
        frames = [ledger]
        if reconstructed is not None:
            validate_ledger(reconstructed)
            if not reconstructed["record_type"].eq("reconstructed").all():
                raise ValueError("the reconstructed artifact may hold only reconstructed records")
            frames.append(reconstructed)
        self._ledger = pd.concat(frames, ignore_index=True) if len(frames) > 1 else ledger.copy()
        self._has_reconstructed = reconstructed is not None
        versions = sorted(self._ledger["model_version"].unique())
        if versions != [HISTORICAL_MODEL_VERSION]:
            raise ValueError(f"official tracker requires only {HISTORICAL_MODEL_VERSION!r}")

    @classmethod
    def from_parquet(
        cls, path: str | Path, reconstructed_path: str | Path | None = None
    ) -> "TrackerService":
        reconstructed = None if reconstructed_path is None else pd.read_parquet(reconstructed_path)
        return cls(pd.read_parquet(path), reconstructed)

    def options(self) -> dict:
        record_types = ["backtest", "live"]
        if self._has_reconstructed:
            record_types.append("reconstructed")
        options = {
            "record_types": record_types,
            "seasons": {
                record_type: sorted(
                    int(value)
                    for value in self._ledger.loc[
                        self._ledger["record_type"].eq(record_type), "season"
                    ].unique()
                )
                for record_type in record_types
            },
            "default_record_type": "backtest",
            "default_season": "all",
            "model_version": HISTORICAL_MODEL_VERSION,
            "qualified_edge": QUALIFIED_EDGE,
            "spread_edge_thresholds": list(SPREAD_EDGE_THRESHOLDS),
            "total_edge_thresholds": list(TOTAL_EDGE_THRESHOLDS),
            "live_available": bool(self._ledger["record_type"].eq("live").any()),
        }
        if self._has_reconstructed:
            options["reconstructed_available"] = True
        return options

    def _season(self, record_type: str, season: str | int) -> str | int:
        if record_type not in RECORD_TYPES:
            raise TrackerInputError("invalid record type")
        if isinstance(season, str) and season == "all":
            return "all"
        if isinstance(season, bool):
            raise TrackerInputError("season must be 'all' or a whole number")
        if isinstance(season, str):
            if not season.strip() or not season.isdecimal():
                raise TrackerInputError("season must be 'all' or a whole number")
            parsed = int(season)
        elif isinstance(season, Integral):
            parsed = int(season)
        else:
            raise TrackerInputError("season must be 'all' or a whole number")

        available = {
            int(value)
            for value in self._ledger.loc[
                self._ledger["record_type"] == record_type, "season"
            ].unique()
        }
        if parsed not in available:
            raise TrackerInputError(
                f"season {parsed} is not available for record type {record_type}"
            )
        return parsed

    def summary(self, record_type: str, season: str | int) -> dict:
        selected_season = self._season(record_type, season)
        try:
            return summarize_selection(self._ledger, record_type, selected_season)
        except ValueError as exc:
            raise TrackerInputError(str(exc)) from exc

    def records(self, record_type: str, season: str | int) -> list[dict]:
        selected_season = self._season(record_type, season)
        if selected_season == "all":
            raise TrackerInputError("records require one concrete season")
        try:
            rows = audit_rows(self._ledger, record_type, selected_season)
        except ValueError as exc:
            raise TrackerInputError(str(exc)) from exc
        return [{column: _json_value(value) for column, value in row.items()} for row in rows]


def _json_value(value):
    """Convert pandas' scalar missing values into JSON-compatible nulls."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Number):
        return float(value)
    return value
