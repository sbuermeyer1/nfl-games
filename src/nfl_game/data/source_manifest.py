"""Fail-closed contracts for Ridge-v2 source snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd


class SourceContractError(ValueError):
    """Raised when a source does not meet the Ridge-v2 data contract."""


@dataclass(frozen=True)
class SourceSnapshot:
    name: str
    seasons: tuple[int, ...]
    retrieved_at: datetime
    schema_sha256: str
    rows: int
    coverage: dict[str, float]
    latest_event_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SourceContractError("source snapshot name must not be blank")
        if not isinstance(self.seasons, tuple) or any(
            not isinstance(season, int) or isinstance(season, bool) for season in self.seasons
        ):
            raise SourceContractError("source snapshot seasons must contain integers")
        if not isinstance(self.schema_sha256, str):
            raise SourceContractError("source snapshot schema_sha256 must be a string")
        if not isinstance(self.coverage, dict):
            raise SourceContractError("source snapshot coverage must be a dictionary")
        _require_utc(self.retrieved_at, "retrieved_at")
        if self.latest_event_at is not None:
            _require_utc(self.latest_event_at, "latest_event_at")
        if not isinstance(self.rows, int) or isinstance(self.rows, bool) or self.rows < 0:
            raise SourceContractError("source snapshot rows must be a non-negative integer")
        for column, value in self.coverage.items():
            if not isinstance(column, str) or not column:
                raise SourceContractError("source snapshot coverage columns must be non-blank strings")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
                raise SourceContractError("source snapshot coverage must contain finite values")
            if not 0.0 <= value <= 1.0:
                raise SourceContractError("source snapshot coverage values must be between zero and one")


def _require_utc(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise SourceContractError(f"{field} must be UTC")


def schema_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 digest of a frame's column names and dtypes."""
    pairs = sorted((name, str(dtype)) for name, dtype in frame.dtypes.items())
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def numeric_coverage(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Measure usable numeric values while rejecting corrupt non-null values."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SourceContractError(f"missing source columns: {missing}")

    coverage: dict[str, float] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & numeric.isna()
        if invalid.any() or not np.isfinite(numeric.dropna()).all():
            raise SourceContractError(f"non-numeric or non-finite values in {column}")
        coverage[column] = float(numeric.notna().mean()) if len(frame) else 0.0
    return coverage


def require_coverage(
    frame: pd.DataFrame, columns: Sequence[str], minimum: float = 0.90
) -> None:
    """Raise when any required source column has insufficient numeric coverage."""
    coverage = numeric_coverage(frame, columns)
    below = {name: value for name, value in coverage.items() if value < minimum}
    if below:
        formatted = ", ".join(f"{name!r}: {value:.4f}" for name, value in below.items())
        raise SourceContractError(f"coverage below {minimum:.4f}: {{{formatted}}}")


def write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    """Atomically replace *path* with canonical JSON, preserving an existing file on failure."""
    staged = path.with_suffix(path.suffix + ".tmp")
    try:
        staged.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        staged.replace(path)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def read_source_manifest(path: Path) -> tuple[SourceSnapshot, ...]:
    """Read canonical source snapshots, rejecting malformed or duplicate records."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"unable to read source manifest: {path}") from exc

    if not isinstance(payload, dict) or set(payload) != {"snapshots"}:
        raise SourceContractError("source manifest must contain only a snapshots list")
    records = payload["snapshots"]
    if not isinstance(records, list):
        raise SourceContractError("source manifest snapshots must be a list")

    snapshots = tuple(_snapshot_from_payload(record) for record in records)
    names = [item.name for item in snapshots]
    if len(names) != len(set(names)):
        raise SourceContractError("duplicate source snapshot names")
    return snapshots


def _snapshot_from_payload(record: object) -> SourceSnapshot:
    fields = {
        "name",
        "seasons",
        "retrieved_at",
        "schema_sha256",
        "rows",
        "coverage",
        "latest_event_at",
    }
    if not isinstance(record, dict) or set(record) != fields:
        raise SourceContractError("source manifest snapshot fields are invalid")
    try:
        retrieved_at = datetime.fromisoformat(record["retrieved_at"])
        latest_value = record["latest_event_at"]
        latest_event_at = datetime.fromisoformat(latest_value) if latest_value is not None else None
        seasons = tuple(record["seasons"])
        coverage = dict(record["coverage"])
    except (TypeError, ValueError) as exc:
        raise SourceContractError("source manifest snapshot values are invalid") from exc
    return SourceSnapshot(
        name=record["name"],
        seasons=seasons,
        retrieved_at=retrieved_at,
        schema_sha256=record["schema_sha256"],
        rows=record["rows"],
        coverage=coverage,
        latest_event_at=latest_event_at,
    )
