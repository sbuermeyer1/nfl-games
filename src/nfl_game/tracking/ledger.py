"""Immutable, reproducible grading for model prediction records."""

import numpy as np
import pandas as pd

HISTORICAL_MODEL_VERSION = "ridge-v1"
OFFICIAL_ESTIMATOR = "ridge"
RECORD_TYPES = frozenset({"backtest", "live"})
GRADE_VALUES = frozenset({"win", "loss", "push", "pending", "no_pick"})
PICK_VALUES = {
    "spread_pick": frozenset({"home", "away"}),
    "total_pick": frozenset({"over", "under"}),
}

LEDGER_COLUMNS = [
    "record_type",
    "model_version",
    "estimator",
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "model_margin",
    "model_total",
    "official_spread_line",
    "official_total_line",
    "published_spread_line",
    "published_total_line",
    "closing_spread_line",
    "closing_total_line",
    "published_at",
    "kickoff_at",
    "actual_margin",
    "actual_total",
    "spread_pick",
    "total_pick",
    "spread_edge",
    "total_edge",
    "spread_grade",
    "total_grade",
    "spread_clv",
    "total_clv",
    "spread_close_grade",
    "total_close_grade",
]

_DERIVED_LABEL_COLUMNS = [
    "spread_pick",
    "total_pick",
    "spread_grade",
    "total_grade",
    "spread_close_grade",
    "total_close_grade",
]
_DERIVED_NUMERIC_COLUMNS = ["spread_edge", "total_edge", "spread_clv", "total_clv"]
_NUMERIC_COLUMNS = [
    "model_margin",
    "model_total",
    "official_spread_line",
    "official_total_line",
    "published_spread_line",
    "published_total_line",
    "closing_spread_line",
    "closing_total_line",
    "actual_margin",
    "actual_total",
    *_DERIVED_NUMERIC_COLUMNS,
]
_IDENTITY_COLUMNS = ["model_version", "estimator", "game_id", "away_team", "home_team"]


def _pick(model, line, high_label, low_label):
    if pd.isna(model) or pd.isna(line) or model == line:
        return pd.NA
    return high_label if model > line else low_label


def _grade(pick, actual, line, high_label):
    if pd.isna(line) or pd.isna(actual):
        return "pending"
    if pd.isna(pick):
        return "no_pick"
    if actual == line:
        return "push"
    high_won = actual > line
    return "win" if high_won == (pick == high_label) else "loss"


def _clv(record_type, pick, published, closing, high_label):
    if record_type != "live" or pd.isna(pick) or pd.isna(published) or pd.isna(closing):
        return np.nan
    movement = closing - published
    return float(movement if pick == high_label else -movement)


def _column(frame, name):
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def grade_ledger(facts):
    """Derive frozen picks, outcome grades, and closing-line value from fact columns."""
    ledger = facts.copy()
    spread_line = _column(ledger, "official_spread_line")
    total_line = _column(ledger, "official_total_line")
    model_margin = _column(ledger, "model_margin")
    model_total = _column(ledger, "model_total")

    ledger["spread_edge"] = model_margin - spread_line
    ledger["total_edge"] = model_total - total_line
    ledger["spread_pick"] = [
        _pick(model, line, "home", "away") for model, line in zip(model_margin, spread_line)
    ]
    ledger["total_pick"] = [
        _pick(model, line, "over", "under") for model, line in zip(model_total, total_line)
    ]

    actual_margin = _column(ledger, "actual_margin")
    actual_total = _column(ledger, "actual_total")
    ledger["spread_grade"] = [
        _grade(pick, actual, line, "home")
        for pick, actual, line in zip(ledger["spread_pick"], actual_margin, spread_line)
    ]
    ledger["total_grade"] = [
        _grade(pick, actual, line, "over")
        for pick, actual, line in zip(ledger["total_pick"], actual_total, total_line)
    ]

    record_type = _column(ledger, "record_type")
    published_spread = _column(ledger, "published_spread_line")
    published_total = _column(ledger, "published_total_line")
    closing_spread = _column(ledger, "closing_spread_line")
    closing_total = _column(ledger, "closing_total_line")
    ledger["spread_clv"] = [
        _clv(kind, pick, published, closing, "home")
        for kind, pick, published, closing in zip(
            record_type, ledger["spread_pick"], published_spread, closing_spread
        )
    ]
    ledger["total_clv"] = [
        _clv(kind, pick, published, closing, "over")
        for kind, pick, published, closing in zip(
            record_type, ledger["total_pick"], published_total, closing_total
        )
    ]

    ledger["spread_close_grade"] = [
        _grade(pick, actual, line, "home") if kind == "live" else pd.NA
        for kind, pick, actual, line in zip(
            record_type, ledger["spread_pick"], actual_margin, closing_spread
        )
    ]
    ledger["total_close_grade"] = [
        _grade(pick, actual, line, "over") if kind == "live" else pd.NA
        for kind, pick, actual, line in zip(
            record_type, ledger["total_pick"], actual_total, closing_total
        )
    ]
    return ledger.reindex(columns=LEDGER_COLUMNS)


def build_backtest_ledger(predictions, model_version=HISTORICAL_MODEL_VERSION):
    """Convert historical walk-forward predictions into a validated ledger."""
    facts = predictions.rename(
        columns={"margin": "actual_margin", "total_points": "actual_total"}
    ).copy()
    facts["record_type"] = "backtest"
    facts["model_version"] = model_version
    facts["estimator"] = OFFICIAL_ESTIMATOR
    facts["official_spread_line"] = _column(facts, "spread_line")
    facts["official_total_line"] = _column(facts, "total_line")
    facts["published_spread_line"] = np.nan
    facts["published_total_line"] = np.nan
    facts["closing_spread_line"] = _column(facts, "spread_line")
    facts["closing_total_line"] = _column(facts, "total_line")
    facts["published_at"] = pd.NaT

    ledger = grade_ledger(facts)
    validate_ledger(ledger)
    return ledger


def _same_values(actual, expected):
    return actual.eq(expected) | (actual.isna() & expected.isna())


def _validate_numeric(ledger):
    for column in _NUMERIC_COLUMNS:
        values = ledger[column]
        numeric = pd.to_numeric(values, errors="coerce")
        if (values.notna() & numeric.isna()).any() or not np.isfinite(
            numeric.dropna().to_numpy(dtype=float)
        ).all():
            raise ValueError(f"non-finite numeric value in {column}")


def _validate_derived(ledger):
    fresh = grade_ledger(ledger)
    for column in _DERIVED_LABEL_COLUMNS:
        if not _same_values(ledger[column], fresh[column]).all():
            raise ValueError(f"derived {column} does not match facts")
    for column in _DERIVED_NUMERIC_COLUMNS:
        try:
            np.testing.assert_allclose(
                ledger[column], fresh[column], rtol=0.0, atol=1e-12, equal_nan=True
            )
        except AssertionError as error:
            raise ValueError(f"derived {column} does not match facts") from error


def validate_ledger(ledger):
    """Raise ValueError unless a persisted ledger is complete, valid, and reproducible."""
    missing = [column for column in LEDGER_COLUMNS if column not in ledger]
    if missing:
        raise ValueError(f"ledger missing columns: {missing}")
    if ledger.empty:
        raise ValueError("ledger is empty")

    for column in _IDENTITY_COLUMNS:
        values = ledger[column]
        if (
            values.isna().any()
            or values.map(lambda value: not isinstance(value, str) or not value.strip()).any()
        ):
            raise ValueError(f"blank identity string in {column}")

    if not ledger["record_type"].isin(RECORD_TYPES).all():
        raise ValueError("invalid record type")
    if not ledger["estimator"].eq(OFFICIAL_ESTIMATOR).all():
        raise ValueError("official ledger estimator must be ridge")
    if ledger.duplicated(["record_type", "model_version", "game_id"]).any():
        raise ValueError("duplicate ledger key")

    for column in ("season", "week"):
        values = pd.to_numeric(ledger[column], errors="coerce")
        if (
            values.isna().any()
            or not np.isfinite(values.to_numpy(dtype=float)).all()
            or not (values > 0).all()
            or not (values % 1 == 0).all()
        ):
            raise ValueError(f"{column} must be a positive whole number")

    _validate_numeric(ledger)
    for column, allowed in PICK_VALUES.items():
        if not ledger[column].dropna().isin(allowed).all():
            raise ValueError(f"invalid {column}")
    for column in ("spread_grade", "total_grade", "spread_close_grade", "total_close_grade"):
        if not ledger[column].dropna().isin(GRADE_VALUES).all():
            raise ValueError(f"invalid {column}")

    backtest = ledger["record_type"].eq("backtest")
    live = ledger["record_type"].eq("live")
    for official, closing in (
        ("official_spread_line", "closing_spread_line"),
        ("official_total_line", "closing_total_line"),
    ):
        if not _same_values(ledger.loc[backtest, official], ledger.loc[backtest, closing]).all():
            raise ValueError("backtest official and closing lines must match")
    if ledger.loc[backtest, "published_at"].notna().any():
        raise ValueError("backtest rows cannot have a published timestamp")
    for official, published in (
        ("official_spread_line", "published_spread_line"),
        ("official_total_line", "published_total_line"),
    ):
        if not _same_values(ledger.loc[live, official], ledger.loc[live, published]).all():
            raise ValueError("live official and published lines must match")

    _validate_derived(ledger)
