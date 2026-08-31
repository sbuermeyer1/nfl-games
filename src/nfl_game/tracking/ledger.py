"""Immutable, reproducible grading for model prediction records."""

from numbers import Integral

import numpy as np
import pandas as pd

HISTORICAL_MODEL_VERSION = "ridge-v1"
OFFICIAL_ESTIMATOR = "ridge"
PUBLICATION_STATUSES = frozenset({"pending", "published", "excluded"})
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
    "spread_publication_status",
    "total_publication_status",
    "spread_exclusion_reason",
    "total_exclusion_reason",
    "published_spread_observed_at",
    "published_total_observed_at",
    "closing_spread_observed_at",
    "closing_total_observed_at",
    "current_kickoff_at",
    "void_reason",
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

_LIVE_ONLY_COLUMNS = [
    "spread_publication_status",
    "total_publication_status",
    "spread_exclusion_reason",
    "total_exclusion_reason",
    "published_spread_observed_at",
    "published_total_observed_at",
    "closing_spread_observed_at",
    "closing_total_observed_at",
    "current_kickoff_at",
    "void_reason",
]
_TIMESTAMP_COLUMNS = [
    "published_at",
    "kickoff_at",
    "published_spread_observed_at",
    "published_total_observed_at",
    "closing_spread_observed_at",
    "closing_total_observed_at",
    "current_kickoff_at",
]


def _pick(model, line, high_label, low_label):
    if pd.isna(model) or pd.isna(line) or model == line:
        return pd.NA
    return high_label if model > line else low_label


def _grade(pick, actual, line, high_label, force_no_pick=False):
    if force_no_pick:
        return "no_pick"
    if pd.isna(line) or pd.isna(actual):
        return "pending"
    if pd.isna(pick):
        return "no_pick"
    if actual == line:
        return "push"
    high_won = actual > line
    return "win" if high_won == (pick == high_label) else "loss"


def _clv(pick, published, closing, high_label, force_no_pick=False):
    """Points the line moved toward the pick between publication and close.

    Defined for any row carrying BOTH a published line and a closing line, which since the
    early-line backfill includes backtest rows. It used to be gated on record_type == "live",
    from when backtest rows had no published number to move from.
    """
    if force_no_pick:
        return np.nan
    if pd.isna(pick) or pd.isna(published) or pd.isna(closing):
        return np.nan
    movement = closing - published
    return float(movement if pick == high_label else -movement)


def _column(frame, name):
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def _force_no_pick(status, void_reason):
    return (not pd.isna(status) and status == "excluded") or not pd.isna(void_reason)


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
    void_reason = _column(ledger, "void_reason")
    spread_status = _column(ledger, "spread_publication_status")
    total_status = _column(ledger, "total_publication_status")
    spread_no_pick = [
        _force_no_pick(status, reason) for status, reason in zip(spread_status, void_reason)
    ]
    total_no_pick = [
        _force_no_pick(status, reason) for status, reason in zip(total_status, void_reason)
    ]
    ledger["spread_grade"] = [
        _grade(pick, actual, line, "home", force)
        for pick, actual, line, force in zip(
            ledger["spread_pick"], actual_margin, spread_line, spread_no_pick
        )
    ]
    ledger["total_grade"] = [
        _grade(pick, actual, line, "over", force)
        for pick, actual, line, force in zip(
            ledger["total_pick"], actual_total, total_line, total_no_pick
        )
    ]

    record_type = _column(ledger, "record_type")
    published_spread = _column(ledger, "published_spread_line")
    published_total = _column(ledger, "published_total_line")
    closing_spread = _column(ledger, "closing_spread_line")
    closing_total = _column(ledger, "closing_total_line")
    ledger["spread_clv"] = [
        _clv(pick, published, closing, "home", force)
        for pick, published, closing, force in zip(
            ledger["spread_pick"], published_spread, closing_spread, spread_no_pick
        )
    ]
    ledger["total_clv"] = [
        _clv(pick, published, closing, "over", force)
        for pick, published, closing, force in zip(
            ledger["total_pick"], published_total, closing_total, total_no_pick
        )
    ]

    ledger["spread_close_grade"] = [
        _grade(pick, actual, line, "home", force) if not pd.isna(status) else pd.NA
        for status, pick, actual, line, force in zip(
            spread_status, ledger["spread_pick"], actual_margin, closing_spread, spread_no_pick
        )
    ]
    ledger["total_close_grade"] = [
        _grade(pick, actual, line, "over", force) if not pd.isna(status) else pd.NA
        for status, pick, actual, line, force in zip(
            total_status, ledger["total_pick"], actual_total, closing_total, total_no_pick
        )
    ]
    return ledger.reindex(columns=LEDGER_COLUMNS)


def build_backtest_ledger(predictions, model_version=HISTORICAL_MODEL_VERSION, early_lines=None):
    """Convert historical walk-forward predictions into a validated ledger.

    With `early_lines` (a frame of game_id / early_spread_line / early_total_line), each game is
    graded against the number that was actually available at publication time, and the market's
    closing number is retained separately. That makes a backtest row mean the same thing as a
    live one: `spread_grade` is settled at the published line, `spread_close_grade` at the close.

    Without it the historical corpus keeps its original meaning -- graded at the close, with no
    published line and no CLV -- which is what the acceptance baseline pins.

    A game with no early line is marked `excluded` rather than dropped or back-filled from the
    close. Back-filling would grade it at a number nobody could have bet, and dropping it would
    lose the record that the market had not priced it.
    """
    facts = predictions.rename(
        columns={"margin": "actual_margin", "total_points": "actual_total"}
    ).copy()
    facts["record_type"] = "backtest"
    facts["model_version"] = model_version
    facts["estimator"] = OFFICIAL_ESTIMATOR
    facts["closing_spread_line"] = _column(facts, "spread_line")
    facts["closing_total_line"] = _column(facts, "total_line")
    facts["published_at"] = pd.NaT
    for column in _LIVE_ONLY_COLUMNS:
        facts[column] = pd.NA

    if early_lines is None:
        facts["official_spread_line"] = _column(facts, "spread_line")
        facts["official_total_line"] = _column(facts, "total_line")
        facts["published_spread_line"] = np.nan
        facts["published_total_line"] = np.nan
    else:
        _apply_early_lines(facts, early_lines)

    ledger = grade_ledger(facts)
    validate_ledger(ledger)
    return ledger


def _apply_early_lines(facts, early_lines):
    """Publish each game at its early line, and mark the unpriced ones excluded."""
    required = {"game_id", "early_spread_line", "early_total_line", "snapshot_at"}
    missing = sorted(required.difference(early_lines.columns))
    if missing:
        raise ValueError(f"early_lines is missing required column(s) {missing}")
    if early_lines["game_id"].duplicated().any():
        raise ValueError("early_lines contains duplicate game_id values")

    lookup = early_lines.drop_duplicates("game_id").set_index("game_id")
    observed = pd.to_datetime(facts["game_id"].map(lookup["snapshot_at"]), utc=True)
    facts["published_at"] = observed
    for kind, column in (("spread", "early_spread_line"), ("total", "early_total_line")):
        early = facts["game_id"].map(lookup[column]).astype(float)
        priced = early.notna()
        facts[f"official_{kind}_line"] = early
        facts[f"published_{kind}_line"] = early
        facts[f"{kind}_publication_status"] = np.where(priced, "published", "excluded")
        facts[f"{kind}_exclusion_reason"] = np.where(priced, None, "no_early_line")
        facts[f"published_{kind}_observed_at"] = observed.where(priced)


def _same_values(actual, expected):
    missing = object()
    actual_values = actual.astype(object).where(actual.notna(), missing)
    expected_values = expected.astype(object).where(expected.notna(), missing)
    return actual_values.eq(expected_values)


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


def _validate_utc_timestamps(ledger):
    for column in _TIMESTAMP_COLUMNS:
        for value in ledger.loc[ledger[column].notna(), column]:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid UTC timestamp in {column}") from error
            if timestamp.tzinfo is None or timestamp.utcoffset() != pd.Timedelta(0):
                raise ValueError(f"invalid UTC timestamp in {column}")


def _is_nonblank_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_market_publication(ledger, live, kind):
    status_column = f"{kind}_publication_status"
    reason_column = f"{kind}_exclusion_reason"
    official_column = f"official_{kind}_line"
    published_column = f"published_{kind}_line"
    observed_column = f"published_{kind}_observed_at"

    status = ledger.loc[live, status_column]
    if status.isna().any() or not status.isin(PUBLICATION_STATUSES).all():
        raise ValueError(f"invalid {kind} publication status")

    official = ledger.loc[live, official_column]
    published_line = ledger.loc[live, published_column]
    reason = ledger.loc[live, reason_column]
    observed_at = ledger.loc[live, observed_column]

    published = status.eq("published")
    valid_published = (
        official.notna() & published_line.notna() & reason.isna() & observed_at.notna()
    )
    if not valid_published.loc[published].all():
        raise ValueError(f"invalid published {kind} market")

    pending = status.eq("pending")
    valid_pending = official.isna() & published_line.isna() & reason.isna()
    if not valid_pending.loc[pending].all():
        raise ValueError(f"invalid pending {kind} market")

    excluded = status.eq("excluded")
    valid_excluded = official.isna() & published_line.isna() & reason.map(_is_nonblank_string)
    if not valid_excluded.loc[excluded].all():
        raise ValueError(f"invalid excluded {kind} market")


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
        values = ledger[column]
        if (
            not values.map(
                lambda value: isinstance(value, Integral) and not isinstance(value, bool)
            ).all()
            or not (values > 0).all()
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
    # A row that went through publication -- live, or a backtest row graded at its early line --
    # is identified by carrying a publication status, not by its record type.
    publishing = ledger[["spread_publication_status", "total_publication_status"]].notna().any(
        axis=1
    )
    unpublished = ~publishing
    for official, closing in (
        ("official_spread_line", "closing_spread_line"),
        ("official_total_line", "closing_total_line"),
    ):
        if not _same_values(
            ledger.loc[unpublished, official], ledger.loc[unpublished, closing]
        ).all():
            raise ValueError("unpublished official and closing lines must match")
    published_fields = ["published_spread_line", "published_total_line", "published_at"]
    if ledger.loc[unpublished, published_fields].notna().any().any():
        raise ValueError("unpublished rows cannot have published snapshot fields")
    if ledger.loc[unpublished, _LIVE_ONLY_COLUMNS].notna().any().any():
        raise ValueError("unpublished rows cannot have live-only fields")
    # A backtest row never has a live lifecycle, whether or not it was graded at an early line.
    if ledger.loc[backtest, ["current_kickoff_at", "void_reason"]].notna().any().any():
        raise ValueError("backtest rows cannot have live lifecycle fields")

    _validate_utc_timestamps(ledger)
    if ledger.loc[live, "current_kickoff_at"].isna().any():
        raise ValueError("live rows require a current kickoff")
    if ledger["void_reason"].dropna().map(lambda value: not _is_nonblank_string(value)).any():
        raise ValueError("void reason must be null or a nonblank string")

    for kind in ("spread", "total"):
        _validate_market_publication(ledger, publishing, kind)

    for official, published in (
        ("official_spread_line", "published_spread_line"),
        ("official_total_line", "published_total_line"),
    ):
        if not _same_values(
            ledger.loc[publishing, official], ledger.loc[publishing, published]
        ).all():
            raise ValueError("published official and published lines must match")

    _validate_derived(ledger)
