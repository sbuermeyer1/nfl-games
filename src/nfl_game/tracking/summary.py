"""Read-only summaries and audit rows for the immutable tracker ledger."""

from numbers import Integral

import numpy as np
import pandas as pd

from nfl_game.tracking.ledger import RECORD_TYPES

QUALIFIED_EDGE = 2.0
SPREAD_EDGE_THRESHOLDS = (5.0, 10.0, 15.0)
LIVE_UNAVAILABLE_MESSAGE = "Live tracking begins with the 2026 season."

_AUDIT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "published_at",
    "kickoff_at",
    "current_kickoff_at",
    "spread_publication_status",
    "total_publication_status",
    "spread_exclusion_reason",
    "total_exclusion_reason",
    "published_spread_line",
    "published_total_line",
    "closing_spread_line",
    "closing_total_line",
    "spread_clv",
    "total_clv",
    "spread_close_grade",
    "total_close_grade",
    "void_reason",
    "model_margin",
    "official_spread_line",
    "spread_pick",
    "spread_edge",
    "actual_margin",
    "spread_grade",
    "model_total",
    "official_total_line",
    "total_pick",
    "total_edge",
    "actual_total",
    "total_grade",
]


def record_summary(grades: pd.Series) -> dict:
    """Return a W-L-P record, excluding pushes from its win-rate denominator."""
    wins = int((grades == "win").sum())
    losses = int((grades == "loss").sum())
    pushes = int((grades == "push").sum())
    n_graded = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "n_graded": n_graded,
        "win_rate": wins / n_graded if n_graded else None,
    }


def _closing_summary(qualified: pd.DataFrame, kind: str) -> dict:
    clv_column = f"{kind}_clv"
    grade_column = f"{kind}_close_grade"
    with_clv = qualified.loc[qualified[clv_column].notna()]
    n_clv = len(with_clv)
    return {
        "average_clv": float(with_clv[clv_column].mean()) if n_clv else None,
        "beat_close_rate": float((with_clv[clv_column] > 0).mean()) if n_clv else None,
        "n_clv": n_clv,
        "record": record_summary(with_clv[grade_column]),
    }


def _core_summary(selected: pd.DataFrame) -> dict:
    qualified_spread = selected.loc[selected["spread_edge"].abs() >= QUALIFIED_EDGE]
    qualified_total = selected.loc[selected["total_edge"].abs() >= QUALIFIED_EDGE]
    summary = {
        "qualified": {
            "spread": record_summary(qualified_spread["spread_grade"]),
            "total": record_summary(qualified_total["total_grade"]),
        },
        "all_predictions": {
            "spread": record_summary(
                selected.loc[selected["spread_edge"].abs() > 0, "spread_grade"]
            ),
            "total": record_summary(selected.loc[selected["total_edge"].abs() > 0, "total_grade"]),
        },
        "spread_edges": [
            {
                "min_edge": threshold,
                "record": record_summary(
                    selected.loc[selected["spread_edge"].abs() >= threshold, "spread_grade"]
                ),
            }
            for threshold in SPREAD_EDGE_THRESHOLDS
        ],
    }
    if selected["record_type"].eq("live").all():
        summary["closing_line"] = {
            "spread": _closing_summary(qualified_spread, "spread"),
            "total": _closing_summary(qualified_total, "total"),
        }
    else:
        summary["closing_line"] = None
    return summary


def _parse_selection(record_type: str, season: str | int) -> int | None:
    if record_type not in RECORD_TYPES:
        raise ValueError("invalid record type")
    if season == "all":
        return None
    if isinstance(season, bool) or not isinstance(season, Integral):
        raise ValueError("season must be 'all' or an integer")  # noqa: TRY004
    return int(season)


def summarize_selection(ledger: pd.DataFrame, record_type: str, season: str | int) -> dict:
    """Summarize one isolated historical or live ledger selection."""
    selected_season = _parse_selection(record_type, season)
    typed = ledger.loc[ledger["record_type"] == record_type]
    if record_type == "live" and typed.empty:
        return {
            "available": False,
            "record_type": "live",
            "message": LIVE_UNAVAILABLE_MESSAGE,
        }
    if selected_season is not None:
        typed = typed.loc[typed["season"] == selected_season]

    summary = {
        "available": True,
        "record_type": record_type,
        "season": "all" if selected_season is None else selected_season,
        **_core_summary(typed),
    }
    if selected_season is None:
        summary["by_season"] = [
            {"season": int(year), **_core_summary(typed.loc[typed["season"] == year])}
            for year in sorted(typed["season"].unique())
        ]
    return summary


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_rows(ledger: pd.DataFrame, record_type: str, season: int) -> list[dict]:
    """Return browser-safe, game-level audit rows for one concrete season."""
    selected_season = _parse_selection(record_type, season)
    if selected_season is None:
        raise ValueError("audit rows require one concrete season")
    selected = ledger.loc[
        (ledger["record_type"] == record_type) & (ledger["season"] == selected_season),
        _AUDIT_COLUMNS,
    ].sort_values(["week", "game_id"])
    return [
        {column: _json_value(value) for column, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]
