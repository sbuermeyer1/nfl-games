import pandas as pd
import pytest

from nfl_game.tracking.ledger import grade_ledger
from nfl_game.tracking.summary import audit_rows, summarize_selection


def tracker_ledger():
    rows = []
    for index, (record_type, season, edge, actual_margin, total_edge, actual_total) in enumerate(
        [
            ("backtest", 2024, 2.0, 6.0, 2.0, 48.0),
            ("backtest", 2024, 5.0, 1.0, -2.0, 40.0),
            ("backtest", 2025, 10.0, 20.0, 1.0, 48.0),
            ("backtest", 2025, 15.0, 3.0, -5.0, 44.0),
            ("live", 2026, 5.0, 9.0, 3.0, 49.0),
        ]
    ):
        line = 3.0
        total_line = 44.0
        published_spread = line if record_type == "live" else None
        published_total = total_line if record_type == "live" else None
        rows.append(
            {
                "record_type": record_type,
                "model_version": "ridge-v1",
                "estimator": "ridge",
                "game_id": f"game-{index}",
                "season": season,
                "week": 1,
                "away_team": "AAA",
                "home_team": "BBB",
                "model_margin": line + edge,
                "model_total": total_line + total_edge,
                "official_spread_line": line,
                "official_total_line": total_line,
                "published_spread_line": published_spread,
                "published_total_line": published_total,
                "closing_spread_line": line + (1.0 if record_type == "live" else 0.0),
                "closing_total_line": total_line + (1.0 if record_type == "live" else 0.0),
                "published_at": (
                    pd.Timestamp("2026-09-01T12:00:00Z") if record_type == "live" else None
                ),
                "kickoff_at": None,
                "actual_margin": actual_margin,
                "actual_total": actual_total,
            }
        )
    return grade_ledger(pd.DataFrame(rows))


def test_summary_separates_live_and_backtest_and_counts_pushes():
    summary = summarize_selection(tracker_ledger(), "backtest", "all")
    assert summary["available"] is True
    assert summary["qualified"]["spread"] == {
        "wins": 2,
        "losses": 1,
        "pushes": 1,
        "n_graded": 3,
        "win_rate": pytest.approx(2 / 3),
    }
    assert summary["all_predictions"]["total"]["n_graded"] == 3
    assert [row["min_edge"] for row in summary["spread_edges"]] == [5.0, 10.0, 15.0]
    assert [row["record"]["n_graded"] for row in summary["spread_edges"]] == [2, 1, 0]
    assert {row["season"] for row in summary["by_season"]} == {2024, 2025}


def test_live_clv_uses_the_frozen_qualified_cohort():
    summary = summarize_selection(tracker_ledger(), "live", "all")
    assert summary["closing_line"]["spread"] == {
        "average_clv": 1.0,
        "beat_close_rate": 1.0,
        "n_clv": 1,
        "record": {"wins": 1, "losses": 0, "pushes": 0, "n_graded": 1, "win_rate": 1.0},
    }


def test_empty_live_selection_is_unavailable_and_audit_requires_one_season():
    historical_only = tracker_ledger().query("record_type == 'backtest'")
    assert summarize_selection(historical_only, "live", "all") == {
        "available": False,
        "record_type": "live",
        "message": "Live tracking begins with the 2026 season.",
    }
    with pytest.raises(ValueError, match="concrete season"):
        audit_rows(historical_only, "backtest", "all")
