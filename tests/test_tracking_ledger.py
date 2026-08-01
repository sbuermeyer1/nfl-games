import numpy as np
import pandas as pd
import pytest

from nfl_game.tracking.ledger import grade_ledger, validate_ledger


def facts(*overrides):
    base = {
        "record_type": "live",
        "model_version": "ridge-v1",
        "estimator": "ridge",
        "game_id": "2026_01_AAA_BBB",
        "season": 2026,
        "week": 1,
        "away_team": "AAA",
        "home_team": "BBB",
        "model_margin": 7.0,
        "model_total": 48.0,
        "official_spread_line": 3.0,
        "official_total_line": 44.0,
        "published_spread_line": 3.0,
        "published_total_line": 44.0,
        "closing_spread_line": 5.0,
        "closing_total_line": 46.0,
        "published_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "kickoff_at": pd.Timestamp("2026-09-06T17:00:00Z"),
        "actual_margin": 6.0,
        "actual_total": 47.0,
    }
    return pd.DataFrame([{**base, **change} for change in overrides])


def test_grade_ledger_pins_direction_grades_and_positive_clv():
    out = grade_ledger(
        facts(
            {},
            {
                "game_id": "2026_01_CCC_DDD",
                "model_margin": -2.0,
                "official_spread_line": 1.0,
                "published_spread_line": 1.0,
                "closing_spread_line": -1.0,
                "actual_margin": -3.0,
                "model_total": 41.0,
                "official_total_line": 44.0,
                "published_total_line": 44.0,
                "closing_total_line": 42.0,
                "actual_total": 40.0,
            },
        )
    ).set_index("game_id")

    home = out.loc["2026_01_AAA_BBB"]
    assert (home.spread_pick, home.spread_grade, home.spread_clv) == ("home", "win", 2.0)
    assert (home.total_pick, home.total_grade, home.total_clv) == ("over", "win", 2.0)

    away = out.loc["2026_01_CCC_DDD"]
    assert (away.spread_pick, away.spread_grade, away.spread_clv) == ("away", "win", 2.0)
    assert (away.total_pick, away.total_grade, away.total_clv) == ("under", "win", 2.0)


def test_push_no_pick_and_pending_are_distinct():
    out = grade_ledger(
        facts(
            {"game_id": "push", "actual_margin": 3.0, "actual_total": 44.0},
            {
                "game_id": "no-pick",
                "model_margin": 3.0,
                "model_total": 44.0,
            },
            {"game_id": "pending", "actual_margin": np.nan, "actual_total": np.nan},
            {
                "game_id": "missing-line",
                "official_spread_line": np.nan,
                "official_total_line": np.nan,
            },
        )
    ).set_index("game_id")

    assert out.loc["push", ["spread_grade", "total_grade"]].tolist() == ["push", "push"]
    assert out.loc["no-pick", ["spread_grade", "total_grade"]].tolist() == [
        "no_pick",
        "no_pick",
    ]
    assert out.loc["pending", ["spread_grade", "total_grade"]].tolist() == [
        "pending",
        "pending",
    ]
    assert out.loc["missing-line", ["spread_grade", "total_grade"]].tolist() == [
        "pending",
        "pending",
    ]


def test_validate_ledger_rejects_duplicates_non_ridge_and_stale_grades():
    valid = grade_ledger(facts({}))
    validate_ledger(valid)

    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(pd.concat([valid, valid], ignore_index=True))

    non_ridge = valid.copy()
    non_ridge.loc[0, "estimator"] = "gbm"
    with pytest.raises(ValueError, match="ridge"):
        validate_ledger(non_ridge)

    stale = valid.copy()
    stale.loc[0, "spread_grade"] = "loss"
    with pytest.raises(ValueError, match="derived"):
        validate_ledger(stale)
