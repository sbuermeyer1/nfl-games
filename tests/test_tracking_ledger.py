import numpy as np
import pandas as pd
import pytest

from nfl_game.tracking.ledger import build_backtest_ledger, grade_ledger, validate_ledger


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


def test_live_loss_and_closing_grades_retain_frozen_official_picks():
    out = grade_ledger(
        facts(
            {
                "game_id": "home-over-loss",
                "actual_margin": 2.0,
                "actual_total": 43.0,
                "closing_spread_line": 1.0,
                "closing_total_line": 46.0,
            },
            {
                "game_id": "away-under-loss",
                "model_margin": 1.0,
                "actual_margin": 4.0,
                "closing_spread_line": 5.0,
                "model_total": 40.0,
                "actual_total": 45.0,
                "closing_total_line": 42.0,
            },
            {
                "game_id": "closing-pending",
                "closing_spread_line": np.nan,
                "closing_total_line": np.nan,
            },
        )
    ).set_index("game_id")

    home_over = out.loc["home-over-loss"]
    assert (home_over.spread_pick, home_over.spread_grade, home_over.spread_close_grade) == (
        "home",
        "loss",
        "win",
    )
    assert (home_over.total_pick, home_over.total_grade, home_over.total_close_grade) == (
        "over",
        "loss",
        "loss",
    )
    assert (home_over.spread_clv, home_over.total_clv) == (-2.0, 2.0)

    away_under = out.loc["away-under-loss"]
    assert (away_under.spread_pick, away_under.spread_grade, away_under.spread_close_grade) == (
        "away",
        "loss",
        "win",
    )
    assert (away_under.total_pick, away_under.total_grade, away_under.total_close_grade) == (
        "under",
        "loss",
        "loss",
    )
    assert (away_under.spread_clv, away_under.total_clv) == (-2.0, 2.0)

    pending = out.loc["closing-pending"]
    assert (pending.spread_close_grade, pending.total_close_grade) == ("pending", "pending")
    assert pd.isna(pending.spread_clv)
    assert pd.isna(pending.total_clv)


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


@pytest.mark.parametrize(
    ("column", "value"),
    [("season", "2025"), ("week", True), ("week", "1")],
)
def test_validate_ledger_rejects_noncanonical_season_and_week_values(column, value):
    ledger = grade_ledger(facts({}))
    ledger[column] = ledger[column].astype(object)
    ledger.loc[0, column] = value

    with pytest.raises(ValueError, match=f"{column} must be a positive whole number"):
        validate_ledger(ledger)


@pytest.mark.parametrize("value", [2025, np.int64(2025)])
def test_validate_ledger_accepts_python_and_numpy_integer_seasons(value):
    ledger = grade_ledger(facts({"season": value, "week": np.int64(1)}))

    validate_ledger(ledger)


def test_build_backtest_ledger_validates_paired_missing_close_grades():
    predictions = pd.DataFrame(
        {
            "game_id": ["2025_01_AAA_BBB"],
            "season": [2025],
            "week": [1],
            "away_team": ["AAA"],
            "home_team": ["BBB"],
            "model_margin": [4.0],
            "model_total": [46.0],
            "spread_line": [3.0],
            "total_line": [44.0],
            "margin": [7.0],
            "total_points": [48.0],
        }
    )

    ledger = build_backtest_ledger(predictions)

    assert ledger.loc[0, "record_type"] == "backtest"
    assert pd.isna(ledger.loc[0, "spread_close_grade"])
    assert pd.isna(ledger.loc[0, "total_close_grade"])
    validate_ledger(ledger)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("published_spread_line", 2.5),
        ("published_total_line", 43.5),
    ],
)
def test_validate_ledger_rejects_published_lines_on_backtest(column, value):
    ledger = grade_ledger(
        facts(
            {
                "record_type": "backtest",
                "closing_spread_line": 3.0,
                "closing_total_line": 44.0,
                "published_spread_line": np.nan,
                "published_total_line": np.nan,
                "published_at": pd.NaT,
            }
        )
    )
    ledger.loc[0, column] = value

    with pytest.raises(ValueError, match="published"):
        validate_ledger(ledger)
