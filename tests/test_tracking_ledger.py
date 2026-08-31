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
        "spread_publication_status": "published",
        "total_publication_status": "published",
        "spread_exclusion_reason": pd.NA,
        "total_exclusion_reason": pd.NA,
        "published_spread_observed_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "published_total_observed_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "closing_spread_observed_at": pd.Timestamp("2026-09-06T17:01:00Z"),
        "closing_total_observed_at": pd.Timestamp("2026-09-06T17:01:00Z"),
        "current_kickoff_at": pd.Timestamp("2026-09-06T17:00:00Z"),
        "void_reason": pd.NA,
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


def test_excluded_market_is_no_pick_and_has_no_edge_clv_or_denominator():
    ledger = grade_ledger(
        facts(
            {
                "official_spread_line": np.nan,
                "published_spread_line": np.nan,
                "closing_spread_line": np.nan,
                "spread_publication_status": "excluded",
                "spread_exclusion_reason": "missing_line_at_deadline",
            }
        )
    )
    row = ledger.iloc[0]
    assert pd.isna(row["spread_pick"])
    assert pd.isna(row["spread_edge"])
    assert row["spread_grade"] == "no_pick"
    assert pd.isna(row["spread_clv"])
    validate_ledger(ledger)


def test_live_published_line_is_immutable_fact_and_clv_direction_stays_positive():
    out = grade_ledger(facts({})).iloc[0]
    assert out["official_spread_line"] == out["published_spread_line"] == 3.0
    assert out["spread_clv"] == 2.0


def test_void_game_is_no_pick_for_both_markets():
    out = grade_ledger(facts({"void_reason": "cancelled"})).iloc[0]
    assert out["spread_grade"] == "no_pick"
    assert out["total_grade"] == "no_pick"
    assert out["spread_close_grade"] == "no_pick"
    assert out["total_close_grade"] == "no_pick"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"spread_publication_status": "unknown"}, "publication status"),
        (
            {"spread_publication_status": "published", "published_spread_line": np.nan},
            "published spread",
        ),
        (
            {"spread_publication_status": "published", "spread_exclusion_reason": "why"},
            "published spread",
        ),
        (
            {"spread_publication_status": "published", "published_spread_observed_at": pd.NaT},
            "published spread",
        ),
        (
            {
                "spread_publication_status": "pending",
                "official_spread_line": 3.0,
                "published_spread_line": 3.0,
            },
            "pending spread",
        ),
        (
            {"spread_publication_status": "pending", "spread_exclusion_reason": "why"},
            "pending spread",
        ),
        (
            {"spread_publication_status": "excluded", "spread_exclusion_reason": "  "},
            "excluded spread",
        ),
        ({"current_kickoff_at": pd.NaT}, "current kickoff"),
        ({"void_reason": "  "}, "void reason"),
        ({"published_at": pd.Timestamp("2026-09-01T12:00:00")}, "UTC timestamp"),
        (
            {"closing_total_observed_at": pd.Timestamp("2026-09-06T18:01:00+01:00")},
            "UTC timestamp",
        ),
    ],
)
def test_validate_live_publication_state_rejects_invalid_facts(change, message):
    ledger = grade_ledger(facts(change))

    with pytest.raises(ValueError, match=message):
        validate_ledger(ledger)


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


def test_backtest_conversion_keeps_live_lifecycle_fields_null():
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
    live_only = [
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

    ledger = build_backtest_ledger(predictions)

    assert ledger[live_only].isna().all().all()
    invalid = ledger.copy()
    invalid.loc[0, "current_kickoff_at"] = pd.Timestamp("2025-09-01T12:00:00Z")
    with pytest.raises(ValueError, match="live-only"):
        validate_ledger(invalid)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("published_spread_line", 2.5),
        ("published_total_line", 43.5),
    ],
)
def test_validate_ledger_rejects_published_lines_on_an_unpublished_row(column, value):
    """A row with no publication lifecycle cannot carry a published line.

    This used to be keyed on record_type == "backtest". Backtest rows graded at an early line
    now legitimately carry published lines, so the rule is keyed on the publication status
    instead -- which is what actually distinguishes the two cases.
    """
    ledger = grade_ledger(
        facts(
            {
                "record_type": "backtest",
                "official_spread_line": 3.0,
                "official_total_line": 44.0,
                "closing_spread_line": 3.0,
                "closing_total_line": 44.0,
                "published_spread_line": np.nan,
                "published_total_line": np.nan,
                "published_at": pd.NaT,
                "spread_publication_status": pd.NA,
                "total_publication_status": pd.NA,
                "published_spread_observed_at": pd.NaT,
                "published_total_observed_at": pd.NaT,
                "closing_spread_observed_at": pd.NaT,
                "closing_total_observed_at": pd.NaT,
                "current_kickoff_at": pd.NaT,
            }
        )
    )
    ledger.loc[0, column] = value

    with pytest.raises(ValueError, match="published"):
        validate_ledger(ledger)


def test_validate_ledger_rejects_a_published_row_whose_official_line_drifts():
    """The mirror of the rule above: once published, official must equal the published number."""
    ledger = grade_ledger(facts({"record_type": "live"}))
    ledger.loc[0, "official_spread_line"] = 99.0

    with pytest.raises(ValueError, match="published official"):
        validate_ledger(ledger)


def _early_line_predictions() -> pd.DataFrame:
    """Two games: one priced 5 days out, one the market had not posted yet."""
    return pd.DataFrame(
        {
            "game_id": ["2025_01_AAA_BBB", "2025_01_CCC_DDD"],
            "season": [2025, 2025],
            "week": [1, 1],
            "away_team": ["AAA", "CCC"],
            "home_team": ["BBB", "DDD"],
            "model_margin": [7.0, 2.0],
            "model_total": [46.0, 44.0],
            "spread_line": [6.0, 1.0],
            "total_line": [44.0, 41.0],
            "margin": [5.0, 3.0],
            "total_points": [48.0, 40.0],
        }
    )


def _early_lines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2025_01_AAA_BBB"],
            "early_spread_line": [3.0],
            "early_total_line": [43.0],
            "snapshot_at": [pd.Timestamp("2025-09-02T00:20:00Z")],
        }
    )


def test_backtest_early_lines_become_the_official_and_published_line():
    ledger = build_backtest_ledger(
        _early_line_predictions(), early_lines=_early_lines()
    ).set_index("game_id")
    row = ledger.loc["2025_01_AAA_BBB"]

    assert row["official_spread_line"] == 3.0
    assert row["published_spread_line"] == 3.0
    assert row["official_total_line"] == 43.0
    assert row["published_total_line"] == 43.0
    # The close is still the market's closing number, not the early one.
    assert row["closing_spread_line"] == 6.0
    assert row["closing_total_line"] == 44.0


def test_backtest_early_and_closing_grades_can_disagree():
    """The whole point: the two grades must be able to differ, or neither means anything.

    model 7.0 vs early 3.0 picks home; the home team wins by 5. That covers the early number
    (5 > 3) and fails the closing one (5 < 6).
    """
    ledger = build_backtest_ledger(
        _early_line_predictions(), early_lines=_early_lines()
    ).set_index("game_id")
    row = ledger.loc["2025_01_AAA_BBB"]

    assert row["spread_pick"] == "home"
    assert row["spread_grade"] == "win"
    assert row["spread_close_grade"] == "loss"


def test_backtest_early_lines_populate_clv_signed_by_the_pick():
    ledger = build_backtest_ledger(
        _early_line_predictions(), early_lines=_early_lines()
    ).set_index("game_id")

    # Picked home at 3.0, the line closed at 6.0: it moved 3 points toward the pick.
    assert ledger.loc["2025_01_AAA_BBB", "spread_clv"] == 3.0


def test_backtest_game_without_an_early_line_is_excluded_not_graded():
    ledger = build_backtest_ledger(
        _early_line_predictions(), early_lines=_early_lines()
    ).set_index("game_id")
    row = ledger.loc["2025_01_CCC_DDD"]

    assert row["spread_publication_status"] == "excluded"
    assert row["spread_exclusion_reason"] == "no_early_line"
    assert row["spread_grade"] == "no_pick"
    assert row["spread_close_grade"] == "no_pick"
    assert pd.isna(row["published_spread_line"])


def test_backtest_without_early_lines_keeps_todays_closing_line_behaviour():
    """Omitting early lines must leave the shipped historical ledger exactly as it was."""
    ledger = build_backtest_ledger(_early_line_predictions()).set_index("game_id")
    row = ledger.loc["2025_01_AAA_BBB"]

    assert row["official_spread_line"] == 6.0
    assert pd.isna(row["published_spread_line"])
    assert pd.isna(row["spread_close_grade"])
    assert pd.isna(row["spread_clv"])
    # Graded against the close, which is what the acceptance baseline pins.
    assert row["spread_grade"] == "loss"


def test_clv_is_null_not_a_crash_when_the_published_line_is_pd_na():
    """The published-line guard in _clv is not redundant against np.nan's propagation.

    `5.0 - np.nan` is nan, so with float columns the guard looks like it does nothing. But
    `5.0 - pd.NA` is pd.NA and `float(pd.NA)` raises TypeError, and an object-dtype column is
    exactly what a ledger round-tripped through parquet can carry. Deleting the guard passes
    every other test in this file and then crashes on real data.
    """
    ledger = grade_ledger(
        facts(
            {
                "record_type": "backtest",
                "official_spread_line": 3.0,
                "official_total_line": 44.0,
                "closing_spread_line": 3.0,
                "closing_total_line": 44.0,
                "published_spread_line": pd.NA,
                "published_total_line": pd.NA,
                "published_at": pd.NaT,
                "spread_publication_status": pd.NA,
                "total_publication_status": pd.NA,
                "published_spread_observed_at": pd.NaT,
                "published_total_observed_at": pd.NaT,
                "closing_spread_observed_at": pd.NaT,
                "closing_total_observed_at": pd.NaT,
                "current_kickoff_at": pd.NaT,
            }
        )
    )

    assert pd.isna(ledger.loc[0, "spread_clv"])
    assert pd.isna(ledger.loc[0, "total_clv"])
    validate_ledger(ledger)
