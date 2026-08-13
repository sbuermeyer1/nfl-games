import json

import numpy as np
import pandas as pd
import pytest

from nfl_game.experiments.v2_evaluation import (
    BootstrapInterval,
    block_bootstrap_mean,
    evaluate_v2,
    joint_market_regression,
    promotion_decision,
    research_gate_decision,
    walk_forward_probabilities,
)


def _calibration_predictions() -> pd.DataFrame:
    rows = []
    for season in range(2019, 2026):
        for index, (edge, cover, over) in enumerate(
            ((-3.0, False, False), (-1.0, True, False), (1.0, False, True), (3.0, True, True))
        ):
            spread = float(index - 2)
            total_line = 42.0 + index
            rows.append(
                {
                    "game_id": f"{season}-{index}",
                    "season": season,
                    "week": index + 1,
                    "margin": spread + (1.0 if cover else -1.0),
                    "total_points": total_line + (1.0 if over else -1.0),
                    "spread_line": spread,
                    "total_line": total_line,
                    "model_margin": spread + edge,
                    "model_total": total_line + edge,
                }
            )
    return pd.DataFrame(rows)


def _prediction_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_rows = []
    v2_rows = []
    for season in range(2019, 2026):
        for index in range(4):
            margin = float(index * 3 - 4)
            total = float(40 + index * 3)
            spread = float(index - 2)
            total_line = float(42 + index)
            common = {
                "game_id": f"{season}-{index}",
                "season": season,
                "week": index + 1,
                "margin": margin,
                "total_points": total,
                "spread_line": spread,
                "total_line": total_line,
            }
            v1_rows.append(
                {
                    **common,
                    "model_margin": margin + (-2.0 if index % 2 else 2.0),
                    "model_total": total + (-3.0 if index % 2 else 3.0),
                }
            )
            v2_rows.append(
                {
                    **common,
                    "model_margin": margin + (-1.0 if index % 2 else 1.0),
                    "model_total": total + (-1.5 if index % 2 else 1.5),
                }
            )
    return pd.DataFrame(v1_rows), pd.DataFrame(v2_rows)


def _probabilities(predictions: pd.DataFrame, *, v2: bool) -> pd.DataFrame:
    report = predictions.loc[predictions["season"].between(2021, 2025)]
    cover = np.where(report["margin"] > report["spread_line"], 0.8, 0.2)
    over = np.where(report["total_points"] > report["total_line"], 0.8, 0.2)
    if not v2:
        cover = np.where(cover > 0.5, 0.7, 0.3)
        over = np.where(over > 0.5, 0.7, 0.3)
    return pd.DataFrame(
        {
            "game_id": report["game_id"].to_numpy(),
            "cover_prob": cover,
            "over_prob": over,
        }
    )


def _passing_report() -> dict[str, object]:
    return {
        "margin_mae": 10.0,
        "total_mae": 10.0,
        "margin_seasons_improved": 3,
        "total_seasons_improved": 3,
        "margin_paired_improvement_lower90": 0.01,
        "total_paired_improvement_lower90": 0.01,
        "margin_market_model_coef": 0.1,
        "margin_market_model_coef_lower90": 0.01,
        "total_market_model_coef": 0.1,
        "total_market_model_coef_lower90": 0.01,
        "ats_hit_rate": 0.487737556561086,
        "ou_hit_rate": 0.4922255192878339,
        "cover_brier": 0.2,
        "v1_cover_brier": 0.2,
        "over_brier": 0.2,
        "v1_over_brier": 0.2,
        "correctness_passed": True,
        "availability_passed": True,
        "determinism_passed": True,
        "source_reliability_passed": True,
    }


def test_walk_forward_calibration_uses_only_prior_oos_predictions():
    seen = []
    out = walk_forward_probabilities(_calibration_predictions(), fit_observer=seen.append)

    assert set(out["season"]) == set(range(2021, 2026))
    for target in ("cover", "over"):
        observation = next(
            item for item in seen if item["prediction_season"] == 2021 and item["target"] == target
        )
        assert observation["training_seasons"] == (2019, 2020)


def test_walk_forward_calibration_never_uses_preseed_seasons():
    predictions = _calibration_predictions()
    older = predictions.loc[predictions["season"] == 2019].copy()
    older["season"] = 2018
    older["game_id"] = [f"2018-{index}" for index in range(len(older))]
    seen = []

    walk_forward_probabilities(
        pd.concat([older, predictions], ignore_index=True), fit_observer=seen.append
    )

    assert all(2018 not in observation["training_seasons"] for observation in seen)


def test_walk_forward_calibration_excludes_pushes_independently_per_target():
    predictions = _calibration_predictions()
    predictions.loc[predictions["game_id"] == "2019-0", "margin"] = predictions.loc[
        predictions["game_id"] == "2019-0", "spread_line"
    ]
    predictions.loc[predictions["game_id"].isin(["2019-1", "2020-1"]), "total_points"] = (
        predictions.loc[predictions["game_id"].isin(["2019-1", "2020-1"]), "total_line"]
    )
    seen = []

    walk_forward_probabilities(predictions, fit_observer=seen.append)

    cover = next(x for x in seen if x["prediction_season"] == 2021 and x["target"] == "cover")
    over = next(x for x in seen if x["prediction_season"] == 2021 and x["target"] == "over")
    assert cover["n_training_games"] == 7
    assert over["n_training_games"] == 6


def test_walk_forward_calibration_rejects_duplicate_ids_and_missing_time_keys():
    duplicate = pd.concat(
        [_calibration_predictions(), _calibration_predictions().iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="unique.*game_id"):
        walk_forward_probabilities(duplicate)

    missing_week = _calibration_predictions()
    missing_week.loc[0, "week"] = np.nan
    with pytest.raises(ValueError, match="week"):
        walk_forward_probabilities(missing_week)


def test_walk_forward_calibration_is_unchanged_by_future_outcomes():
    original = _calibration_predictions()
    poisoned = original.copy()
    poisoned.loc[poisoned["season"] > 2021, "margin"] *= -50
    poisoned.loc[poisoned["season"] > 2021, "total_points"] *= -50

    before = walk_forward_probabilities(original).query("season == 2021")
    after = walk_forward_probabilities(poisoned).query("season == 2021")

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_block_bootstrap_mean_is_exact_and_deterministic():
    frame = pd.DataFrame({"season": [2021, 2021], "week": [1, 2], "value": [1.0, 3.0]})
    interval = block_bootstrap_mean(frame, "value", draws=5, seed=0)

    assert interval == BootstrapInterval(estimate=2.0, lower90=1.0, upper90=2.6)
    assert interval == block_bootstrap_mean(frame, "value", draws=5, seed=0)


@pytest.mark.parametrize(
    "frame, draws, match",
    (
        (pd.DataFrame(columns=["season", "week", "value"]), 5, "empty"),
        (pd.DataFrame({"season": [2021], "week": [1], "value": [np.nan]}), 5, "finite"),
        (pd.DataFrame({"season": [2021], "week": [np.nan], "value": [1.0]}), 5, "week"),
        (pd.DataFrame({"season": [2021], "week": [1], "value": [1.0]}), 0, "draws"),
    ),
)
def test_block_bootstrap_mean_rejects_malformed_inputs(frame, draws, match):
    with pytest.raises(ValueError, match=match):
        block_bootstrap_mean(frame, "value", draws=draws)


def test_joint_market_regression_reports_exact_model_increment_and_bootstrap():
    rows = []
    for week in range(1, 5):
        for index, (market, model) in enumerate(((0, 0), (1, 0), (0, 1), (2, 3))):
            rows.append(
                {
                    "season": 2021,
                    "week": week,
                    "actual": 1.0 + 2.0 * market + 3.0 * model,
                    "market": market,
                    "model": model,
                    "row": index,
                }
            )
    out = joint_market_regression(pd.DataFrame(rows), "actual", "market", "model", draws=50, seed=0)

    assert out["intercept"] == pytest.approx(1.0)
    assert out["market_coef"] == pytest.approx(2.0)
    assert out["model_coef"] == pytest.approx(3.0)
    assert out["model_coef_lower90"] == pytest.approx(3.0)
    assert out["model_coef_upper90"] == pytest.approx(3.0)


def test_joint_market_regression_rejects_degenerate_inputs():
    frame = pd.DataFrame(
        {
            "season": [2021, 2021, 2021],
            "week": [1, 1, 1],
            "actual": [1.0, 2.0, 3.0],
            "market": [1.0, 1.0, 1.0],
            "model": [2.0, 2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="degenerate"):
        joint_market_regression(frame, "actual", "market", "model", draws=10)


def test_evaluate_v2_reports_only_outer_seasons_on_identical_paired_rows(recwarn):
    v1, v2 = _prediction_pair()
    report = evaluate_v2(
        v1,
        v2,
        _probabilities(v1, v2=False),
        _probabilities(v2, v2=True),
        quality_checks={
            "correctness": True,
            "availability": True,
            "determinism": True,
            "source_reliability": True,
        },
        bootstrap_draws=50,
    )

    assert report["report_seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert report["n_games"] == 20
    assert report["margin_mae"] == pytest.approx(1.0)
    assert report["v1_margin_mae"] == pytest.approx(2.0)
    assert report["total_mae"] == pytest.approx(1.5)
    assert report["v1_total_mae"] == pytest.approx(3.0)
    assert report["margin_paired_improvement"] == pytest.approx(1.0)
    assert report["total_paired_improvement"] == pytest.approx(1.5)
    assert report["margin_seasons_improved"] == 5
    assert report["total_seasons_improved"] == 5
    assert set(report["per_season"]) == {"2021", "2022", "2023", "2024", "2025"}
    assert report["per_season"]["2021"]["market_margin_mae"] == pytest.approx(2.0)
    assert report["per_season"]["2021"]["market_total_mae"] == pytest.approx(2.0)
    for key in (
        "cover_calibration",
        "v1_cover_calibration",
        "over_calibration",
        "v1_over_calibration",
    ):
        assert set(report[key]) == {"intercept", "slope", "reliability"}
        assert isinstance(report[key]["reliability"], list)
    json.dumps(report, allow_nan=False)
    assert not recwarn.list


def test_evaluate_v2_cohorts_are_cumulative_and_report_pushes():
    v1, v2 = _prediction_pair()
    report = evaluate_v2(
        v1,
        v2,
        _probabilities(v1, v2=False),
        _probabilities(v2, v2=True),
        quality_checks={
            "correctness": True,
            "availability": True,
            "determinism": True,
            "source_reliability": True,
        },
        bootstrap_draws=20,
    )

    ats = report["ats_cohorts"]
    assert [cohort["min_edge"] for cohort in ats] == [0.0, 2.0, 5.0, 10.0, 15.0]
    assert [cohort["total_picks"] for cohort in ats] == sorted(
        [cohort["total_picks"] for cohort in ats], reverse=True
    )
    assert all(cohort["n"] == cohort["wins"] + cohort["losses"] for cohort in ats)


def test_evaluate_v2_rejects_duplicate_ids_missing_probabilities_and_no_pairs():
    v1, v2 = _prediction_pair()
    duplicated = pd.concat([v2, v2.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique.*game_id"):
        evaluate_v2(
            v1,
            duplicated,
            _probabilities(v1, v2=False),
            _probabilities(v2, v2=True),
            bootstrap_draws=5,
        )

    missing_probs = _probabilities(v2, v2=True).iloc[1:]
    with pytest.raises(ValueError, match="probabilit"):
        evaluate_v2(
            v1,
            v2,
            _probabilities(v1, v2=False),
            missing_probs,
            bootstrap_draws=5,
        )

    with pytest.raises(ValueError, match="paired"):
        evaluate_v2(
            v1.iloc[:0],
            v2,
            _probabilities(v1, v2=False),
            _probabilities(v2, v2=True),
            bootstrap_draws=5,
        )


@pytest.mark.parametrize(
    "quality",
    (
        None,
        {},
        {"correctness": True},
        {
            "correctness": True,
            "availability": True,
            "determinism": True,
            "source_reliability": "true",
        },
    ),
)
def test_evaluate_v2_cannot_turn_missing_or_nonboolean_quality_evidence_into_a_pass(quality):
    v1, v2 = _prediction_pair()
    kwargs = {} if quality is None else {"quality_checks": quality}
    report = evaluate_v2(
        v1,
        v2,
        _probabilities(v1, v2=False),
        _probabilities(v2, v2=True),
        bootstrap_draws=5,
        **kwargs,
    )

    decision = research_gate_decision(report)
    assert not decision.approved
    assert "correctness" in decision.pending or "source reliability" in decision.pending


@pytest.mark.parametrize(
    "key, value, expected",
    (
        ("margin_mae", 10.274, "margin MAE"),
        ("total_mae", 10.684, "total MAE"),
        ("margin_seasons_improved", 2, "margin season improvement"),
        ("total_seasons_improved", 2, "total season improvement"),
        ("margin_paired_improvement_lower90", 0.0, "margin paired improvement"),
        ("total_paired_improvement_lower90", 0.0, "total paired improvement"),
        ("margin_market_model_coef", 0.0, "margin market contribution"),
        ("margin_market_model_coef_lower90", 0.0, "margin market contribution"),
        ("total_market_model_coef", 0.0, "total market contribution"),
        ("total_market_model_coef_lower90", 0.0, "total market contribution"),
        ("ats_hit_rate", 0.4877375565610859, "ATS hit rate"),
        ("ou_hit_rate", 0.4922255192878338, "O/U hit rate"),
        ("cover_brier", 0.2000000000001, "cover Brier"),
        ("over_brier", 0.2000000000001, "over Brier"),
        ("correctness_passed", False, "correctness"),
        ("availability_passed", False, "availability"),
        ("determinism_passed", False, "determinism"),
        ("source_reliability_passed", False, "source reliability"),
    ),
)
def test_research_gate_reports_each_exact_failure(key, value, expected):
    report = _passing_report()
    report[key] = value
    decision = research_gate_decision(report)

    assert not decision.approved
    assert expected in decision.failures


def test_research_gate_reports_all_failures_and_never_accepts_nan():
    report = _passing_report()
    report["margin_mae"] = float("nan")
    report["availability_passed"] = False
    decision = research_gate_decision(report)

    assert not decision.approved
    assert "margin MAE" in decision.failures
    assert "availability" in decision.failures


def test_research_gate_reports_missing_evidence_as_pending():
    report = _passing_report()
    del report["margin_mae"]
    decision = research_gate_decision(report)

    assert not decision.approved
    assert "margin MAE" in decision.pending


def test_edge_cohorts_cannot_override_research_gates():
    report = _passing_report()
    report["ats_cohorts"] = [{"min_edge": 15.0, "hit_rate": 0.0}]
    report["ou_cohorts"] = [{"min_edge": 15.0, "hit_rate": 0.0}]
    decision = research_gate_decision(report)

    assert decision.approved
    assert decision.failures == ()
    assert decision.pending == ()


@pytest.mark.parametrize(
    "shadow, approved, failure, pending",
    (
        (None, False, (), ("shadow production rebuild",)),
        (False, False, ("shadow production rebuild",), ()),
        (True, True, (), ()),
    ),
)
def test_full_promotion_requires_successful_shadow_rebuild(shadow, approved, failure, pending):
    decision = promotion_decision(_passing_report(), shadow_rebuild_passed=shadow)

    assert decision.approved is approved
    assert decision.failures == failure
    assert decision.pending == pending
