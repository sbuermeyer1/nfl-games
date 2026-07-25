import numpy as np
import pandas as pd
import pytest

from nfl_game.backtest import ats_by_threshold, evaluate, market_comparison_regression, walk_forward
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import GameModel


def _features(seasons=(2021, 2022, 2023), n_per=100, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in seasons:
        df = pd.DataFrame({c: rng.normal(size=n_per) for c in FEATURE_COLS})
        df["game_id"] = [f"{s}_{i}" for i in range(n_per)]
        df["season"] = s
        df["week"] = rng.integers(1, 18, n_per)
        df["margin"] = 3.0 * df["net_rating_diff"] + rng.normal(scale=3.0, size=n_per)
        df["total_points"] = 44.0 + rng.normal(scale=5.0, size=n_per)
        df["spread_line"] = df["margin"] + rng.normal(scale=2.0, size=n_per)
        df["total_line"] = df["total_points"] + rng.normal(scale=2.0, size=n_per)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_walk_forward_only_scores_test_seasons():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    assert sorted(out["season"].unique()) == [2022, 2023]


def test_walk_forward_never_trains_on_the_test_season():
    """walk_forward's predictions for a test season must be byte-for-byte identical to
    a GameModel fit directly on features_df[features_df["season"] < test_season] and
    applied to that season's rows.

    An MAE-comparison version of this test ("honest error must be worse than a model
    leaked on the test season") does not actually detect a leak: mutation testing showed
    a walk_forward that trains on the test season *plus* all prior seasons still passes,
    because a model trained on 300 rows including the leaked 100 is not tight enough
    in-sample to beat a model trained on only those same 100 rows. Exact prediction
    equality against an explicitly-scoped reference model has no such slack: a `<` -> `<=`
    mutation in the train-season filter changes which rows enter fit() and immediately
    produces different predictions."""
    feats = _features()
    test_season = 2023

    out = walk_forward(feats, test_seasons=[test_season], alpha=0.01)

    train = feats[feats["season"] < test_season]
    test = feats[feats["season"] == test_season]
    expected = GameModel(alpha=0.01).fit(train).predict(test)

    merged = out.merge(expected, on="game_id", suffixes=("", "_expected"), validate="one_to_one")
    assert len(merged) == len(test)
    np.testing.assert_allclose(merged["model_margin"], merged["model_margin_expected"])
    np.testing.assert_allclose(merged["model_total"], merged["model_total_expected"])


def test_walk_forward_skips_season_with_no_prior_data():
    out = walk_forward(_features(), test_seasons=[2021, 2022])
    assert sorted(out["season"].unique()) == [2022]


def test_walk_forward_skips_a_fold_with_a_degenerate_training_slice():
    """A training slice that GameModel.fit refuses (too few distinct values for some
    feature to support a coefficient) must be skipped exactly like a season with no
    prior data at all -- a fold that cannot be trained should not contribute
    predictions to the caller (e.g. a calibration corpus)."""
    feats = _features(seasons=(2021, 2022, 2023))
    # Collapse 2021's rest_diff to a handful of imputed-looking values (mirroring the
    # real ryoe_diff pattern): mostly one value, with only two rows carrying anything
    # different. Its std stays well above the RobustStandardScaler eps floor, so this
    # is specifically the "too few distinct values" failure the new guard exists for,
    # not the already-fixed near-zero-variance one.
    n2021 = (feats["season"] == 2021).sum()
    degenerate = np.zeros(n2021)
    degenerate[0], degenerate[1] = 1.5, -1.5
    feats.loc[feats["season"] == 2021, "rest_diff"] = degenerate

    # test season 2022 trains on 2021 alone -> degenerate -> must be skipped.
    # test season 2023 trains on 2021+2022 combined -> 2022's healthy rest_diff values
    # give it plenty of distinct values -> must still be scored.
    out = walk_forward(feats, test_seasons=[2022, 2023])
    assert sorted(out["season"].unique()) == [2023]


def test_evaluate_reports_model_and_market_mae():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    m = evaluate(out)
    assert "margin_mae" in m and "market_margin_mae" in m
    assert "total_mae" in m and "market_total_mae" in m
    assert m["margin_mae"] > 0


def test_evaluate_reports_ats_hit_rate_and_n():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    m = evaluate(out)
    assert 0.0 <= m["ats_hit_rate"] <= 1.0
    assert m["ats_n"] > 0


def test_evaluate_excludes_games_with_no_posted_line():
    """A game with no posted spread_line must be dropped entirely, not scored as a push.

    NaN != NaN is True, so the old push filter (`margin != spread_line`) kept
    no-line rows, and `NaN > x` is False on both sides of the ATS comparison, so the
    row silently counted as a hit. Game "a" here has no line and must not appear in
    ats_n, n_games, margin_mae, or market_margin_mae at all.
    """
    preds = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "margin": [5.0, 7.0, 1.0],
            "total_points": [44.0, 44.0, 44.0],
            "model_margin": [10.0, 10.0, 10.0],
            "model_total": [44.0, 44.0, 44.0],
            "spread_line": [np.nan, 3.0, 3.0],
            "total_line": [44.0, 44.0, 44.0],
        }
    )
    m = evaluate(preds)
    assert m["n_games"] == 2
    assert m["ats_n"] == 2
    assert m["ats_hit_rate"] == pytest.approx(0.5)


def test_evaluate_excludes_pushes_from_ats():
    preds = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2023, 2023],
            "week": [1, 1],
            "margin": [7.0, 3.0],
            "total_points": [44.0, 44.0],
            "model_margin": [10.0, 1.0],
            "model_total": [45.0, 45.0],
            "spread_line": [7.0, 1.0],
            "total_line": [44.0, 44.0],
        }
    )
    m = evaluate(preds)
    # game "a" is an exact push against the spread and must not be counted
    assert m["ats_n"] == 1


def test_evaluate_ats_hit_rate_pins_sign_convention():
    """All-correct picks must score 1.0, and a mixed case pins the exact rate so a
    `model_margin > spread_line` -> `<` sign flip in the pick comparison would be
    caught: it would invert the mixed case's 2/3 to 1/3 rather than leaving it
    ambiguous inside [0, 1]."""
    perfect = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2023, 2023],
            "week": [1, 1],
            "margin": [10.0, -10.0],
            "total_points": [44.0, 44.0],
            "model_margin": [8.0, -8.0],
            "model_total": [44.0, 44.0],
            "spread_line": [3.0, -3.0],
            "total_line": [44.0, 44.0],
        }
    )
    # a: home favored by 3, model likes home even more (8 > 3) -> picks home;
    #    home wins by 10, covering the +3 line -> hit.
    # b: home dog by 3 (spread -3), model likes home less (-8 < -3) -> picks away;
    #    home loses by 10, well short of covering -3 -> hit.
    assert evaluate(perfect)["ats_hit_rate"] == 1.0

    mixed = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "margin": [7.0, 7.0, 1.0],
            "total_points": [44.0, 44.0, 44.0],
            "model_margin": [10.0, 10.0, 10.0],
            "model_total": [44.0, 44.0, 44.0],
            "spread_line": [3.0, 3.0, 3.0],
            "total_line": [44.0, 44.0, 44.0],
        }
    )
    # model always picks home (10 > 3). Home covers in a, b (margin 7 > 3) but not in
    # c (margin 1 < 3) -> 2 hits, 1 miss -> 2/3. Flipping the comparison to `<` would
    # make the model always pick away instead, inverting this to 1/3.
    m = evaluate(mixed)
    assert m["ats_hit_rate"] == pytest.approx(2 / 3)
    assert m["ats_n"] == 3


def test_evaluate_ou_hit_rate_pins_sign_convention():
    """Same discipline as the ATS test, applied to the over/under direction:
    model_total > total_line means "picks the over"."""
    perfect = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2023, 2023],
            "week": [1, 1],
            "margin": [0.0, 0.0],
            "total_points": [50.0, 38.0],
            "model_margin": [0.0, 0.0],
            "model_total": [48.0, 40.0],
            "spread_line": [10.0, 10.0],
            "total_line": [44.0, 44.0],
        }
    )
    # a: model predicts 48 (> 44) -> picks the over; actual 50 (> 44) -> went over -> hit.
    # b: model predicts 40 (< 44) -> picks the under; actual 38 (< 44) -> went under -> hit.
    assert evaluate(perfect)["ou_hit_rate"] == 1.0

    mixed = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "margin": [0.0, 0.0, 0.0],
            "total_points": [48.0, 48.0, 40.0],
            "model_margin": [0.0, 0.0, 0.0],
            "model_total": [50.0, 50.0, 50.0],
            "spread_line": [10.0, 10.0, 10.0],
            "total_line": [44.0, 44.0, 44.0],
        }
    )
    # model always picks the over (50 > 44). Actual went over in a, b (48 > 44) but
    # not in c (40 < 44) -> 2 hits, 1 miss -> 2/3. Flipping the comparison would
    # invert this to 1/3.
    m = evaluate(mixed)
    assert m["ou_hit_rate"] == pytest.approx(2 / 3)
    assert m["ou_n"] == 3


def test_evaluate_market_mae_matches_hand_computed_values():
    """market_margin_mae and market_total_mae must be the line's own error against the
    actual outcome on the same games -- not a swapped, negated, or hard-coded value.
    Negating spread_line inside market_margin_mae, or replacing market_total_mae with a
    literal, both survive a suite that only checks key presence."""
    preds = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "margin": [7.0, 7.0, 1.0],
            "total_points": [48.0, 48.0, 40.0],
            "model_margin": [0.0, 0.0, 0.0],
            "model_total": [0.0, 0.0, 0.0],
            "spread_line": [3.0, 3.0, 3.0],
            "total_line": [44.0, 44.0, 44.0],
        }
    )
    m = evaluate(preds)
    # |3-7|, |3-7|, |3-1| = 4, 4, 2 -> mean 10/3
    assert m["market_margin_mae"] == pytest.approx(10 / 3)
    # |44-48|, |44-48|, |44-40| = 4, 4, 4 -> mean 4.0
    assert m["market_total_mae"] == pytest.approx(4.0)


def test_ats_by_threshold_buckets_by_edge_size():
    """ats_by_threshold is printed in the deliverable report but had no test coverage.
    Pin its bucketing and per-bucket hit rate with a hand-built frame."""
    preds = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 1],
            "margin": [10.0, 1.0, 10.0, 1.0],
            "total_points": [44.0, 44.0, 44.0, 44.0],
            "model_margin": [4.0, 4.0, 9.0, 9.0],
            "model_total": [44.0, 44.0, 44.0, 44.0],
            "spread_line": [3.0, 3.0, 3.0, 3.0],
            "total_line": [44.0, 44.0, 44.0, 44.0],
        }
    )
    # edges: |4-3|=1 (a,b), |9-3|=6 (c,d).
    # threshold 0 includes all 4: model picks home in every game (4>3, 9>3).
    #   a: margin 10>3 covers -> hit. b: margin 1<3 doesn't cover -> miss.
    #   c: margin 10>3 covers -> hit. d: margin 1<3 doesn't cover -> miss.
    #   -> 2 hits / 4 = 0.5
    # threshold 5 includes only c, d (edge 6 >= 5) -> 1 hit / 2 = 0.5
    out = ats_by_threshold(preds, thresholds=(0, 5))
    row0 = out[out["min_edge"] == 0].iloc[0]
    row5 = out[out["min_edge"] == 5].iloc[0]
    assert row0["n"] == 4
    assert row0["hit_rate"] == pytest.approx(0.5)
    assert row5["n"] == 2
    assert row5["hit_rate"] == pytest.approx(0.5)


def test_market_regression_returns_both_coefficients():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    r = market_comparison_regression(out)
    assert "market_coef" in r and "model_coef" in r
    # the synthetic market line is a near-perfect signal, so it must dominate
    assert r["market_coef"] > r["model_coef"]
