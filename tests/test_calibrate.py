import numpy as np
import pandas as pd
import pytest

from nfl_game.model.calibrate import Calibrator, brier_score, reliability_table


def _preds(n=800, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.normal(scale=6.0, size=n)
    edge = rng.normal(scale=3.0, size=n)
    margin = spread + edge * 0.5 + rng.normal(scale=10.0, size=n)
    total_line = rng.normal(loc=45, scale=4.0, size=n)
    t_edge = rng.normal(scale=3.0, size=n)
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "spread_line": spread,
            "model_margin": spread + edge,
            "margin": margin,
            "total_line": total_line,
            "model_total": total_line + t_edge,
            "total_points": total_line + t_edge * 0.5 + rng.normal(scale=9.0, size=n),
        }
    )


def test_predict_returns_expected_columns():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=20, seed=1))
    assert list(out.columns) == ["game_id", "cover_prob", "over_prob"]


def test_probabilities_are_in_range():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=100, seed=1))
    assert out["cover_prob"].between(0, 1).all()
    assert out["over_prob"].between(0, 1).all()


def test_bigger_edge_means_higher_cover_probability():
    c = Calibrator().fit(_preds())
    df = _preds(n=2, seed=3)
    df["spread_line"] = [0.0, 0.0]
    df["model_margin"] = [1.0, 7.0]
    out = c.predict(df)
    assert out.iloc[1]["cover_prob"] > out.iloc[0]["cover_prob"]


def test_zero_edge_is_near_a_coin_flip():
    c = Calibrator().fit(_preds())
    df = _preds(n=1, seed=4)
    df["spread_line"] = [0.0]
    df["model_margin"] = [0.0]
    assert c.predict(df).iloc[0]["cover_prob"] == pytest.approx(0.5, abs=0.08)


def test_bigger_edge_means_higher_over_probability():
    c = Calibrator().fit(_preds())
    df = _preds(n=2, seed=3)
    df["total_line"] = [0.0, 0.0]
    df["model_total"] = [1.0, 7.0]
    out = c.predict(df)
    assert out.iloc[1]["over_prob"] > out.iloc[0]["over_prob"]


def test_zero_total_edge_is_near_a_coin_flip():
    c = Calibrator().fit(_preds())
    df = _preds(n=1, seed=4)
    df["total_line"] = [0.0]
    df["model_total"] = [0.0]
    assert c.predict(df).iloc[0]["over_prob"] == pytest.approx(0.5, abs=0.08)


def test_brier_score_rewards_accuracy():
    outcomes = np.array([1, 1, 0, 0])
    good = np.array([0.9, 0.8, 0.2, 0.1])
    bad = np.array([0.1, 0.2, 0.8, 0.9])
    assert brier_score(good, outcomes) < brier_score(bad, outcomes)


def test_reliability_table_shape():
    c = Calibrator().fit(_preds())
    p = _preds(n=400, seed=5)
    out = c.predict(p)
    covered = (p["margin"] > p["spread_line"]).astype(int).to_numpy()
    table = reliability_table(out["cover_prob"].to_numpy(), covered, bins=4)
    assert set(table.columns) == {"bin", "n", "mean_pred", "observed"}
    assert len(table) <= 4


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        Calibrator().predict(_preds(n=3))


def test_predict_all_clean_batch_has_no_nan_probs():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=5, seed=7))
    assert list(out.columns) == ["game_id", "cover_prob", "over_prob"]
    assert len(out) == 5
    assert out["cover_prob"].notna().all()
    assert out["over_prob"].notna().all()


def test_predict_with_nan_spread_line_yields_nan_cover_prob_only():
    c = Calibrator().fit(_preds())
    df = _preds(n=3, seed=8)
    df.loc[1, "spread_line"] = np.nan
    out = c.predict(df)

    assert list(out.columns) == ["game_id", "cover_prob", "over_prob"]
    assert len(out) == 3
    assert list(out["game_id"]) == list(df["game_id"])
    assert np.isnan(out.iloc[1]["cover_prob"])
    assert not np.isnan(out.iloc[1]["over_prob"])
    assert not np.isnan(out.iloc[0]["cover_prob"])
    assert not np.isnan(out.iloc[2]["cover_prob"])


def test_predict_with_nan_total_line_yields_nan_over_prob_only():
    c = Calibrator().fit(_preds())
    df = _preds(n=3, seed=8)
    df.loc[1, "total_line"] = np.nan
    out = c.predict(df)

    assert list(out.columns) == ["game_id", "cover_prob", "over_prob"]
    assert len(out) == 3
    assert list(out["game_id"]) == list(df["game_id"])
    assert np.isnan(out.iloc[1]["over_prob"])
    assert not np.isnan(out.iloc[1]["cover_prob"])
    assert not np.isnan(out.iloc[0]["over_prob"])
    assert not np.isnan(out.iloc[2]["over_prob"])


def test_fit_trains_cover_model_independently_of_total_line_nulls():
    """A row missing only total_line is unusable for the over target but should still
    teach the cover model, since cover only needs margin/spread_line/model_margin."""
    df_full = _preds(n=40, seed=11)
    baseline = Calibrator().fit(df_full)

    df_missing_total_line = df_full.copy()
    df_missing_total_line.loc[:4, "total_line"] = np.nan
    fitted_with_gap = Calibrator().fit(df_missing_total_line)

    probe = _preds(n=3, seed=12)
    base_out = baseline.predict(probe)
    gap_out = fitted_with_gap.predict(probe)

    pd.testing.assert_series_equal(base_out["cover_prob"], gap_out["cover_prob"], check_names=False)


def test_pushes_do_not_depress_the_intercept():
    """A push (margin == spread_line) returns the stake -- it is not a loss for the
    home side. Training `covered` as `margin > spread_line` with no push filter counts
    every push as "did not cover", which pulls the fitted intercept down and produces
    an artificially low cover_prob at zero edge.

    Build a sample that is half genuine, edge-independent 50/50 outcomes (so the
    honest zero-edge cover probability is ~0.5) and half exact pushes concentrated at
    zero edge. If pushes were miscounted as losses, cover_prob at zero edge would be
    pulled well below 0.5; excluding them (matching backtest.evaluate's own treatment)
    must keep it near 0.5."""
    rng = np.random.default_rng(0)
    n_real = 400
    spread = rng.normal(scale=6.0, size=n_real)
    edge = rng.normal(scale=3.0, size=n_real)
    real = pd.DataFrame(
        {
            "game_id": [f"r{i}" for i in range(n_real)],
            "spread_line": spread,
            "model_margin": spread + edge,
            # outcome noise is independent of edge, so cover rate is ~50% at every edge
            "margin": spread + rng.normal(scale=8.0, size=n_real),
            "total_line": 45.0,
            "model_total": 45.0,
            # non-push noise on the total side -- this test targets spread pushes only
            "total_points": 45.0 + rng.normal(scale=8.0, size=n_real),
        }
    )
    real = real[real["margin"] != real["spread_line"]].reset_index(drop=True)

    n_push = 400
    push = pd.DataFrame(
        {
            "game_id": [f"p{i}" for i in range(n_push)],
            "spread_line": 3.0,
            "model_margin": 3.0,  # zero edge
            "margin": 3.0,  # exact push
            "total_line": 45.0,
            "model_total": 45.0,
            # non-push noise on the total side -- this test targets spread pushes only
            "total_points": 45.0 + rng.normal(scale=8.0, size=n_push),
        }
    )
    df = pd.concat([real, push], ignore_index=True)

    c = Calibrator().fit(df)
    zero_edge = pd.DataFrame(
        {
            "game_id": ["z"],
            "spread_line": [0.0],
            "model_margin": [0.0],
            "total_line": [45.0],
            "model_total": [45.0],
        }
    )
    prob = c.predict(zero_edge).iloc[0]["cover_prob"]
    assert prob == pytest.approx(0.5, abs=0.1)


def test_total_pushes_do_not_depress_the_intercept():
    """The over/under mirror of test_pushes_do_not_depress_the_intercept.

    Calibrator.fit filters pushes on BOTH targets, but the spread half was the only
    one with coverage: deleting the `total_points != total_line` filter left the whole
    suite green. That is the same asymmetry that let every over_prob mutation survive
    in Task 10 -- the cover side pinned, the over side unguarded -- so it is pinned
    here explicitly.

    Same construction as the spread test with the roles swapped: half genuine,
    edge-independent 50/50 totals outcomes, half exact total pushes concentrated at
    zero edge, and non-push noise on the spread side so only the total filter is
    under test."""
    rng = np.random.default_rng(0)
    n_real = 400
    total_line = 45.0 + rng.normal(scale=4.0, size=n_real)
    edge = rng.normal(scale=3.0, size=n_real)
    real = pd.DataFrame(
        {
            "game_id": [f"r{i}" for i in range(n_real)],
            # non-push noise on the spread side -- this test targets total pushes only
            "spread_line": 3.0,
            "model_margin": 3.0,
            "margin": 3.0 + rng.normal(scale=8.0, size=n_real),
            "total_line": total_line,
            "model_total": total_line + edge,
            # outcome noise is independent of edge, so the over rate is ~50% at every edge
            "total_points": total_line + rng.normal(scale=8.0, size=n_real),
        }
    )
    real = real[real["total_points"] != real["total_line"]].reset_index(drop=True)

    n_push = 400
    push = pd.DataFrame(
        {
            "game_id": [f"p{i}" for i in range(n_push)],
            # non-push noise on the spread side -- this test targets total pushes only
            "spread_line": 3.0,
            "model_margin": 3.0,
            "margin": 3.0 + rng.normal(scale=8.0, size=n_push),
            "total_line": 44.0,
            "model_total": 44.0,  # zero edge
            "total_points": 44.0,  # exact push
        }
    )
    df = pd.concat([real, push], ignore_index=True)

    c = Calibrator().fit(df)
    zero_edge = pd.DataFrame(
        {
            "game_id": ["z"],
            "spread_line": [0.0],
            "model_margin": [0.0],
            "total_line": [44.0],
            "model_total": [44.0],
        }
    )
    prob = c.predict(zero_edge).iloc[0]["over_prob"]
    assert prob == pytest.approx(0.5, abs=0.1)


def test_reliability_table_mean_pred_within_bin_edges_but_observed_need_not_be():
    # Deliberately miscalibrated: low-probability predictions all turned out to be
    # correct (outcome 1) and high-probability predictions all turned out wrong
    # (outcome 0). mean_pred must still track the bin's own probability range (that's
    # a mean of the same values used to build the bin), while observed is free to land
    # anywhere -- here, the opposite corner of [0, 1]. This distinguishes the two
    # aggregations even though `test_reliability_table_shape` only checks column names.
    probs = np.array([0.05, 0.1, 0.9, 0.95])
    outcomes = np.array([1, 1, 0, 0])
    table = reliability_table(probs, outcomes, bins=4)

    for _, row in table.iterrows():
        interval = row["bin"]
        assert interval.left <= row["mean_pred"] <= interval.right

    low_bin = table.sort_values("bin").iloc[0]
    assert low_bin["mean_pred"] == pytest.approx(0.075)
    assert low_bin["observed"] == pytest.approx(1.0)
    assert not (low_bin["bin"].left <= low_bin["observed"] <= low_bin["bin"].right)
