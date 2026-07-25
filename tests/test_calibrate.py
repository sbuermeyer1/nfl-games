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
