import numpy as np
import pandas as pd

from nfl_game.backtest import evaluate, market_comparison_regression, walk_forward
from nfl_game.model.features import FEATURE_COLS


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
    """A model fit on its own test season scores better in-sample. Honest
    walk-forward error must be strictly worse than that leaked baseline —
    no slack, because any slack is exactly where a real leak would hide."""
    from nfl_game.model.predict import GameModel

    feats = _features()
    honest = walk_forward(feats, test_seasons=[2023], alpha=0.01)
    mae_honest = (honest["model_margin"] - honest["margin"]).abs().mean()

    test_rows = feats[feats["season"] == 2023]
    leaked_pred = GameModel(alpha=0.01).fit(test_rows).predict(test_rows)
    mae_leaked = np.abs(
        leaked_pred["model_margin"].to_numpy() - test_rows["margin"].to_numpy()
    ).mean()

    assert mae_honest > mae_leaked


def test_walk_forward_skips_season_with_no_prior_data():
    out = walk_forward(_features(), test_seasons=[2021, 2022])
    assert sorted(out["season"].unique()) == [2022]


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


def test_market_regression_returns_both_coefficients():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    r = market_comparison_regression(out)
    assert "market_coef" in r and "model_coef" in r
    # the synthetic market line is a near-perfect signal, so it must dominate
    assert r["market_coef"] > r["model_coef"]
