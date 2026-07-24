import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import ESTIMATORS, GameModel


def _train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["game_id"] = [f"g{i}" for i in range(n)]
    # margin is a known linear function of two features plus noise
    df["margin"] = (
        3.0 * df["net_rating_diff"] + 1.5 * df["rest_diff"] + rng.normal(scale=0.5, size=n)
    )
    df["total_points"] = 44.0 + 2.0 * df["off_pass_edge_home"] + rng.normal(scale=0.5, size=n)
    return df


def test_predict_returns_expected_columns():
    m = GameModel().fit(_train())
    out = m.predict(_train(n=10, seed=1))
    assert list(out.columns) == ["game_id", "model_margin", "model_total"]
    assert len(out) == 10


def test_recovers_a_known_linear_signal():
    train = _train()
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    pred = m.predict(train)
    corr = np.corrcoef(pred["model_margin"], train["margin"])[0, 1]
    assert corr > 0.95


def test_total_model_is_separate_from_margin():
    train = _train()
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    pred = m.predict(train)
    assert pred["model_total"].mean() == pytest.approx(44.0, abs=1.0)


def test_gbm_estimator_also_fits():
    train = _train()
    m = GameModel(estimator="gbm").fit(train)
    pred = m.predict(train)
    assert pred["model_margin"].notna().all()


def test_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        GameModel(estimator="magic")


def test_rows_with_null_targets_are_dropped_at_fit():
    train = _train()
    train.loc[:50, "margin"] = np.nan
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    assert m.n_train_margin_ == len(train) - 51


def test_can_predict_rows_with_null_targets():
    train = _train()
    future = _train(n=5, seed=2)
    future[["margin", "total_points"]] = np.nan
    m = GameModel().fit(train)
    out = m.predict(future)
    assert out["model_margin"].notna().all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GameModel().predict(_train(n=3))


def test_estimators_registry_exposes_both():
    assert set(ESTIMATORS) == {"ridge", "gbm"}


def _mixed_scale_train(n=300, seed=0):
    """Training data whose feature scales mirror the real ones.

    The real FEATURE_COLS mix 0/1 flags, EPA rating diffs near 0.1, and temperatures
    near 60. The signal lives entirely in the small-scale rating feature; temperature
    is pure noise. An unscaled L2 penalty shrinks by raw coefficient size, so it
    punishes the small-scale signal hardest and leaves the large-scale noise alone.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["game_id"] = [f"g{i}" for i in range(n)]
    df["net_rating_diff"] = rng.normal(scale=0.1, size=n)
    df["temp_outdoor"] = rng.normal(loc=60.0, scale=15.0, size=n)
    df["wind_outdoor"] = rng.normal(loc=8.0, scale=5.0, size=n)
    df["is_dome"] = rng.integers(0, 2, size=n)
    df["margin"] = 30.0 * df["net_rating_diff"] + rng.normal(scale=0.5, size=n)
    df["total_points"] = 44.0 + 20.0 * df["net_rating_diff"] + rng.normal(scale=0.5, size=n)
    return df


def test_predictions_are_invariant_to_feature_units():
    """Changing a feature's units must not change what the model predicts.

    Temperature in hundredths of a degree is the same information as temperature in
    degrees. Any model whose output moves when only the units move is being steered
    by scale rather than by signal.
    """
    train, test = _mixed_scale_train(), _mixed_scale_train(n=20, seed=1)
    base = GameModel(estimator="ridge", alpha=10.0).fit(train).predict(test)

    rescaled_train, rescaled_test = train.copy(), test.copy()
    for frame in (rescaled_train, rescaled_test):
        frame["temp_outdoor"] *= 100.0
    rescaled = GameModel(estimator="ridge", alpha=10.0).fit(rescaled_train).predict(rescaled_test)

    assert rescaled["model_margin"].to_numpy() == pytest.approx(
        base["model_margin"].to_numpy(), abs=1e-6
    )


def test_ridge_recovers_signal_carried_by_a_small_scale_feature():
    """The EPA ratings are the model's signal and they live on a ~0.1 scale."""
    train = _mixed_scale_train()
    pred = GameModel(estimator="ridge", alpha=10.0).fit(train).predict(train)
    mae = np.abs(pred["model_margin"] - train["margin"]).mean()
    assert mae < 1.0
