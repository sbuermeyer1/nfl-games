import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import (
    DEFAULT_ALPHA,
    ESTIMATORS,
    DegenerateFeatureError,
    GameModel,
)


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


def test_gbm_warns_that_it_ignores_a_non_default_alpha():
    """--alpha is a documented CLI flag on both scripts, but the gbm factory discards
    it. Silently doing nothing lets a user believe they swept a hyperparameter when
    every run was byte-identical, so the no-op has to be said out loud."""
    with pytest.warns(UserWarning, match="ignores alpha"):
        GameModel(estimator="gbm", alpha=5.0)


def test_alpha_warning_does_not_fire_for_ridge(recwarn):
    """The converse, guarding the estimator half of the condition: ridge genuinely uses
    alpha, so warning there would be wrong and would train users to ignore the warning.
    """
    GameModel(estimator="ridge", alpha=5.0)
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []


def test_alpha_warning_does_not_fire_for_gbm_at_the_default(recwarn):
    """The other half of the condition: every caller passes alpha unconditionally
    (argparse supplies its default), so warning whenever gbm sees an alpha at all would
    fire on every plain `--estimator gbm` run. Only a value the user actually chose --
    one differing from the default -- represents a knob they think they turned."""
    GameModel(estimator="gbm", alpha=DEFAULT_ALPHA)
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []


def test_predict_names_every_missing_column():
    """A malformed frame must fail with a message that says what is wrong and where to
    get a right one, not pandas' bare KeyError from inside .to_numpy -- and it must name
    every missing column, since fixing them one error at a time is the slow way to find
    out you built the wrong frame.

    Neither column named here appears in the message's static prose, so these assertions
    can only be satisfied by the reported list. An earlier version dropped game_id and
    checked for it in the message, which passed even with game_id removed from the
    check: the guidance sentence mentions game_id by name."""
    m = GameModel().fit(_train())
    bad = _train(n=5, seed=1).drop(columns=["rest_diff", "div_game"])

    with pytest.raises(ValueError, match="missing required column") as exc:
        m.predict(bad)
    assert "rest_diff" in str(exc.value)
    assert "div_game" in str(exc.value)


def test_predict_requires_game_id_and_not_only_the_feature_columns():
    """game_id is read separately from FEATURE_COLS, a line below the .to_numpy call, so
    a check covering only the features lets the frame through to a second and much more
    opaque KeyError -- exactly the failure this check exists to replace."""
    m = GameModel().fit(_train())
    bad = _train(n=5, seed=1).drop(columns=["game_id"])

    with pytest.raises(ValueError, match="missing required column"):
        m.predict(bad)


def test_predict_schema_check_runs_after_the_fit_check():
    """An unfitted model given a malformed frame must still report the unfitted state:
    that is the caller's actual mistake, and a schema complaint would send them off
    fixing the wrong thing."""
    with pytest.raises(RuntimeError, match="fit"):
        GameModel().predict(_train(n=3).drop(columns=["game_id"]))


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


def test_near_constant_feature_with_near_zero_mean_does_not_explode_predictions():
    """A feature that is constant to floating-point noise must not blow up predictions.

    This mirrors ryoe_diff in an early season: NGS rushing data is too sparse, so
    nearly every row gets the same imputed value and the feature's variance within the
    training slice is on the order of 1e-34 around a mean near zero. sklearn's own
    constant-feature detection (``_is_constant_feature``) is relative to the feature's
    mean, so a near-zero-mean feature like this slips through with ``scale_`` on the
    order of 1e-17 instead of being floored to 1.0. Standardizing a future value that
    differs even slightly from the training mean then divides by that near-zero scale
    and explodes -- predictions should stay in a sane range regardless.
    """
    rng = np.random.default_rng(0)
    n = 256
    train = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    train["game_id"] = [f"g{i}" for i in range(n)]
    # Near-constant feature: mean near zero, variance on the order of float64 noise.
    train["ryoe_diff"] = 1e-17 * rng.normal(size=n)
    train["margin"] = (
        3.0 * train["net_rating_diff"] + 1.5 * train["rest_diff"] + rng.normal(scale=0.5, size=n)
    )
    train["total_points"] = 44.0 + 2.0 * train["off_pass_edge_home"] + rng.normal(scale=0.5, size=n)

    test = pd.DataFrame({c: rng.normal(size=10) for c in FEATURE_COLS})
    test["game_id"] = [f"t{i}" for i in range(10)]
    # A later slate where the feature actually varies, unlike the near-constant
    # training slice -- this is what dividing by ~1e-17 turns into an explosion.
    test["ryoe_diff"] = rng.normal(size=10)

    m = GameModel(estimator="ridge", alpha=1.0).fit(train)
    pred = m.predict(test)

    assert pred["model_margin"].abs().max() < 1000
    assert pred["model_total"].abs().max() < 1000


def _degenerate_ryoe(n):
    """Mirror the real 2018 walk-forward training fold's ryoe_diff exactly: 512 rows,
    416 at 0.0, ~94 at float noise around it, only 2 rows carrying a real value. std
    comes out to ~1.05e-2 -- far above RobustStandardScaler's ~2.22e-15 floor, so the
    scaler does not catch it, but only 5 distinct values across the whole column."""
    values = np.zeros(n)
    values[0], values[1] = 0.1685771, -0.1685771
    half = (n - 2) // 2
    values[2 : 2 + half] = 2.775558e-17
    values[2 + half : 2 + 2 * half] = -2.775558e-17
    return values


def test_degenerate_feature_with_few_distinct_values_raises():
    """A feature whose variance is nowhere near sklearn's epsilon floor can still be
    degenerate: too few distinct values to fit a coefficient on, because almost every
    row shares the same imputed default. GameModel.fit must refuse to train on it
    rather than silently letting the ridge pipeline learn noise as if it were signal."""
    n = 512
    train = _train(n=n)
    train["ryoe_diff"] = _degenerate_ryoe(n)
    assert train["ryoe_diff"].std() > 1e-4  # confirms this is NOT the eps-floor case
    assert train["ryoe_diff"].nunique() < 10

    with pytest.raises(DegenerateFeatureError, match="ryoe_diff"):
        GameModel(estimator="ridge", alpha=1.0).fit(train)


def test_all_nan_feature_column_is_treated_as_degenerate():
    """The most degenerate column possible must not be the one case exempted.

    An all-NaN column used to be skipped by the guard and then reach Ridge.fit, which
    raises a raw sklearn ValueError that walk_forward does not catch -- so a single bad
    column crashed the entire backtest instead of dropping one fold.
    """
    train = _train(n=300)
    train["ryoe_diff"] = np.nan

    with pytest.raises(DegenerateFeatureError, match="ryoe_diff"):
        GameModel(estimator="ridge", alpha=1.0).fit(train)


def test_degenerate_feature_guard_also_applies_to_gbm():
    """The guard is about whether the training slice itself can support a coefficient
    for that feature, not about which estimator is asked to fit it -- so it must fire
    the same way regardless of estimator."""
    n = 512
    train = _train(n=n)
    train["ryoe_diff"] = _degenerate_ryoe(n)
    with pytest.raises(DegenerateFeatureError, match="ryoe_diff"):
        GameModel(estimator="gbm").fit(train)


def test_binary_flags_and_low_cardinality_healthy_features_do_not_trigger_the_guard():
    """is_dome, div_game, and ngs_imputed_any are legitimate 0/1 indicator flags that
    take only two values in every real fold -- that is normal, not degenerate, and must
    be exempt. rest_diff is a small-range integer difference that has as few as 15
    distinct values in the smallest real walk-forward training slice (season 2019,
    trained on 2016-2018) -- comfortably healthy, and must not trip the guard either."""
    n = 300
    rng = np.random.default_rng(1)
    train = _train(n=n)
    train["is_dome"] = rng.integers(0, 2, size=n)
    train["div_game"] = rng.integers(0, 2, size=n)
    train["ngs_imputed_any"] = rng.integers(0, 2, size=n)
    train["rest_diff"] = rng.integers(-6, 7, size=n)  # 13 possible values, like real data

    GameModel(estimator="ridge", alpha=1.0).fit(train)  # must not raise
