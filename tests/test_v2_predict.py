import numpy as np
import pandas as pd
import pytest

from nfl_game.model.v2 import RidgeV2Model, fit_target_ridge
from nfl_game.model.v2_config import FeatureManifest, TargetConfig, V2ModelConfig


def _target_config(candidate: str, alpha: float) -> TargetConfig:
    return TargetConfig(
        candidate=candidate,
        alpha=alpha,
        short_halflife=4,
        long_halflife=16,
        prior_season_weight=0.4,
    )


def _config() -> V2ModelConfig:
    return V2ModelConfig(
        margin=_target_config("C0", 0.1),
        total=_target_config("C1", 100.0),
    )


def _manifest() -> FeatureManifest:
    return FeatureManifest(
        version="ridge-v2-test",
        margin_by_candidate={"C0": ("margin_signal", "margin_imputed_any")},
        total_by_candidate={"C1": ("total_signal", "total_imputed_any")},
        sources={},
        constants={},
    )


def _training_rows() -> pd.DataFrame:
    margin_signal = np.linspace(-3.0, 3.0, 12)
    total_signal = np.linspace(10.0, 21.0, 12)
    return pd.DataFrame(
        {
            "game_id": [f"train-{index}" for index in range(12)],
            "margin_signal": margin_signal,
            "margin_imputed_any": np.zeros(12),
            "total_signal": total_signal,
            "total_imputed_any": np.zeros(12),
            "spread_line": np.linspace(-7.0, 7.0, 12),
            "total_line": np.linspace(39.0, 50.0, 12),
            "cover_prob": np.linspace(0.4, 0.6, 12),
            "over_prob": np.linspace(0.6, 0.4, 12),
            "margin": 1.0 + 2.0 * margin_signal,
            "total_points": 20.0 + 3.0 * total_signal,
        }
    )


def _probe_rows() -> pd.DataFrame:
    return _training_rows().iloc[[2, 5, 8]].copy().reset_index(drop=True)


def test_margin_prediction_uses_only_the_margin_schema():
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())
    changed = _probe_rows()
    changed["margin_signal"] += 10.0

    before = model.predict(_probe_rows())
    after = model.predict(changed)

    assert not np.allclose(before["model_margin"], after["model_margin"])
    np.testing.assert_allclose(before["model_total"], after["model_total"])


def test_total_prediction_uses_only_the_total_schema():
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())
    changed = _probe_rows()
    changed["total_signal"] += 10.0

    before = model.predict(_probe_rows())
    after = model.predict(changed)

    np.testing.assert_allclose(before["model_margin"], after["model_margin"])
    assert not np.allclose(before["model_total"], after["model_total"])


def test_prediction_returns_only_ids_and_target_predictions():
    predicted = RidgeV2Model(_config(), _manifest()).fit(_training_rows()).predict(_probe_rows())

    assert list(predicted.columns) == ["game_id", "model_margin", "model_total"]


def test_each_pipeline_receives_its_configured_ridge_alpha():
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())

    assert model._margin.named_steps["ridge"].alpha == 0.1
    assert model._total.named_steps["ridge"].alpha == 100.0


def test_fit_target_ridge_returns_a_ridge_pipeline_with_the_requested_alpha():
    pipeline = fit_target_ridge(_training_rows(), "margin", _target_config("C0", 10.0), _manifest())

    assert pipeline.named_steps["ridge"].alpha == 10.0


def test_fit_independently_rejects_a_market_column_in_the_selected_manifest_schema():
    unsafe_manifest = _manifest()
    object.__setattr__(
        unsafe_manifest,
        "margin_by_candidate",
        {"C0": ("margin_signal", "spread_line")},
    )

    with pytest.raises(ValueError, match="market.*spread_line"):
        RidgeV2Model(_config(), unsafe_manifest).fit(_training_rows())


@pytest.mark.parametrize("missing", ("margin_signal", "total_signal"))
def test_fit_names_a_missing_selected_feature(missing):
    with pytest.raises(ValueError, match=missing):
        RidgeV2Model(_config(), _manifest()).fit(_training_rows().drop(columns=missing))


def test_fit_names_a_missing_target_column():
    with pytest.raises(ValueError, match="total_points"):
        RidgeV2Model(_config(), _manifest()).fit(_training_rows().drop(columns="total_points"))


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
def test_fit_rejects_nonfinite_selected_feature_values(invalid):
    bad = _training_rows()
    bad.loc[0, "margin_signal"] = invalid

    with pytest.raises(ValueError, match="non-finite.*margin_signal"):
        RidgeV2Model(_config(), _manifest()).fit(bad)


def test_null_target_filtering_happens_before_selected_matrix_validation():
    train = _training_rows()
    train.loc[0, ["margin", "margin_signal"]] = np.nan

    RidgeV2Model(_config(), _manifest()).fit(train)


def test_fit_rejects_a_target_with_no_non_null_rows_before_matrix_validation():
    train = _training_rows().drop(columns="margin_signal")
    train["margin"] = np.nan

    with pytest.raises(ValueError, match="no non-null.*margin"):
        RidgeV2Model(_config(), _manifest()).fit(train)


@pytest.mark.parametrize("invalid", (np.inf, -np.inf))
def test_fit_rejects_a_nonfinite_target(invalid):
    train = _training_rows()
    train.loc[0, "margin"] = invalid

    with pytest.raises(ValueError, match="non-finite.*margin"):
        RidgeV2Model(_config(), _manifest()).fit(train)


def test_fit_rejects_a_degenerate_nonbinary_training_feature():
    train = _training_rows()
    train["margin_signal"] = 2.0

    with pytest.raises(ValueError, match="degenerate.*margin_signal"):
        RidgeV2Model(_config(), _manifest()).fit(train)


def test_constant_binary_imputation_flags_are_allowed():
    train = _training_rows()
    train["margin_imputed_any"] = 0
    train["total_imputed_any"] = 1

    RidgeV2Model(_config(), _manifest()).fit(train)


def test_predict_before_fit_raises_even_for_a_malformed_frame():
    with pytest.raises(RuntimeError, match="fit"):
        RidgeV2Model(_config(), _manifest()).predict(pd.DataFrame())


@pytest.mark.parametrize("missing", ("margin_signal", "total_signal"))
def test_predict_independently_names_a_missing_selected_feature(missing):
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())

    with pytest.raises(ValueError, match=missing):
        model.predict(_probe_rows().drop(columns=missing))


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("column", ("margin_signal", "total_signal"))
def test_predict_independently_rejects_nonfinite_selected_values(column, invalid):
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())
    bad = _probe_rows()
    bad.loc[0, column] = invalid

    with pytest.raises(ValueError, match=f"non-finite.*{column}"):
        model.predict(bad)


def test_predict_requires_the_game_id_column():
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())

    with pytest.raises(ValueError, match="game_id"):
        model.predict(_probe_rows().drop(columns="game_id"))


def test_predict_rejects_missing_game_id_values():
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())
    bad = _probe_rows()
    bad.loc[0, "game_id"] = None

    with pytest.raises(ValueError, match="missing game_id"):
        model.predict(bad)


def test_predict_rejects_duplicate_game_id_values():
    model = RidgeV2Model(_config(), _manifest()).fit(_training_rows())
    bad = _probe_rows()
    bad.loc[1, "game_id"] = bad.loc[0, "game_id"]

    with pytest.raises(ValueError, match="duplicate game_id"):
        model.predict(bad)


def test_failed_second_target_fit_does_not_leave_a_partially_fitted_model():
    model = RidgeV2Model(_config(), _manifest())
    bad = _training_rows()
    bad["total_signal"] = 42.0

    with pytest.raises(ValueError, match="degenerate.*total_signal"):
        model.fit(bad)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(_probe_rows())


def test_fitted_state_does_not_leak_between_model_instances():
    fitted = RidgeV2Model(_config(), _manifest()).fit(_training_rows())
    unfitted = RidgeV2Model(_config(), _manifest())

    assert fitted.predict(_probe_rows()).notna().all().all()
    with pytest.raises(RuntimeError, match="fit"):
        unfitted.predict(_probe_rows())
