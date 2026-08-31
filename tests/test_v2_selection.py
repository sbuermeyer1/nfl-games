import json
import warnings
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from nfl_game.experiments.v2_selection import (
    MIN_INNER_GAMES,
    MIN_INNER_SEASONS,
    ONE_STANDARD_ERROR_TOLERANCE,
    nested_walk_forward_v2,
    select_target_config,
)
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import GameModel
from nfl_game.model.v2_config import (
    PRIOR_SEASON_WEIGHTS,
    RATING_WINDOWS,
    FeatureManifest,
    TargetConfig,
    rating_setting_key,
    rating_variant_physical_column,
)
from nfl_game.model.v2_features import (
    DEFAULT_CONSTANTS,
    MARGIN_FEATURES_BY_BLOCK,
    TOTAL_FEATURES_BY_BLOCK,
)


def _config(candidate: str, alpha: float = 1.0) -> TargetConfig:
    return TargetConfig(
        candidate=candidate,
        alpha=alpha,
        short_halflife=8,
        long_halflife=24,
        prior_season_weight=0.6,
    )


def _score_rows(
    config: TargetConfig,
    maes: tuple[float, ...],
    games: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    counts = games or tuple(200 for _ in maes)
    return [
        {
            "config": config,
            "validation_season": 2020 + index,
            "mae": mae,
            "n_games": n_games,
        }
        for index, (mae, n_games) in enumerate(zip(maes, counts, strict=True))
    ]


def _physical_column(canonical: str, short: int, long: int, prior: float) -> str:
    return rating_variant_physical_column(canonical, short, long, prior)


def _variant_contract(*canonicals: str) -> dict[str, object]:
    canonicals = canonicals or ("signal",)
    variants = {}
    for short, long in RATING_WINDOWS:
        for prior in PRIOR_SEASON_WEIGHTS:
            mapping = {
                canonical: _physical_column(canonical, short, long, prior)
                for canonical in canonicals
            }
            variants[rating_setting_key(short, long, prior)] = {
                "margin": dict(mapping),
                "total": dict(mapping),
            }
    return {"rating_variant_columns": variants, "c5_production_eligible": False}


def _manifest(*, declare_c5: bool = False, c5_eligible: object = False) -> FeatureManifest:
    c0 = tuple(FEATURE_COLS)
    candidate = (*FEATURE_COLS, "signal")
    columns = {"C0": c0, "C1": candidate, "C2": candidate, "C4": candidate}
    if declare_c5:
        columns["C5"] = candidate
    constants = _variant_contract()
    constants["c5_production_eligible"] = c5_eligible
    return FeatureManifest(
        version="selection-test",
        margin_by_candidate=columns,
        total_by_candidate=columns,
        sources={},
        constants=constants,
    )


def _features(*, through: int = 2024, games_per_season: int = 200) -> pd.DataFrame:
    rows = []
    for season in range(2019, through + 1):
        for game in range(games_per_season):
            signal = float((game % 31) - 15)
            row = {
                "game_id": f"{season}-{game:03d}",
                "season": season,
                "week": 1 + game // 16,
                "signal": signal,
                "margin": signal,
                "total_points": signal,
            }
            for index, column in enumerate(FEATURE_COLS):
                row[column] = float(((game + index * 3 + season) % 37) - 18)
            for short, long in RATING_WINDOWS:
                for prior in PRIOR_SEASON_WEIGHTS:
                    row[_physical_column("signal", short, long, prior)] = signal
            rows.append(row)
    return pd.DataFrame(rows)


class _OffsetPredictor:
    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["signal"].to_numpy(dtype=float) + self.offset


class _RecordingFitter:
    def __init__(self, errors=None):
        self.errors = errors or {}
        self.calls: list[dict[str, object]] = []

    def __call__(self, train, target, config, manifest, *, validation_season):
        self.calls.append(
            {
                "target": target,
                "config": config,
                "training_seasons": tuple(sorted(train["season"].unique())),
                "validation_season": validation_season,
                "training_games": len(train),
            }
        )
        key = (target, config.candidate, config.alpha)
        return _OffsetPredictor(float(self.errors.get(key, 1.0)))


def _c0_manifest() -> FeatureManifest:
    return FeatureManifest(
        version="c0-selection-test",
        margin_by_candidate={"C0": tuple(FEATURE_COLS)},
        total_by_candidate={"C0": tuple(FEATURE_COLS)},
        sources={},
        constants={},
    )


def _c0_features(seasons=(2020, 2021, 2022), n_per=200, seed=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        frame = pd.DataFrame({column: rng.normal(size=n_per) for column in FEATURE_COLS})
        frame["game_id"] = [f"{season}-{game:03d}" for game in range(n_per)]
        frame["season"] = season
        frame["week"] = 1 + np.arange(n_per) // 16
        frame["margin"] = 2.5 * frame["net_rating_diff"] + rng.normal(size=n_per)
        frame["total_points"] = 44.0 + 1.5 * frame["off_pass_edge_home"] + rng.normal(size=n_per)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _default_variant_features() -> tuple[pd.DataFrame, FeatureManifest, tuple[int, int, float]]:
    rng = np.random.default_rng(99)
    desired = (12, 32, 0.8)
    frames = []
    for season in range(2019, 2024):
        n = 200
        frame = pd.DataFrame({column: rng.normal(size=n) for column in FEATURE_COLS})
        frame["game_id"] = [f"{season}-variant-{game:03d}" for game in range(n)]
        frame["season"] = season
        frame["week"] = 1 + np.arange(n) // 16
        for short, long in RATING_WINDOWS:
            for prior in PRIOR_SEASON_WEIGHTS:
                frame[_physical_column("rating_signal", short, long, prior)] = rng.normal(size=n)
        desired_column = _physical_column("rating_signal", *desired)
        frame["rating_signal"] = rng.normal(size=n)
        frame["margin"] = 8.0 * frame[desired_column] + rng.normal(scale=0.05, size=n)
        frame["total_points"] = 44.0 + 8.0 * frame[desired_column] + rng.normal(scale=0.05, size=n)
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    candidate_columns = (*FEATURE_COLS, "rating_signal")
    manifest = FeatureManifest(
        version="default-rating-variant-test",
        margin_by_candidate={"C0": tuple(FEATURE_COLS), "C1": candidate_columns},
        total_by_candidate={"C0": tuple(FEATURE_COLS), "C1": candidate_columns},
        sources={},
        constants=_variant_contract("rating_signal"),
    )
    return features, manifest, desired


def _contract_manifest(
    canonicals: tuple[str, ...] = ("rating_a", "rating_b"),
) -> FeatureManifest:
    c0 = tuple(FEATURE_COLS)
    c1 = (*c0, *canonicals)
    return FeatureManifest(
        version="strict-rating-contract-test",
        margin_by_candidate={"C0": c0, "C1": c1},
        total_by_candidate={"C0": c0, "C1": c1},
        sources={},
        constants=_variant_contract(*canonicals),
    )


def _empty_contract_features(
    canonicals: tuple[str, ...] = ("rating_a", "rating_b"),
) -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "week",
        "margin",
        "total_points",
        *FEATURE_COLS,
        *canonicals,
    ]
    for short, long in RATING_WINDOWS:
        for prior in PRIOR_SEASON_WEIGHTS:
            columns.extend(
                _physical_column(canonical, short, long, prior) for canonical in canonicals
            )
    return pd.DataFrame(
        {
            column: pd.Series(dtype="object" if column == "game_id" else "float64")
            for column in dict.fromkeys(columns)
        }
    )


def test_selection_contract_constants_are_pinned():
    assert MIN_INNER_SEASONS == 2
    assert MIN_INNER_GAMES == 400
    assert ONE_STANDARD_ERROR_TOLERANCE == 0.05


def test_candidate_with_one_season_or_399_games_is_ineligible():
    scores = pd.DataFrame(
        [
            *_score_rows(_config("C2"), (1.0,), (500,)),
            *_score_rows(_config("C4"), (0.5, 0.5), (199, 200)),
        ]
    )

    assert select_target_config(scores, target="margin") is None


def test_simplicity_selects_c2_when_c4_improves_by_only_point_zero_three():
    scores = pd.DataFrame(
        [
            *_score_rows(_config("C2"), (1.03, 1.03)),
            *_score_rows(_config("C4"), (1.00, 1.00)),
        ]
    )
    selected = select_target_config(scores, target="margin")
    assert selected is not None
    assert selected.config.candidate == "C2"
    assert selected.mean_inner_mae == 1.03


def test_simplicity_selects_c4_when_it_improves_by_point_zero_six():
    scores = pd.DataFrame(
        [
            *_score_rows(_config("C2"), (1.06, 1.06)),
            *_score_rows(_config("C4"), (1.00, 1.00)),
        ]
    )
    selected = select_target_config(scores, target="total_points")
    assert selected is not None
    assert selected.config.candidate == "C4"


def test_inner_seasons_are_scored_equally_instead_of_weighting_games():
    scores = pd.DataFrame(
        [
            *_score_rows(_config("C2"), (0.0, 10.0), (399, 1)),
            *_score_rows(_config("C4"), (4.0, 4.0), (200, 200)),
        ]
    )
    selected = select_target_config(scores, target="margin")
    assert selected is not None
    assert selected.config.candidate == "C4"
    assert selected.mean_inner_mae == 4.0


def test_exact_config_ties_use_target_config_key():
    first = _config("C2", alpha=10.0)
    second = replace(first, alpha=1.0)
    scores = pd.DataFrame([*_score_rows(first, (1.0, 1.0)), *_score_rows(second, (1.0, 1.0))])
    selected = select_target_config(scores, target="margin")
    assert selected is not None
    assert selected.config.key() == min(first.key(), second.key())


def test_inner_selection_never_reads_validation_or_outer_seasons():
    fitter = _RecordingFitter()
    nested_walk_forward_v2(_features(), [2024], _manifest(), target_fitter=fitter)
    assert fitter.calls
    assert all(
        not call["training_seasons"] or max(call["training_seasons"]) < call["validation_season"]
        for call in fitter.calls
    )
    assert all(2024 not in call["training_seasons"] for call in fitter.calls)


def test_poisoned_future_season_cannot_change_earlier_selection_or_predictions():
    errors = {("margin", "C2", 0.1): 0.2, ("total_points", "C4", 10.0): 0.1}
    clean = nested_walk_forward_v2(
        _features(), [2024], _manifest(), target_fitter=_RecordingFitter(errors)
    )
    poisoned_features = _features(through=2025)
    poisoned_features.loc[poisoned_features["season"] == 2025, ["signal", "margin"]] = 1e12
    poisoned = nested_walk_forward_v2(
        poisoned_features, [2024], _manifest(), target_fitter=_RecordingFitter(errors)
    )
    assert clean.selections == poisoned.selections
    pd.testing.assert_frame_equal(clean.predictions, poisoned.predictions)


def test_margin_and_total_select_independent_candidates_and_alphas():
    errors = {("margin", "C2", 0.1): 0.1, ("total_points", "C4", 10.0): 0.1}
    result = nested_walk_forward_v2(
        _features(), [2024], _manifest(), target_fitter=_RecordingFitter(errors)
    )
    selection = result.selections[0]
    assert (selection.margin.config.candidate, selection.margin.config.alpha) == ("C2", 0.1)
    assert (selection.total.config.candidate, selection.total.config.alpha) == ("C4", 10.0)


def test_outer_refit_uses_every_prior_season_and_scores_each_game_once():
    fitter = _RecordingFitter()
    result = nested_walk_forward_v2(_features(), [2024], _manifest(), target_fitter=fitter)
    refits = [
        call
        for call in fitter.calls
        if call["validation_season"] == 2024
        and call["config"]
        in {result.selections[0].margin.config, result.selections[0].total.config}
        and call["training_games"] == 5 * 200
    ]
    assert {call["target"] for call in refits} == {"margin", "total_points"}
    assert set(result.predictions["game_id"]) == set(_features().query("season == 2024")["game_id"])
    assert not result.predictions["game_id"].duplicated().any()


def test_seed_fallback_is_json_safe_repeatable_and_future_invariant():
    clean_features = _features(through=2020)
    first = nested_walk_forward_v2(
        clean_features, [2020], _manifest(), target_fitter=_RecordingFitter()
    )
    repeated = nested_walk_forward_v2(
        clean_features, [2020], _manifest(), target_fitter=_RecordingFitter()
    )
    poisoned_features = _features(through=2021)
    poisoned_features.loc[poisoned_features["season"] == 2021, :] = poisoned_features.loc[
        poisoned_features["season"] == 2021, :
    ].assign(margin=1e12, total_points=-1e12)
    poisoned = nested_walk_forward_v2(
        poisoned_features, [2020], _manifest(), target_fitter=_RecordingFitter()
    )
    assert first.selections == repeated.selections == poisoned.selections
    pd.testing.assert_frame_equal(first.predictions, repeated.predictions)
    pd.testing.assert_frame_equal(first.predictions, poisoned.predictions)
    for target_selection in (first.selections[0].margin, first.selections[0].total):
        assert target_selection.config.candidate == "C0"
        assert target_selection.mean_inner_mae is None
        assert target_selection.validation_seasons == ()
        assert target_selection.validation_games == 0
        assert '"mean_inner_mae": null' in json.dumps(asdict(target_selection))


def test_default_path_materializes_real_persisted_rating_variants_for_all_three_settings():
    features, manifest, desired = _default_variant_features()
    serialized = json.dumps(manifest.to_dict())
    restored = FeatureManifest.from_dict(json.loads(serialized))
    result = nested_walk_forward_v2(features, [2023], restored)
    selection = result.selections[0]
    for target_selection in (selection.margin, selection.total):
        assert target_selection.config.candidate == "C1"
        assert (
            target_selection.config.short_halflife,
            target_selection.config.long_halflife,
            target_selection.config.prior_season_weight,
        ) == desired


def test_empty_non_c0_input_still_rejects_a_missing_declared_variant_column():
    features, manifest, _ = _default_variant_features()
    missing = _physical_column("rating_signal", 4, 16, 0.4)
    with pytest.raises(ValueError, match=f"rating variant.*{missing}"):
        nested_walk_forward_v2(features.drop(columns=missing).iloc[0:0], [2023], manifest)


def test_c0_predictions_match_ridge_v1_game_model_exactly():
    features = _c0_features()
    train = features.query("season < 2022")
    test = features.query("season == 2022")
    expected = GameModel(estimator="ridge", alpha=1.0).fit(train).predict(test)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = nested_walk_forward_v2(features, [2022], _c0_manifest()).predictions
    merged = actual.merge(expected, on="game_id", suffixes=("", "_v1"), validate="one_to_one")
    np.testing.assert_array_equal(merged["model_margin"], merged["model_margin_v1"])
    np.testing.assert_array_equal(merged["model_total"], merged["model_total_v1"])


def test_c0_outer_fold_with_five_distinct_continuous_values_warns_and_skips():
    features = _c0_features(seasons=(2021, 2022), n_per=100)
    prior = features["season"] == 2021
    for index, column in enumerate(FEATURE_COLS):
        features.loc[prior, column] = (np.arange(prior.sum()) + index) % 5
    with pytest.warns(RuntimeWarning, match="skipping test season 2022.*degenerate"):
        result = nested_walk_forward_v2(features, [2022], _c0_manifest())
    assert result.predictions.empty
    assert result.selections == ()


def test_schema_valid_empty_and_no_game_inputs_return_exact_empty_contract():
    template = _c0_features(seasons=(2021,), n_per=20)
    empty = template.iloc[0:0]
    empty_result = nested_walk_forward_v2(empty, [2022], _c0_manifest())
    no_game_result = nested_walk_forward_v2(template, [2030], _c0_manifest())
    expected_columns = [*template.columns, "model_margin", "model_total"]
    assert list(empty_result.predictions.columns) == expected_columns
    assert list(no_game_result.predictions.columns) == expected_columns
    assert empty_result.predictions.empty and no_game_result.predictions.empty
    assert empty_result.selections == no_game_result.selections == ()


def test_empty_input_still_rejects_missing_manifested_feature_schema():
    empty = _c0_features(seasons=(2021,), n_per=20).iloc[0:0].drop(columns=FEATURE_COLS[0])
    with pytest.raises(ValueError, match=FEATURE_COLS[0]):
        nested_walk_forward_v2(empty, [2022], _c0_manifest())


def test_complete_nine_setting_variant_contract_round_trips_and_validates_on_empty_input():
    manifest = _contract_manifest()
    restored = FeatureManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    result = nested_walk_forward_v2(_empty_contract_features(), [2024], restored)
    assert result.predictions.empty
    variants = restored.constants["rating_variant_columns"]
    assert set(variants) == {
        rating_setting_key(short, long, prior)
        for short, long in RATING_WINDOWS
        for prior in PRIOR_SEASON_WEIGHTS
    }
    for short, long in RATING_WINDOWS:
        for prior in PRIOR_SEASON_WEIGHTS:
            setting = variants[rating_setting_key(short, long, prior)]
            for target in ("margin", "total"):
                assert set(setting[target]) == {"rating_a", "rating_b"}
                assert setting[target] == {
                    canonical: _physical_column(canonical, short, long, prior)
                    for canonical in ("rating_a", "rating_b")
                }


def test_real_default_manifest_round_trip_pins_every_c1_variant_column():
    def cumulative(blocks):
        columns = []
        result = {}
        for candidate, added in blocks.items():
            columns.extend(added)
            result[candidate] = tuple(dict.fromkeys(columns))
        return result

    manifest = FeatureManifest(
        version="real-default-round-trip-test",
        margin_by_candidate=cumulative(MARGIN_FEATURES_BY_BLOCK),
        total_by_candidate=cumulative(TOTAL_FEATURES_BY_BLOCK),
        sources={},
        constants=DEFAULT_CONSTANTS,
    )
    restored = FeatureManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    restored.validate_selection_contract()

    for target, blocks in (
        ("margin", MARGIN_FEATURES_BY_BLOCK),
        ("total", TOTAL_FEATURES_BY_BLOCK),
    ):
        c0 = set(blocks["C0"])
        expected = tuple(column for column in blocks["C1"] if column not in c0)
        if target == "margin":
            assert "home_indicator" in expected
        for short, long in RATING_WINDOWS:
            for prior in PRIOR_SEASON_WEIGHTS:
                config = TargetConfig("C1", 1.0, short, long, prior)
                mapping = restored.rating_variant_columns(target, config)
                assert tuple(mapping) == expected
                assert mapping == {
                    canonical: _physical_column(canonical, short, long, prior)
                    for canonical in expected
                }


def test_all_nine_settings_cannot_alias_physical_columns_to_canonical_names():
    payload = _contract_manifest().to_dict()
    for setting in payload["constants"]["rating_variant_columns"].values():
        for target in ("margin", "total"):
            setting[target] = {canonical: canonical for canonical in ("rating_a", "rating_b")}
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="exact.*physical"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


@pytest.mark.parametrize("mutation", ("incomplete", "extra"))
def test_variant_mapping_must_equal_the_complete_c1_minus_c0_canonical_set(mutation):
    payload = _contract_manifest().to_dict()
    first = next(iter(payload["constants"]["rating_variant_columns"].values()))["margin"]
    if mutation == "incomplete":
        first.pop("rating_b")
    else:
        first["rogue_feature"] = "rogue_feature__s4_l16_p04"
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="canonical.*set"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


def test_variant_mapping_rejects_physical_reuse_within_a_setting():
    payload = _contract_manifest().to_dict()
    first = next(iter(payload["constants"]["rating_variant_columns"].values()))["margin"]
    first["rating_b"] = first["rating_a"]
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="physical.*reuse|exact.*physical"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


def test_variant_mapping_rejects_a_physical_name_from_the_wrong_setting():
    payload = _contract_manifest().to_dict()
    variants = payload["constants"]["rating_variant_columns"]
    keys = list(variants)
    variants[keys[1]]["total"]["rating_a"] = variants[keys[0]]["total"]["rating_a"]
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="exact.*physical|wrong.*setting"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


@pytest.mark.parametrize("canonical_target", ("margin", "total"))
def test_variant_physical_column_cannot_alias_a_canonical_in_either_target_on_empty_input(
    canonical_target,
):
    payload = _contract_manifest().to_dict()
    colliding = _physical_column("rating_a", 4, 16, 0.4)
    candidate_key = "margin_by_candidate" if canonical_target == "margin" else "total_by_candidate"
    payload[candidate_key]["C2"] = [
        *payload[candidate_key]["C1"],
        colliding,
    ]
    manifest = FeatureManifest.from_dict(payload)

    with pytest.raises(ValueError, match="physical.*canonical|canonical.*alias"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


@pytest.mark.parametrize(
    "leak",
    (
        "margin",
        "total_points",
        "spread_line",
        "cover_prob",
        "game_id",
        "season",
        "home_team",
        "net_rating_diff",
    ),
)
def test_variant_physical_mapping_cannot_point_to_outcome_market_identity_or_canonical_data(leak):
    payload = _contract_manifest().to_dict()
    first = next(iter(payload["constants"]["rating_variant_columns"].values()))["margin"]
    first["rating_a"] = leak
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="exact.*physical|forbidden"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


def test_empty_input_rejects_duplicate_manifest_feature_labels_before_presence_checks():
    payload = _contract_manifest().to_dict()
    payload["margin_by_candidate"]["C1"].append("rating_a")
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="duplicate.*manifest.*rating_a"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


def test_empty_input_rejects_an_unsupported_manifest_candidate_before_presence_checks():
    payload = _contract_manifest().to_dict()
    payload["margin_by_candidate"]["C9"] = list(payload["margin_by_candidate"]["C1"])
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="unsupported.*C9"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


def test_non_c0_candidate_requires_c1_and_must_contain_the_complete_c1_contract():
    payload = _contract_manifest().to_dict()
    payload["margin_by_candidate"]["C2"] = list(FEATURE_COLS)
    payload["total_by_candidate"]["C2"] = list(FEATURE_COLS)
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="C2.*complete C1|C1.*C2"):
        nested_walk_forward_v2(_empty_contract_features(), [2024], manifest)


def test_later_candidate_without_a_declared_c1_contract_fails_closed():
    payload = _manifest().to_dict()
    payload["margin_by_candidate"].pop("C1")
    payload["total_by_candidate"].pop("C1")
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises(ValueError, match="requires.*C1"):
        nested_walk_forward_v2(_features().iloc[0:0], [2024], manifest)


@pytest.mark.parametrize("invalid", (None, "false", 0, 1))
def test_declared_c5_requires_a_real_boolean_eligibility_flag(invalid):
    manifest = _manifest(declare_c5=True, c5_eligible=invalid)
    with pytest.raises((TypeError, ValueError), match="c5_production_eligible.*bool"):
        nested_walk_forward_v2(_features().iloc[0:0], [2024], manifest)


def test_declared_c5_requires_the_eligibility_flag_to_be_present():
    payload = _manifest(declare_c5=True).to_dict()
    payload["constants"].pop("c5_production_eligible")
    manifest = FeatureManifest.from_dict(payload)
    with pytest.raises((TypeError, ValueError), match="c5_production_eligible.*bool"):
        nested_walk_forward_v2(_features().iloc[0:0], [2024], manifest)


@pytest.mark.parametrize("eligible, expected", ((False, False), (True, True)))
def test_c5_enters_the_grid_only_when_eligibility_is_explicitly_true(eligible, expected):
    fitter = _RecordingFitter()
    nested_walk_forward_v2(
        _features(through=2022),
        [2022],
        _manifest(declare_c5=True, c5_eligible=eligible),
        target_fitter=fitter,
    )
    assert any(call["config"].candidate == "C5" for call in fitter.calls) is expected


def _rows_for_seasons(config, by_season: dict[int, float], n_games: int = 200):
    return [
        {"config": config, "validation_season": season, "mae": mae, "n_games": n_games}
        for season, mae in sorted(by_season.items())
    ]


def test_candidates_are_scored_on_the_season_set_they_all_share():
    """The 2026-08-25 defect: C0 was allowed to skip degenerate seasons and its rivals were not.

    `_inner_evaluations` lets C0 `continue` past a DegenerateFeatureError while any other
    candidate re-raises, so in the real run C0 was scored from 2019 while C1-C5 carried 2017 and
    2018 as well -- seasons whose MAE runs 17-25. A plain mean over each config's own seasons
    then compares an easy denominator against a hard one, and C0 wins on the cut rather than on
    merit.

    Shaped exactly like that here: C2 is worse than C0 on the two catastrophic seasons it alone
    is charged for, and better than C0 on every season the two share.
    """
    c0 = _config("C0")
    c2 = _config("C2")
    scores = pd.DataFrame(
        [
            *_rows_for_seasons(c0, {2019: 10.0, 2020: 10.0, 2021: 10.0}),
            *_rows_for_seasons(c2, {2017: 20.0, 2018: 20.0, 2019: 9.0, 2020: 9.0, 2021: 9.0}),
        ]
    )

    selected = select_target_config(scores, target="margin")

    assert selected is not None
    # Scored on their own season sets C0 wins 10.00 to 13.40; on the shared set C2 wins 9.0.
    assert selected.config.candidate == "C2"
    assert selected.mean_inner_mae == pytest.approx(9.0)
    assert selected.validation_seasons == (2019, 2020, 2021)


def test_a_config_scored_on_no_shared_season_is_ineligible():
    """A candidate that shares nothing with the others cannot be compared, so it cannot win."""
    scores = pd.DataFrame(
        [
            *_rows_for_seasons(_config("C0"), {2019: 1.0, 2020: 1.0, 2021: 1.0}),
            *_rows_for_seasons(_config("C4"), {2016: 0.001, 2017: 0.001}),
        ]
    )

    selected = select_target_config(scores, target="margin")

    assert selected is not None
    assert selected.config.candidate == "C0"


def test_common_season_set_still_applies_the_eligibility_floors():
    """Intersecting must not smuggle in a config that no longer clears MIN_INNER_SEASONS."""
    scores = pd.DataFrame(
        [
            *_rows_for_seasons(_config("C0"), {2019: 5.0, 2020: 5.0, 2021: 5.0}),
            *_rows_for_seasons(_config("C4"), {2021: 0.001}),
        ]
    )

    selected = select_target_config(scores, target="margin")

    assert selected is not None
    assert selected.config.candidate == "C0"
    assert selected.validation_seasons == (2019, 2020, 2021)
