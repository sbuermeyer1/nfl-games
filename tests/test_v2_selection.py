import math
from dataclasses import replace

import numpy as np
import pandas as pd

from nfl_game.experiments.v2_selection import (
    MIN_INNER_GAMES,
    MIN_INNER_SEASONS,
    ONE_STANDARD_ERROR_TOLERANCE,
    nested_walk_forward_v2,
    select_target_config,
)
from nfl_game.model.v2_config import FeatureManifest, TargetConfig


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


def _manifest() -> FeatureManifest:
    columns = {candidate: ("signal",) for candidate in ("C0", "C2", "C4")}
    return FeatureManifest(
        version="selection-test",
        margin_by_candidate=columns,
        total_by_candidate=columns,
        sources={},
        constants={"c5_production_eligible": False},
    )


def _features(*, through: int = 2024, games_per_season: int = 200) -> pd.DataFrame:
    rows = []
    for season in range(2019, through + 1):
        for game in range(games_per_season):
            signal = float((game % 31) - 15)
            rows.append(
                {
                    "game_id": f"{season}-{game:03d}",
                    "season": season,
                    "week": 1 + game // 16,
                    "signal": signal,
                    "margin": signal,
                    "total_points": signal,
                }
            )
    return pd.DataFrame(rows)


class _OffsetPredictor:
    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return matrix["signal"].to_numpy(dtype=float) + self.offset


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


def test_nested_selection_never_reads_validation_or_outer_seasons():
    fitter = _RecordingFitter()

    nested_walk_forward_v2(_features(), [2024], _manifest(), target_fitter=fitter)

    assert fitter.calls
    assert all(
        not call["training_seasons"] or max(call["training_seasons"]) < call["validation_season"]
        for call in fitter.calls
    )
    assert all(2024 not in call["training_seasons"] for call in fitter.calls)


def test_poisoned_future_season_cannot_change_earlier_selection_or_predictions():
    errors = {
        ("margin", "C2", 0.1): 0.2,
        ("total_points", "C4", 10.0): 0.1,
    }
    clean = nested_walk_forward_v2(
        _features(), [2024], _manifest(), target_fitter=_RecordingFitter(errors)
    )
    poisoned_features = _features(through=2025)
    poisoned_features.loc[poisoned_features["season"] == 2025, ["signal", "margin"]] = 1e12
    poisoned = nested_walk_forward_v2(
        poisoned_features,
        [2024],
        _manifest(),
        target_fitter=_RecordingFitter(errors),
    )

    assert clean.selections == poisoned.selections
    pd.testing.assert_frame_equal(clean.predictions, poisoned.predictions)


def test_margin_and_total_select_independent_candidates_and_alphas():
    errors = {
        ("margin", "C2", 0.1): 0.1,
        ("total_points", "C4", 10.0): 0.1,
    }

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


def test_seed_fold_uses_honest_c0_fallback_without_v2_evidence():
    result = nested_walk_forward_v2(
        _features(through=2020),
        [2020],
        _manifest(),
        target_fitter=_RecordingFitter(),
    )

    selection = result.selections[0]
    for target_selection in (selection.margin, selection.total):
        assert target_selection.config.candidate == "C0"
        assert math.isnan(target_selection.mean_inner_mae)
        assert target_selection.validation_seasons == ()
        assert target_selection.validation_games == 0
    assert len(result.predictions) == 200
