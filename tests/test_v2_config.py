import json
from typing import get_args

import pytest

from nfl_game.model.v2_config import (
    ALPHAS,
    CANDIDATES,
    MARKET_COLUMNS,
    PRIOR_SEASON_WEIGHTS,
    RATING_WINDOWS,
    CandidateId,
    FeatureManifest,
    TargetConfig,
    target_tuning_grid,
)
from nfl_game.paths import (
    V2_ABLATION_PATH,
    V2_CALIBRATION_PATH,
    V2_EVALUATION_PATH,
    V2_FEATURES_PATH,
    V2_MANIFEST_PATH,
    V2_OUTER_PREDICTIONS_PATH,
    V2_TRACKER_LEDGER_PATH,
)


def test_target_grid_is_fixed_and_deterministic():
    grid = target_tuning_grid("C2")
    assert len(grid) == 4 * 3 * 3
    assert grid[0].candidate == "C2"
    assert {c.alpha for c in grid} == {0.1, 1.0, 10.0, 100.0}
    assert {(c.short_halflife, c.long_halflife) for c in grid} == {
        (4, 16),
        (8, 24),
        (12, 32),
    }
    assert {c.prior_season_weight for c in grid} == {0.4, 0.6, 0.8}


def test_candidate_contracts_are_fixed():
    assert CANDIDATES == ("C0", "C1", "C2", "C3", "C4", "C5")
    assert get_args(CandidateId) == CANDIDATES
    assert ALPHAS == (0.1, 1.0, 10.0, 100.0)
    assert RATING_WINDOWS == ((4, 16), (8, 24), (12, 32))
    assert PRIOR_SEASON_WEIGHTS == (0.4, 0.6, 0.8)
    assert MARKET_COLUMNS == frozenset(
        {"spread_line", "total_line", "away_moneyline", "home_moneyline"}
    )


def test_target_config_key_is_stable_json():
    config = TargetConfig("C2", 1.0, 8, 24, 0.6)
    assert config.key() == (
        '{"alpha":1.0,"candidate":"C2","long_halflife":24,'
        '"prior_season_weight":0.6,"short_halflife":8}'
    )


def test_manifest_rejects_market_features():
    with pytest.raises(ValueError, match="market column"):
        FeatureManifest(
            version="ridge-v2-test",
            margin_by_candidate={"C1": ("spread_line",)},
            total_by_candidate={"C1": ("pace_sum",)},
            sources={},
            constants={},
        )


def test_manifest_round_trips_through_json_compatible_dictionary():
    manifest = FeatureManifest(
        version="ridge-v2-test",
        margin_by_candidate={"C1": ("rating_diff", "rest_diff")},
        total_by_candidate={"C1": ("pace_sum",)},
        sources={"weather": "nflverse"},
        constants={"min_games": 400, "enabled": True},
    )

    payload = json.loads(json.dumps(manifest.to_dict()))
    restored = FeatureManifest.from_dict(payload)

    assert restored == manifest
    assert restored.columns("margin", "C1") == ("rating_diff", "rest_diff")


def test_v2_artifact_paths_use_approved_filenames():
    assert V2_FEATURES_PATH.name == "game_features_ridge_v2.parquet"
    assert V2_MANIFEST_PATH.name == "ridge_v2_manifest.json"
    assert V2_OUTER_PREDICTIONS_PATH.name == "ridge_v2_outer_predictions.parquet"
    assert V2_EVALUATION_PATH.name == "ridge_v2_evaluation.json"
    assert V2_ABLATION_PATH.name == "ridge_v2_ablation.parquet"
    assert V2_CALIBRATION_PATH.name == "ridge_v2_calibration.json"
    assert V2_TRACKER_LEDGER_PATH.name == "tracker_ledger_ridge_v2.parquet"
