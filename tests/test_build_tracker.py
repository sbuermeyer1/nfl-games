import pandas as pd
import pytest
from scripts import build_tracker


def predictions():
    return pd.DataFrame(
        {
            "game_id": ["2025_01_AAA_BBB", "2025_01_CCC_DDD"],
            "season": [2025, 2025],
            "week": [1, 1],
            "away_team": ["AAA", "CCC"],
            "home_team": ["BBB", "DDD"],
            "model_margin": [7.0, -2.0],
            "model_total": [48.0, 40.0],
            "spread_line": [3.0, 1.0],
            "total_line": [44.0, 44.0],
            "margin": [8.0, -3.0],
            "total_points": [50.0, 38.0],
        }
    )


def test_builder_is_ridge_only_and_forwards_the_requested_seasons(monkeypatch):
    calls = []

    def fake_walk_forward(features, test_seasons, estimator, alpha):
        calls.append((features.copy(), test_seasons, estimator, alpha))
        return predictions()

    monkeypatch.setattr(build_tracker, "walk_forward", fake_walk_forward)
    expected = {
        "games": 2,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    features = pd.DataFrame({"sentinel": [1]})
    ledger = build_tracker.build_historical_ledger(
        features,
        test_seasons=[2025],
        model_version="ridge-v1",
        expected_baseline=expected,
    )

    assert calls[0][1:] == ([2025], "ridge", 1.0)
    assert set(ledger["record_type"]) == {"backtest"}
    assert set(ledger["model_version"]) == {"ridge-v1"}


def test_acceptance_gate_rejects_any_corpus_or_hit_rate_drift():
    ledger = build_tracker.build_backtest_ledger(predictions())
    wrong = {
        "games": 3,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(ledger, wrong)


def test_cli_writes_the_validated_ledger(tmp_path, monkeypatch):
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "tracker.parquet"
    pd.DataFrame({"sentinel": [1]}).to_parquet(features_path)
    monkeypatch.setattr(
        build_tracker,
        "build_historical_ledger",
        lambda *args, **kwargs: build_tracker.build_backtest_ledger(predictions()),
    )

    build_tracker.main(["--features", str(features_path), "--output", str(output_path)])

    written = pd.read_parquet(output_path)
    assert written["game_id"].tolist() == predictions()["game_id"].tolist()
