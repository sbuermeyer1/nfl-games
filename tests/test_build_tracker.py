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


def predictions_with_pushes():
    frame = predictions()
    frame.loc[0, ["margin", "total_points"]] = [3.0, 44.0]
    return frame


def test_builder_is_ridge_only_and_forwards_the_requested_seasons(monkeypatch):
    calls = []

    def fake_walk_forward(features, test_seasons, estimator, alpha):
        calls.append((features.copy(), test_seasons, estimator, alpha))
        return predictions()

    monkeypatch.setattr(build_tracker, "walk_forward", fake_walk_forward)
    expected = {
        "games": 2,
        "ats_wins": 2,
        "ats_losses": 0,
        "ats_pushes": 0,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_wins": 2,
        "ou_losses": 0,
        "ou_pushes": 0,
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
        "ats_wins": 2,
        "ats_losses": 0,
        "ats_pushes": 0,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_wins": 2,
        "ou_losses": 0,
        "ou_pushes": 0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(ledger, wrong)


def test_acceptance_gate_rejects_spread_push_becoming_pending():
    accepted = build_tracker.build_backtest_ledger(predictions_with_pushes())
    expected = build_tracker.acceptance_metrics(accepted)
    drifted_predictions = predictions_with_pushes()
    drifted_predictions.loc[0, "margin"] = pd.NA
    drifted = build_tracker.build_backtest_ledger(drifted_predictions)

    assert accepted.loc[0, "spread_grade"] == "push"
    assert drifted.loc[0, "spread_grade"] == "pending"
    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(drifted, expected)


def test_acceptance_gate_rejects_total_push_becoming_no_pick():
    accepted = build_tracker.build_backtest_ledger(predictions_with_pushes())
    expected = build_tracker.acceptance_metrics(accepted)
    drifted_predictions = predictions_with_pushes()
    drifted_predictions.loc[0, "model_total"] = drifted_predictions.loc[0, "total_line"]
    drifted = build_tracker.build_backtest_ledger(drifted_predictions)

    assert accepted.loc[0, "total_grade"] == "push"
    assert drifted.loc[0, "total_grade"] == "no_pick"
    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(drifted, expected)


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


@pytest.mark.parametrize(
    "actual",
    [
        {
            "games": 2,
            "ats_wins": 2,
            "ats_losses": 0,
            "ats_pushes": 0,
            "ats_hit_rate": 1.0,
            "ou_wins": 2,
            "ou_losses": 0,
            "ou_pushes": 0,
            "ou_n": 2,
            "ou_hit_rate": 1.0,
        },
        {
            "games": 2,
            "ats_wins": 2,
            "ats_losses": 0,
            "ats_pushes": 0,
            "ats_n": 2,
            "ats_hit_rate": 1.0,
            "ou_wins": 2,
            "ou_losses": 0,
            "ou_pushes": 0,
            "ou_n": 2,
            "ou_hit_rate": 1.0,
            "unexpected": 0,
        },
    ],
)
def test_acceptance_gate_rejects_missing_or_extra_actual_metrics(monkeypatch, actual):
    expected = {
        "games": 2,
        "ats_wins": 2,
        "ats_losses": 0,
        "ats_pushes": 0,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_wins": 2,
        "ou_losses": 0,
        "ou_pushes": 0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    monkeypatch.setattr(build_tracker, "acceptance_metrics", lambda ledger: actual)

    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(pd.DataFrame(), expected)


@pytest.mark.parametrize(
    "expected",
    [
        {
            "games": 2,
            "ats_wins": 2,
            "ats_losses": 0,
            "ats_pushes": 0,
            "ats_hit_rate": 1.0,
            "ou_wins": 2,
            "ou_losses": 0,
            "ou_pushes": 0,
            "ou_n": 2,
            "ou_hit_rate": 1.0,
        },
        {
            "games": 2,
            "ats_wins": 2,
            "ats_losses": 0,
            "ats_pushes": 0,
            "ats_n": 2,
            "ats_hit_rate": 1.0,
            "ou_wins": 2,
            "ou_losses": 0,
            "ou_pushes": 0,
            "ou_n": 2,
            "ou_hit_rate": 1.0,
            "unexpected": 0,
        },
    ],
)
def test_acceptance_gate_rejects_missing_or_extra_expected_metrics(monkeypatch, expected):
    actual = {
        "games": 2,
        "ats_wins": 2,
        "ats_losses": 0,
        "ats_pushes": 0,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_wins": 2,
        "ou_losses": 0,
        "ou_pushes": 0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    monkeypatch.setattr(build_tracker, "acceptance_metrics", lambda ledger: actual)

    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(pd.DataFrame(), expected)


def _early_lines():
    """Only the first game is priced, so the excluded path is exercised too."""
    return pd.DataFrame(
        {
            "game_id": ["2025_01_AAA_BBB"],
            "early_spread_line": [1.0],
            "early_total_line": [46.0],
            "snapshot_at": [pd.Timestamp("2025-09-02T00:20:00Z")],
        }
    )


def _patch_walk_forward(monkeypatch, frame=None):
    monkeypatch.setattr(
        build_tracker,
        "walk_forward",
        lambda features, test_seasons, estimator, alpha: (
            predictions() if frame is None else frame
        ),
    )


def test_early_line_build_ships_early_grades_and_keeps_the_legacy_gate(monkeypatch):
    """The shipped ledger is graded at the early line; the legacy corpus is still asserted."""
    _patch_walk_forward(monkeypatch)
    legacy_seen = {}

    real_assert = build_tracker.assert_acceptance_baseline

    def spy(ledger, expected, metrics=None):
        legacy_seen.setdefault("calls", []).append(
            (ledger["official_spread_line"].tolist(), metrics)
        )
        return real_assert(ledger, expected, metrics)

    monkeypatch.setattr(build_tracker, "assert_acceptance_baseline", spy)
    monkeypatch.setattr(build_tracker, "EXPECTED_BASELINE", build_tracker.acceptance_metrics(
        build_tracker.build_backtest_ledger(predictions())
    ))
    early_ledger = build_tracker.build_backtest_ledger(
        predictions(), early_lines=_early_lines()
    )
    monkeypatch.setattr(
        build_tracker, "EXPECTED_EARLY_BASELINE",
        build_tracker.early_acceptance_metrics(early_ledger),
    )

    out = build_tracker.build_historical_ledger(
        pd.DataFrame({"season": [2025]}),
        expected_baseline=build_tracker.EXPECTED_BASELINE,
        early_lines=_early_lines(),
        expected_early_baseline=build_tracker.EXPECTED_EARLY_BASELINE,
    )

    # Two assertions ran: the legacy build on closing lines, then the shipped early build.
    assert len(legacy_seen["calls"]) == 2
    assert legacy_seen["calls"][0][0] == [3.0, 1.0]  # legacy = closing lines
    assert legacy_seen["calls"][0][1] is None
    assert legacy_seen["calls"][1][1] is build_tracker.early_acceptance_metrics
    # And the ledger returned is the early-line one, not the legacy one.
    assert out.loc[out["game_id"].eq("2025_01_AAA_BBB"), "official_spread_line"].iloc[0] == 1.0


def test_early_line_build_still_fails_on_legacy_corpus_drift(monkeypatch):
    """Grading at the early line must not weaken the original guard."""
    _patch_walk_forward(monkeypatch)
    drifted = dict(build_tracker.EXPECTED_BASELINE)
    drifted["ats_wins"] = drifted["ats_wins"] + 1

    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.build_historical_ledger(
            pd.DataFrame({"season": [2025]}),
            expected_baseline=drifted,
            early_lines=_early_lines(),
        )


def test_early_baseline_gate_rejects_a_change_in_the_excluded_count(monkeypatch):
    """A line-history rebuild that lost coverage is the quietest way this artifact could move."""
    _patch_walk_forward(monkeypatch)
    legacy = build_tracker.acceptance_metrics(build_tracker.build_backtest_ledger(predictions()))
    early_ledger = build_tracker.build_backtest_ledger(
        predictions(), early_lines=_early_lines()
    )
    expected = build_tracker.early_acceptance_metrics(early_ledger)
    assert expected["excluded_spread"] == 1, "fixture must exclude a game or this cannot fail"
    expected["excluded_spread"] = 0

    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.build_historical_ledger(
            pd.DataFrame({"season": [2025]}),
            expected_baseline=legacy,
            early_lines=_early_lines(),
            expected_early_baseline=expected,
        )


def test_historical_baseline_selects_the_record_the_corpus_actually_carries(monkeypatch):
    """The live tracker reads a persisted ledger; it must accept either build, and only that one.

    Getting this wrong halts the 15-minute cron for the season, so both directions are pinned.
    """
    legacy = build_tracker.build_backtest_ledger(predictions())
    early = build_tracker.build_backtest_ledger(predictions(), early_lines=_early_lines())
    monkeypatch.setattr(build_tracker, "EXPECTED_BASELINE",
                        build_tracker.acceptance_metrics(legacy))
    monkeypatch.setattr(build_tracker, "EXPECTED_EARLY_BASELINE",
                        build_tracker.early_acceptance_metrics(early))

    build_tracker.assert_historical_baseline(legacy)
    build_tracker.assert_historical_baseline(early)

    # Each corpus must be rejected against the OTHER record, or the selection does nothing.
    monkeypatch.setattr(build_tracker, "EXPECTED_BASELINE",
                        build_tracker.acceptance_metrics(early))
    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_historical_baseline(legacy)


def test_historical_baseline_rejects_an_early_corpus_that_drifted(monkeypatch):
    early = build_tracker.build_backtest_ledger(predictions(), early_lines=_early_lines())
    drifted = dict(build_tracker.early_acceptance_metrics(early))
    drifted["ats_wins"] = drifted["ats_wins"] + 1
    monkeypatch.setattr(build_tracker, "EXPECTED_EARLY_BASELINE", drifted)

    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_historical_baseline(early)
