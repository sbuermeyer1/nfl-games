from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backtest_v2_ftn

from nfl_game.model.features import FEATURE_COLS
from nfl_game.paths import PROCESSED_DIR

RNG = np.random.default_rng(11)


def _features(seasons=(2022, 2023, 2024)) -> pd.DataFrame:
    rows = []
    for season in seasons:
        for week in (1, 2, 3, 4):
            for home, away in (("BUF", "LA"), ("KC", "MIA")):
                row = {
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "margin": float(RNG.normal(0, 10)),
                    "total_points": float(RNG.normal(45, 10)),
                    "spread_line": 1.5,
                    "total_line": 44.0,
                }
                for name in FEATURE_COLS:
                    row[name] = float(RNG.normal())
                rows.append(row)
    return pd.DataFrame(rows)


def _charted(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    play_id = 0
    for _, game in features.iterrows():
        for team in (game["home_team"], game["away_team"]):
            for _ in range(6):
                play_id += 1
                rows.append(
                    {
                        "nflverse_game_id": game["game_id"],
                        "nflverse_play_id": play_id,
                        "season": game["season"],
                        "week": game["week"],
                        "team": team,
                        "is_motion": bool(RNG.integers(0, 2)),
                        "is_play_action": bool(RNG.integers(0, 2)),
                        "is_rpo": False,
                        "is_screen_pass": False,
                        "is_qb_out_of_pocket": False,
                        "is_interception_worthy": False,
                        "is_catchable_ball": True,
                        "is_drop": False,
                        "is_qb_fault_sack": False,
                        "n_blitzers": int(RNG.integers(0, 3)),
                        "n_pass_rushers": 4,
                    }
                )
    return pd.DataFrame(rows)


def _loaders(features: pd.DataFrame) -> dict[str, object]:
    charted = _charted(features)
    pbp = charted.rename(
        columns={
            "nflverse_game_id": "game_id",
            "nflverse_play_id": "play_id",
            "team": "posteam",
        }
    )[["game_id", "play_id", "posteam", "season", "week"]]
    return {
        "load_ftn_charting": lambda seasons: charted.drop(columns=["team"]),
        "load_pbp": lambda seasons: pbp,
    }


def test_outer_seasons_require_a_prior_ftn_season():
    features = _features()

    assert backtest_v2_ftn.eligible_outer_seasons(features, 2023) == [2023, 2024]

    with pytest.raises(ValueError, match="no prior FTN history"):
        backtest_v2_ftn.eligible_outer_seasons(features, 2022)


def test_the_two_arms_train_on_identical_rows_and_differ_only_by_the_ftn_columns():
    features = _features()
    merged, ftn_columns, team_games = backtest_v2_ftn.build_ftn_game_features(
        features, loaders=_loaders(features)
    )
    # Team-games, not game rows: two teams chart plays in every game.
    assert team_games == 2 * len(features)

    results = backtest_v2_ftn.paired_arm_results(merged, ftn_columns, [2023, 2024])

    assert set(results["target"]) == {"margin", "total_points"}
    # One n_train per season, shared by both arms: the arms cannot differ by training set.
    for season, group in results.groupby("outer_season"):
        assert group["n_train"].nunique() == 1
        assert group["n_test"].nunique() == 1
        assert group["train_seasons"].iloc[0] == f"2022-{season - 1}"
    assert set(results["label"]) == {"E1-research"}


def test_ftn_contribution_is_signed_so_a_worse_e1_arm_shows_negative():
    results = pd.DataFrame(
        [
            {"core_mae": 10.0, "e1_mae": 10.5},
            {"core_mae": 10.0, "e1_mae": 9.5},
        ]
    )
    contribution = results["core_mae"] - results["e1_mae"]

    assert contribution.iloc[0] == pytest.approx(-0.5)
    assert contribution.iloc[1] == pytest.approx(0.5)


def test_the_research_report_refuses_every_task_13_to_17_artifact_and_ridge_v1():
    protected = [
        "game_features.parquet",
        "tracker_ledger.parquet",
        "schedule_2026.parquet",
        "game_features_ridge_v2.parquet",
        "ridge_v2_manifest.json",
        "ridge_v2_outer_predictions.parquet",
        "ridge_v2_evaluation.json",
        "ridge_v2_ablation.parquet",
        "ridge_v2_calibration.json",
        "tracker_ledger_ridge_v2.parquet",
    ]

    assert set(protected) == set(backtest_v2_ftn.FORBIDDEN_OUTPUT_NAMES)

    for name in protected:
        with pytest.raises(ValueError, match="refusing to write"):
            backtest_v2_ftn.main(["--report", str(PROCESSED_DIR / name), "--write"])


def test_default_run_writes_nothing_and_write_produces_a_research_only_report(tmp_path):
    features = _features()
    deps = {"features": features, "loaders": _loaders(features)}
    report = tmp_path / "e1.json"

    dry = backtest_v2_ftn.main(["--report", str(report)], dependencies=deps)
    assert dry == 0
    assert not report.exists()

    written = backtest_v2_ftn.main(["--report", str(report), "--write"], dependencies=deps)
    assert written == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["label"] == "E1-research"
    assert payload["production_eligible"] is False
    assert payload["outer_seasons"] == [2023, 2024]
    assert len(payload["results"]) == 4


def test_report_names_every_season_and_the_pooled_direction(tmp_path, capsys):
    features = _features()
    deps = {"features": features, "loaders": _loaders(features)}

    backtest_v2_ftn.main(["--report", str(tmp_path / "e1.json")], dependencies=deps)
    out = capsys.readouterr().out

    assert "E1-research" in out
    assert "2023" in out and "2024" in out
    assert "FTN better in" in out
    assert "Ridge v1 remains official" in out
