from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backtest_v2

from nfl_game.experiments.v2_evaluation import REPORT_SEASONS
from nfl_game.paths import PROCESSED_DIR

CALIBRATION_SEEDS = (2019, 2020)
ALL_SEASONS = (*CALIBRATION_SEEDS, *REPORT_SEASONS)


def _predictions(offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for season in ALL_SEASONS:
        for week in (1, 2, 3, 4):
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_AAA_BBB",
                    "season": season,
                    "week": week,
                    "margin": 3.0 + week,
                    "total_points": 44.0 + week,
                    "spread_line": 2.5,
                    "total_line": 43.5,
                    "model_margin": 3.0 + week + offset,
                    "model_total": 44.0 + week + offset,
                }
            )
    return pd.DataFrame(rows)


def _probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    report = frame.loc[frame["season"].isin(REPORT_SEASONS)]
    return pd.DataFrame(
        {
            "game_id": report["game_id"].to_numpy(),
            "cover_prob": np.full(len(report), 0.5),
            "over_prob": np.full(len(report), 0.5),
        }
    )


def _passing_report() -> dict[str, object]:
    per_season = {
        str(season): {"n_games": 4, "margin_improvement": 0.4, "total_improvement": 0.3}
        for season in REPORT_SEASONS
    }
    return {
        "report_seasons": list(REPORT_SEASONS),
        "per_season": per_season,
        "n_games": 20,
        "margin_mae": 9.5,
        "total_mae": 10.0,
        "margin_seasons_improved": 5,
        "total_seasons_improved": 5,
        "margin_paired_improvement_lower90": 0.2,
        "total_paired_improvement_lower90": 0.1,
        "margin_market_model_coef": 0.3,
        "margin_market_model_coef_lower90": 0.1,
        "total_market_model_coef": 0.4,
        "total_market_model_coef_lower90": 0.2,
        "ats_hit_rate": 0.55,
        "ou_hit_rate": 0.54,
        "cover_brier": 0.20,
        "v1_cover_brier": 0.25,
        "over_brier": 0.21,
        "v1_over_brier": 0.26,
        "correctness_passed": True,
        "availability_passed": True,
        "determinism_passed": True,
        "source_reliability_passed": True,
    }


def _failing_report() -> dict[str, object]:
    report = _passing_report()
    report["margin_mae"] = 12.0
    return report


@dataclass
class _Result:
    predictions: pd.DataFrame
    selections: tuple


def _ablation_frame() -> pd.DataFrame:
    """Both ablation outcomes, in the schema remove_one_block_ablations really emits."""
    return pd.DataFrame(
        [
            {
                "season": 2021,
                "target": "margin",
                "candidate": "C4",
                "removed_block": "C2",
                "n_removed_columns": 8,
                "n_kept_columns": 41,
                "mae_full": 10.0,
                "mae_ablated": 10.2,
                "block_contribution": 0.2,
                "status": "measured",
                "detail": "",
            },
            {
                "season": 2021,
                "target": "margin",
                "candidate": "C4",
                "removed_block": "C1",
                "n_removed_columns": 13,
                "n_kept_columns": 36,
                "mae_full": 10.0,
                "mae_ablated": float("nan"),
                "block_contribution": float("nan"),
                "status": "not_constructible",
                "detail": "ValueError: rating variant maps canonical column(s) outside the schema",
            },
        ]
    )


def _fake_dependencies(report: dict[str, object] | None = None, **overrides):
    v2 = _predictions(offset=0.1)
    deps = {
        "v1_features": pd.DataFrame({"game_id": ["x"], "season": [2019]}),
        "v2_features": pd.DataFrame({"game_id": ["x"], "season": [2019]}),
        "manifest": object(),
        "manifest_payload": {"output": {}, "source_snapshots": []},
        "v1_walk_forward": lambda features, seasons: _predictions(offset=0.5),
        "nested_walk_forward": lambda features, seasons, manifest: _Result(v2, ()),
        "probabilities": lambda predictions, fit_observer=None: _probabilities(predictions),
        "evaluate": lambda **kwargs: dict(report or _passing_report()),
        "ablations": lambda **kwargs: _ablation_frame(),
        "quality_checks": lambda **kwargs: {
            "correctness": True,
            "availability": True,
            "determinism": True,
            "source_reliability": True,
        },
    }
    deps.update(overrides)
    return deps


def _output_args(tmp_path: Path) -> list[str]:
    return [
        "--predictions",
        str(tmp_path / "ridge_v2_outer_predictions.parquet"),
        "--evaluation",
        str(tmp_path / "ridge_v2_evaluation.json"),
        "--ablation",
        str(tmp_path / "ridge_v2_ablation.parquet"),
        "--calibration",
        str(tmp_path / "ridge_v2_calibration.json"),
    ]


def test_cli_never_targets_v1_artifacts():
    args = backtest_v2._parser().parse_args([])
    outputs = {args.predictions, args.evaluation, args.ablation, args.calibration}
    assert PROCESSED_DIR / "game_features.parquet" not in outputs
    assert PROCESSED_DIR / "tracker_ledger.parquet" not in outputs
    assert PROCESSED_DIR / "schedule_2026.parquet" not in outputs
    assert len(outputs) == 4


def test_default_run_is_a_dry_run_that_writes_nothing(tmp_path, capsys):
    exit_code = backtest_v2.main(_output_args(tmp_path), dependencies=_fake_dependencies())

    assert exit_code == 0
    assert list(tmp_path.iterdir()) == []
    assert "dry-run: no artifacts written" in capsys.readouterr().out


def test_write_publishes_all_four_research_artifacts(tmp_path):
    exit_code = backtest_v2.main(
        [*_output_args(tmp_path), "--write"], dependencies=_fake_dependencies()
    )

    assert exit_code == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "ridge_v2_ablation.parquet",
        "ridge_v2_calibration.json",
        "ridge_v2_evaluation.json",
        "ridge_v2_outer_predictions.parquet",
    ]
    predictions = pd.read_parquet(tmp_path / "ridge_v2_outer_predictions.parquet")
    assert set(predictions["season"]) == set(REPORT_SEASONS)
    # The exact contract, in order: no feature columns ride along into this artifact.
    assert list(predictions.columns) == [
        *backtest_v2.PREDICTION_COLUMNS,
        "cover_prob",
        "over_prob",
    ]
    evaluation = json.loads((tmp_path / "ridge_v2_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["report"]["margin_mae"] == 9.5
    assert evaluation["gates"]["1"]["status"] == "PASS"
    assert evaluation["gates"]["11"]["status"] == "PENDING"


def test_failed_gate_exits_nonzero_only_when_required(tmp_path):
    quiet = backtest_v2.main(
        _output_args(tmp_path), dependencies=_fake_dependencies(_failing_report())
    )
    strict = backtest_v2.main(
        [*_output_args(tmp_path), "--require-research-gates"],
        dependencies=_fake_dependencies(_failing_report()),
    )
    passing_strict = backtest_v2.main(
        [*_output_args(tmp_path), "--require-research-gates"],
        dependencies=_fake_dependencies(),
    )

    assert quiet == 0
    assert strict == 1
    assert passing_strict == 0


def test_gate_report_prints_every_gate_with_its_status(tmp_path, capsys):
    backtest_v2.main(_output_args(tmp_path), dependencies=_fake_dependencies(_failing_report()))
    out = capsys.readouterr().out

    statuses = {
        int(line.split()[1]): line.split()[2]
        for line in out.splitlines()
        if line.startswith("gate ")
    }

    assert sorted(statuses) == list(range(1, 12))
    # Read the gate lines themselves: asserting only that "FAIL" appears somewhere passes
    # on the failure summary line even when every gate is printed as PASS.
    assert statuses[1] == "FAIL"  # the injected failure is margin MAE
    assert statuses[2] == "PASS"  # total MAE is untouched and must not be tarred with it
    assert statuses[11] == "PENDING"  # the shadow rebuild cannot run in this task
    assert "margin MAE" in out


def test_write_restores_originals_when_a_later_artifact_fails(tmp_path):
    for name in ("ridge_v2_outer_predictions.parquet", "ridge_v2_ablation.parquet"):
        pd.DataFrame({"original": [1]}).to_parquet(tmp_path / name, index=False)
    for name in ("ridge_v2_evaluation.json", "ridge_v2_calibration.json"):
        (tmp_path / name).write_text('{"original": true}', encoding="utf-8")
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def explode(source: Path, destination: Path) -> None:
        """Fail on the third publication, after two destinations were already replaced."""
        if destination.name == "ridge_v2_ablation.parquet":
            raise RuntimeError("publication failed")
        source.replace(destination)

    with pytest.raises(RuntimeError, match="publication failed"):
        backtest_v2.main(
            [*_output_args(tmp_path), "--write"],
            dependencies=_fake_dependencies(replace_file=explode),
        )

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == originals


def test_write_refuses_a_ridge_v1_destination(tmp_path):
    args = [
        "--predictions",
        str(tmp_path / "game_features.parquet"),
        "--evaluation",
        str(tmp_path / "ridge_v2_evaluation.json"),
        "--ablation",
        str(tmp_path / "ridge_v2_ablation.parquet"),
        "--calibration",
        str(tmp_path / "ridge_v2_calibration.json"),
        "--write",
    ]

    with pytest.raises(ValueError, match="frozen Ridge-v1"):
        backtest_v2.main(args, dependencies=_fake_dependencies())

    assert not (tmp_path / "game_features.parquet").exists()


def test_parser_rejects_combined_dry_run_and_write():
    with pytest.raises(SystemExit):
        backtest_v2._parser().parse_args(["--dry-run", "--write"])


def test_quality_checks_are_measured_from_the_run_not_asserted(tmp_path):
    """A missing outer game must reach the gate as a failure, not be assumed passing."""
    incomplete = _predictions(offset=0.1)
    incomplete = incomplete.loc[incomplete["game_id"] != "2021_01_AAA_BBB"]
    captured: dict[str, object] = {}

    def evaluate(**kwargs):
        captured.update(kwargs["quality_checks"])
        report = _passing_report()
        report.update(
            {f"{key}_passed": bool(value) for key, value in kwargs["quality_checks"].items()}
        )
        return report

    deps = _fake_dependencies(
        nested_walk_forward=lambda features, seasons, manifest: _Result(incomplete, ()),
        evaluate=evaluate,
        quality_checks=backtest_v2.measure_quality_checks,
    )
    exit_code = backtest_v2.main(
        [*_output_args(tmp_path), "--require-research-gates"], dependencies=deps
    )

    assert captured["availability"] is False
    assert exit_code == 1
