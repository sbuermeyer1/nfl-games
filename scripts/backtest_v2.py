"""Locked Ridge-v2 research experiment: nested backtest, promotion gates, four artifacts.

This script is research only. It never touches a Ridge-v1 artifact, the tracker, the web
package, or any workflow: Ridge v1 stays official until every gate passes and the user
approves promotion separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.backtest import evaluate as evaluate_v1
from nfl_game.backtest import walk_forward
from nfl_game.data.source_manifest import schema_fingerprint
from nfl_game.experiments.v2_evaluation import (
    CALIBRATION_SEASONS,
    REPORT_SEASONS,
    evaluate_v2,
    research_gate_decision,
    walk_forward_probabilities,
)
from nfl_game.experiments.v2_selection import nested_walk_forward_v2
from nfl_game.model.v2_config import CANDIDATES, FeatureManifest
from nfl_game.paths import (
    PROCESSED_DIR,
    V2_ABLATION_PATH,
    V2_CALIBRATION_PATH,
    V2_EVALUATION_PATH,
    V2_FEATURES_PATH,
    V2_MANIFEST_PATH,
    V2_OUTER_PREDICTIONS_PATH,
)
from nfl_game.pipeline.build_v2 import PROTECTED_V1_ARTIFACT_NAMES, semantic_frame_digest

V1_FEATURES_PATH = PROCESSED_DIR / "game_features.parquet"

# The frozen Ridge-v1 acceptance baseline (CLAUDE.md "Regression baseline"). Gate 10's
# correctness check reproduces it inside this run rather than trusting the recorded table.
V1_BASELINE = {
    "n_games": 1359,
    "margin_mae": 10.274,
    "total_mae": 10.684,
    "ats_hit_rate": 0.4977,
    "ou_hit_rate": 0.5022,
}

# Gates 1-11 exactly as the design spec numbers them, mapped onto the failure labels
# `research_gate_decision` reports. Gate 11 is the shadow rebuild, which this task cannot run.
GATE_SPECS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "margin MAE below Ridge-v1 10.274", ("margin MAE",)),
    (2, "total MAE below Ridge-v1 10.684", ("total MAE",)),
    (3, "margin MAE improves in >= 3 of 5 outer seasons", ("margin season improvement",)),
    (4, "total MAE improves in >= 3 of 5 outer seasons", ("total season improvement",)),
    (
        5,
        "paired improvement bootstrap lower90 > 0 for both targets",
        ("margin paired improvement", "total paired improvement"),
    ),
    (
        6,
        "positive v2 market coefficient and lower90 for both targets",
        ("margin market contribution", "total market contribution"),
    ),
    (7, "ATS hit rate within one point of Ridge-v1 0.497738", ("ATS hit rate",)),
    (8, "O/U hit rate within one point of Ridge-v1 0.502226", ("O/U hit rate",)),
    (9, "cover and over Brier no worse than Ridge-v1", ("cover Brier", "over Brier")),
    (
        10,
        "correctness, availability, determinism, source reliability",
        ("correctness", "availability", "determinism", "source reliability"),
    ),
)
SHADOW_GATE = (11, "shadow production rebuild leaves Ridge-v1 unchanged")

# Structural evidence the gates are computed from. These are not numbered gates, but a
# missing or inconsistent value here makes every metric gate unreadable, so they are printed.
EVIDENCE_LABELS = ("outer season evidence", "game count evidence")


@dataclass(frozen=True)
class ExperimentArtifacts:
    """The four research payloads, built in memory before anything is written."""

    predictions: pd.DataFrame
    evaluation: dict[str, object]
    ablation: pd.DataFrame
    calibration: dict[str, object]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="run and report without writing")
    mode.add_argument("--write", action="store_true", help="atomically write all four artifacts")
    parser.add_argument("--features", type=Path, default=V2_FEATURES_PATH)
    parser.add_argument("--manifest", type=Path, default=V2_MANIFEST_PATH)
    parser.add_argument("--v1-features", type=Path, default=V1_FEATURES_PATH)
    parser.add_argument("--predictions", type=Path, default=V2_OUTER_PREDICTIONS_PATH)
    parser.add_argument("--evaluation", type=Path, default=V2_EVALUATION_PATH)
    parser.add_argument("--ablation", type=Path, default=V2_ABLATION_PATH)
    parser.add_argument("--calibration", type=Path, default=V2_CALIBRATION_PATH)
    parser.add_argument(
        "--require-research-gates",
        action="store_true",
        help="exit nonzero when gates 1-10 do not all pass",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="print the failing stack instead of a one-line error",
    )
    return parser


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    """The four research destinations, in publication order."""
    return {
        "predictions": Path(args.predictions),
        "evaluation": Path(args.evaluation),
        "ablation": Path(args.ablation),
        "calibration": Path(args.calibration),
    }


def _candidate_mapping_field(target: str) -> str:
    return "margin_by_candidate" if target == "margin" else "total_by_candidate"


def remove_one_block_ablations(
    *,
    features: pd.DataFrame,
    manifest: FeatureManifest,
    selections: Sequence[object],
    predictions: pd.DataFrame,
    fitter: Callable[..., object] | None = None,
) -> pd.DataFrame:
    """Refit each selected configuration with one cumulative block removed.

    The ladder is cumulative, so the block introduced at candidate ``Cj`` is exactly
    ``columns(Cj) - columns(Cj-1)``. Removing the C1 rating block is not constructible: the
    rating-variant contract requires a non-C0 schema to declare its canonical rating columns,
    and relaxing that would change the manifest this locked experiment is measuring. Those
    rows are emitted with ``status="not_constructible"`` and the exact error, so the
    accounting stays visible instead of silently skipping a block.
    """
    from nfl_game.experiments.v2_selection import _default_target_fitter

    fit = fitter or _default_target_fitter
    rows: list[dict[str, object]] = []
    for selection in selections:
        season = int(selection.season)
        prior = features.loc[features["season"] < season]
        test = features.loc[features["season"] == season]
        if prior.empty or test.empty:
            continue
        for target, chosen in (("margin", selection.margin), ("total_points", selection.total)):
            config = chosen.config
            candidate = str(config.candidate)
            mapping = dict(getattr(manifest, _candidate_mapping_field(target)))
            actual = test[target].to_numpy(dtype=float)
            full = np.asarray(
                fit(prior.copy(), target, config, manifest, validation_season=season).predict(
                    test.copy()
                ),
                dtype=float,
            )
            mae_full = float(np.abs(full - actual).mean())
            index = CANDIDATES.index(candidate)
            for block in CANDIDATES[1 : index + 1]:
                previous = CANDIDATES[CANDIDATES.index(block) - 1]
                if block not in mapping or previous not in mapping:
                    continue
                removed = tuple(
                    column for column in mapping[block] if column not in set(mapping[previous])
                )
                kept = tuple(column for column in mapping[candidate] if column not in set(removed))
                row: dict[str, object] = {
                    "season": season,
                    "target": target,
                    "candidate": candidate,
                    "removed_block": block,
                    "n_removed_columns": len(removed),
                    "n_kept_columns": len(kept),
                    "mae_full": mae_full,
                    "mae_ablated": float("nan"),
                    "block_contribution": float("nan"),
                    "status": "measured",
                    "detail": "",
                }
                if not removed or not kept:
                    row["status"] = "not_constructible"
                    row["detail"] = "empty block or empty remaining schema"
                    rows.append(row)
                    continue
                ablated_manifest = replace(
                    manifest, **{_candidate_mapping_field(target): {**mapping, candidate: kept}}
                )
                try:
                    predicted = np.asarray(
                        fit(
                            prior.copy(),
                            target,
                            config,
                            ablated_manifest,
                            validation_season=season,
                        ).predict(test.copy()),
                        dtype=float,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded as accounting, not swallowed
                    row["status"] = "not_constructible"
                    row["detail"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    continue
                mae_ablated = float(np.abs(predicted - actual).mean())
                row["mae_ablated"] = mae_ablated
                # Positive means the block helps: removing it made the error worse.
                row["block_contribution"] = mae_ablated - mae_full
                rows.append(row)
    del predictions
    return pd.DataFrame(
        rows,
        columns=[
            "season",
            "target",
            "candidate",
            "removed_block",
            "n_removed_columns",
            "n_kept_columns",
            "mae_full",
            "mae_ablated",
            "block_contribution",
            "status",
            "detail",
        ],
    )


def measure_quality_checks(
    *,
    v1_predictions: pd.DataFrame,
    v2_predictions: pd.DataFrame,
    v2_probabilities: pd.DataFrame,
    recompute_probabilities: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    features: pd.DataFrame | None = None,
    manifest_payload: Mapping[str, object] | None = None,
    determinism_probe: Callable[[], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    evidence: dict[str, object] | None = None,
) -> dict[str, bool]:
    """Measure gate 10's four checks from this run's own objects.

    Every check is computed here rather than asserted: a hardcoded ``True`` would make gate
    10 unfailable, which is the one failure mode a promotion gate cannot afford.
    """
    record: dict[str, object] = {} if evidence is None else evidence

    v1_report = v1_predictions.loc[v1_predictions["season"].isin(REPORT_SEASONS)]
    v2_report = v2_predictions.loc[v2_predictions["season"].isin(REPORT_SEASONS)]

    # Correctness: Ridge-v1 reproduces its frozen acceptance baseline inside this run.
    try:
        measured = evaluate_v1(v1_report)
        correctness = (
            int(measured["n_games"]) == V1_BASELINE["n_games"]
            and round(float(measured["margin_mae"]), 3) == V1_BASELINE["margin_mae"]
            and round(float(measured["total_mae"]), 3) == V1_BASELINE["total_mae"]
            and round(float(measured["ats_hit_rate"]), 4) == V1_BASELINE["ats_hit_rate"]
            and round(float(measured["ou_hit_rate"]), 4) == V1_BASELINE["ou_hit_rate"]
        )
        record["correctness_measured"] = {
            key: (int(measured[key]) if key == "n_games" else round(float(measured[key]), 4))
            for key in V1_BASELINE
        }
    except Exception as exc:  # noqa: BLE001 - a failed measurement is a failed check
        correctness = False
        record["correctness_measured"] = f"{type(exc).__name__}: {exc}"
    record["correctness_expected"] = dict(V1_BASELINE)

    # Availability: v2 predicts exactly the Ridge-v1 report corpus, with finite outputs.
    v1_ids = set(v1_report["game_id"])
    v2_ids = set(v2_report["game_id"])
    finite = bool(
        np.isfinite(v2_report[["model_margin", "model_total"]].to_numpy(dtype=float)).all()
    )
    probability_ids = set(v2_probabilities["game_id"])
    availability = bool(v1_ids == v2_ids and finite and v1_ids <= probability_ids)
    record["availability_detail"] = {
        "v1_report_games": len(v1_ids),
        "v2_report_games": len(v2_ids),
        "missing_from_v2": sorted(v1_ids - v2_ids)[:10],
        "extra_in_v2": sorted(v2_ids - v1_ids)[:10],
        "calibrated_games": len(probability_ids),
        "finite_predictions": finite,
    }

    # Determinism: recomputing the calibrated probabilities, and re-running one outer season
    # of nested selection when a probe is supplied, must reproduce the run bit for bit.
    determinism_parts: list[bool] = []
    if recompute_probabilities is not None:
        try:
            again = recompute_probabilities(v2_predictions)
            pd.testing.assert_frame_equal(
                again.reset_index(drop=True), v2_probabilities.reset_index(drop=True)
            )
            determinism_parts.append(True)
            record["determinism_probabilities"] = "identical"
        except Exception as exc:  # noqa: BLE001
            determinism_parts.append(False)
            record["determinism_probabilities"] = f"{type(exc).__name__}: {exc}"
    if determinism_probe is not None:
        try:
            first, second = determinism_probe()
            pd.testing.assert_frame_equal(
                first.reset_index(drop=True), second.reset_index(drop=True)
            )
            determinism_parts.append(True)
            record["determinism_refit"] = "identical"
        except Exception as exc:  # noqa: BLE001
            determinism_parts.append(False)
            record["determinism_refit"] = f"{type(exc).__name__}: {exc}"
    determinism = bool(determinism_parts) and all(determinism_parts)
    if not determinism_parts:
        record["determinism_probabilities"] = "not measured"

    # Source reliability: the feature frame still hashes to the digests its manifest recorded.
    reliability = False
    if features is not None and manifest_payload is not None:
        output = manifest_payload.get("output")
        snapshots = manifest_payload.get("source_snapshots")
        stored_schema = output.get("schema_sha256") if isinstance(output, Mapping) else None
        stored_features = (
            output.get("features_semantic_sha256") if isinstance(output, Mapping) else None
        )
        actual_schema = schema_fingerprint(features)
        actual_features = semantic_frame_digest(features)
        populated = bool(snapshots) and all(
            isinstance(snapshot, Mapping) and int(snapshot.get("rows") or 0) > 0
            for snapshot in snapshots
        )
        reliability = bool(
            stored_schema == actual_schema and stored_features == actual_features and populated
        )
        record["source_reliability_detail"] = {
            "schema_sha256_matches": stored_schema == actual_schema,
            "features_semantic_sha256_matches": stored_features == actual_features,
            "populated_source_snapshots": populated,
            "n_source_snapshots": len(snapshots) if isinstance(snapshots, Sequence) else 0,
        }
    else:
        record["source_reliability_detail"] = "features or manifest payload unavailable"

    return {
        "correctness": bool(correctness),
        "availability": bool(availability),
        "determinism": bool(determinism),
        "source_reliability": bool(reliability),
    }


def _gate_status(labels: Sequence[str], decision) -> str:
    if any(label in decision.failures for label in labels):
        return "FAIL"
    if any(label in decision.pending for label in labels):
        return "PENDING"
    return "PASS"


def gate_table(decision, *, shadow_status: str = "PENDING") -> dict[str, dict[str, object]]:
    """Gates 1-11 with the status and the labels each one was decided from."""
    table: dict[str, dict[str, object]] = {}
    for number, description, labels in GATE_SPECS:
        table[str(number)] = {
            "description": description,
            "labels": list(labels),
            "status": _gate_status(labels, decision),
        }
    table[str(SHADOW_GATE[0])] = {
        "description": SHADOW_GATE[1],
        "labels": ["shadow production rebuild"],
        "status": shadow_status,
    }
    return table


def print_gate_report(
    report: Mapping[str, object],
    decision,
    *,
    selections: Sequence[object] = (),
    ablation: pd.DataFrame | None = None,
    shadow_status: str = "PENDING",
) -> None:
    """Print the complete research report: metrics, selections, ablations, gates."""
    print("\n=== Ridge-v2 locked experiment ===")
    print(f"report seasons:   {report.get('report_seasons')}")
    print(f"games:            {report.get('n_games')}")
    for key, label in (
        ("margin_mae", "margin MAE"),
        ("total_mae", "total MAE"),
        ("ats_hit_rate", "ATS hit rate"),
        ("ou_hit_rate", "O/U hit rate"),
        ("cover_brier", "cover Brier"),
        ("over_brier", "over Brier"),
    ):
        value = report.get(key)
        baseline_key = {
            "margin_mae": "v1_margin_mae",
            "total_mae": "v1_total_mae",
            "ats_hit_rate": "v1_ats_hit_rate",
            "ou_hit_rate": "v1_ou_hit_rate",
            "cover_brier": "v1_cover_brier",
            "over_brier": "v1_over_brier",
        }[key]
        baseline = report.get(baseline_key)
        rendered = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        against = f"   v1: {baseline:.4f}" if isinstance(baseline, (int, float)) else ""
        print(f"{label:<17} {rendered}{against}")

    per_season = report.get("per_season")
    if isinstance(per_season, Mapping):
        print("\n--- per outer season (positive improvement = v2 better) ---")
        print(f"{'season':<8}{'games':>7}{'margin':>12}{'total':>12}")
        for season in sorted(per_season):
            evidence = per_season[season]
            if not isinstance(evidence, Mapping):
                continue
            print(
                f"{season:<8}{evidence.get('n_games', ''):>7}"
                f"{float(evidence.get('margin_improvement', float('nan'))):>12.4f}"
                f"{float(evidence.get('total_improvement', float('nan'))):>12.4f}"
            )

    if selections:
        print("\n--- selected configuration per outer season ---")
        for selection in selections:
            for target, chosen in (
                ("margin", selection.margin),
                ("total", selection.total),
            ):
                config = chosen.config
                inner = chosen.mean_inner_mae
                inner_text = f"{inner:.4f}" if isinstance(inner, (int, float)) else "n/a"
                print(
                    f"  {selection.season}  {target:<7} {config.candidate}"
                    f"  alpha={config.alpha:<6} halflives={config.short_halflife}/"
                    f"{config.long_halflife}  prior_weight={config.prior_season_weight}"
                    f"  inner MAE {inner_text}"
                )

    if ablation is not None and not ablation.empty:
        print("\n--- remove-one-block ablations (positive = the block helps) ---")
        measured = ablation.loc[ablation["status"] == "measured"]
        if not measured.empty:
            summary = (
                measured.groupby(["target", "removed_block"])["block_contribution"]
                .agg(["mean", "count"])
                .reset_index()
            )
            for _, row in summary.iterrows():
                print(
                    f"  {row['target']:<13} remove {row['removed_block']}: "
                    f"mean {row['mean']:+.4f} MAE over {int(row['count'])} season(s)"
                )
        skipped = ablation.loc[ablation["status"] != "measured"]
        for _, row in skipped.iterrows():
            print(
                f"  {row['target']:<13} remove {row['removed_block']} ({row['season']}): "
                f"{row['status']} - {str(row['detail'])[:90]}"
            )

    print("\n--- structural evidence ---")
    for label in EVIDENCE_LABELS:
        if label in decision.failures:
            status = "FAIL"
        elif label in decision.pending:
            status = "PENDING"
        else:
            status = "PASS"
        print(f"  {status:<8} {label}")

    print("\n--- promotion gates ---")
    for number, description, labels in GATE_SPECS:
        status = _gate_status(labels, decision)
        print(f"gate {number:>2}  {status:<8} {description}")
    print(f"gate {SHADOW_GATE[0]:>2}  {shadow_status:<8} {SHADOW_GATE[1]}")

    if decision.failures:
        print(f"\nRESEARCH GATES FAILED: {', '.join(decision.failures)}")
        print("Ridge v1 remains official.")
    elif decision.pending:
        print(f"\nRESEARCH GATES INCOMPLETE: {', '.join(decision.pending)}")
        print("Ridge v1 remains official.")
    else:
        print("\nResearch gates 1-10 all PASS. Gate 11 (shadow rebuild) is PENDING approval.")


def build_experiment_artifacts(
    *,
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    report: Mapping[str, object],
    decision,
    selections: Sequence[object],
    ablation: pd.DataFrame,
    calibration_fits: Sequence[Mapping[str, object]],
    quality_evidence: Mapping[str, object],
    manifest_payload: Mapping[str, object] | None,
) -> ExperimentArtifacts:
    """Assemble the four research payloads without writing anything."""
    report_rows = predictions.loc[predictions["season"].isin(REPORT_SEASONS)].copy()
    merged = report_rows.merge(probabilities, on="game_id", how="left", validate="one_to_one")

    selection_records = [
        {
            "season": int(selection.season),
            "target": target,
            "candidate": str(chosen.config.candidate),
            "alpha": float(chosen.config.alpha),
            "short_halflife": int(chosen.config.short_halflife),
            "long_halflife": int(chosen.config.long_halflife),
            "prior_season_weight": float(chosen.config.prior_season_weight),
            "mean_inner_mae": (
                None if chosen.mean_inner_mae is None else float(chosen.mean_inner_mae)
            ),
            "validation_seasons": [int(value) for value in chosen.validation_seasons],
            "validation_games": int(chosen.validation_games),
        }
        for selection in selections
        for target, chosen in (("margin", selection.margin), ("total_points", selection.total))
    ]

    output = manifest_payload.get("output") if isinstance(manifest_payload, Mapping) else None
    evaluation = {
        "model_version": "ridge-v2",
        "experiment": "locked ridge-v2 research experiment",
        "feature_schema_version": (
            manifest_payload.get("feature_schema_version")
            if isinstance(manifest_payload, Mapping)
            else None
        ),
        "features_semantic_sha256": (
            output.get("features_semantic_sha256") if isinstance(output, Mapping) else None
        ),
        "source_manifest_sha256": (
            manifest_payload.get("source_manifest_sha256")
            if isinstance(manifest_payload, Mapping)
            else None
        ),
        "report_seasons": list(REPORT_SEASONS),
        "calibration_seasons": list(CALIBRATION_SEASONS),
        "report": dict(report),
        "gates": gate_table(decision),
        "research_gates_passed": bool(decision.approved),
        "failures": list(decision.failures),
        "pending": list(decision.pending),
        "quality_evidence": dict(quality_evidence),
        "selections": selection_records,
        "ablation_rows": len(ablation),
    }

    calibration = {
        "model_version": "ridge-v2",
        "calibration_seasons": list(CALIBRATION_SEASONS),
        "report_seasons": list(REPORT_SEASONS),
        "method": "leak-free per-season logistic calibration on model-minus-line edge",
        "fits": [dict(fit) for fit in calibration_fits],
        "n_calibrated_games": len(probabilities),
    }
    return ExperimentArtifacts(
        predictions=merged,
        evaluation=evaluation,
        ablation=ablation,
        calibration=calibration,
    )


def _write_json(payload: Mapping[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def write_experiment_artifacts_atomic(
    artifacts: ExperimentArtifacts,
    paths: Mapping[str, Path],
    *,
    write_parquet: Callable[[pd.DataFrame, Path], None] | None = None,
    replace_file: Callable[[Path, Path], None] | None = None,
) -> None:
    """Stage, validate, then publish all four artifacts, restoring every original on failure."""
    write_parquet = write_parquet or (lambda frame, path: frame.to_parquet(path, index=False))
    replace_file = replace_file or (lambda source, destination: source.replace(destination))

    ordered = [
        (label, Path(paths[label]))
        for label in ("predictions", "evaluation", "ablation", "calibration")
    ]
    for label, path in ordered:
        if path.name in PROTECTED_V1_ARTIFACT_NAMES:
            raise ValueError(
                f"refusing to write the Ridge-v2 {label} artifact over "
                f"frozen Ridge-v1 file {path.name!r}"
            )
    if len({path.resolve() for _, path in ordered}) != len(ordered):
        raise ValueError("the four Ridge-v2 research destinations must all differ")

    payloads: dict[str, object] = {
        "predictions": artifacts.predictions,
        "evaluation": artifacts.evaluation,
        "ablation": artifacts.ablation,
        "calibration": artifacts.calibration,
    }
    first_parent = ordered[0][1].parent
    first_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ridge-v2-experiment-", dir=first_parent) as temp:
        temp_dir = Path(temp)
        staged: dict[str, Path] = {}
        for label, path in ordered:
            path.parent.mkdir(parents=True, exist_ok=True)
            staged_path = temp_dir / f"{label}-{path.name}"
            payload = payloads[label]
            if isinstance(payload, pd.DataFrame):
                write_parquet(payload, staged_path)
                reread = pd.read_parquet(staged_path)
                if len(reread) != len(payload):
                    raise ValueError(f"staged {label} artifact lost rows on round trip")
            else:
                _write_json(payload, staged_path)  # type: ignore[arg-type]
                json.loads(staged_path.read_text(encoding="utf-8"))
            staged[label] = staged_path

        backups: dict[str, Path | None] = {}
        for label, path in ordered:
            if path.exists():
                backup = temp_dir / f"backup-{label}-{path.name}"
                shutil.copy2(path, backup)
                backups[label] = backup
            else:
                backups[label] = None

        published: list[tuple[str, Path]] = []
        try:
            for label, path in ordered:
                publication = temp_dir / f"publish-{label}-{path.name}"
                shutil.copy2(staged[label], publication)
                replace_file(publication, path)
                published.append((label, path))
        except Exception:
            for label, path in reversed(published):
                backup = backups[label]
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup, path)
            raise


def _dependency(dependencies: Mapping[str, object], name: str, default):
    value = dependencies.get(name)
    return default if value is None else value


def main(
    argv: Sequence[str] | None = None,
    dependencies: Mapping[str, object] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    deps: Mapping[str, object] = dependencies or {}
    try:
        return _run(args, deps)
    except Exception:
        if args.traceback:
            traceback.print_exc()
            return 2
        raise


def _run(args: argparse.Namespace, deps: Mapping[str, object]) -> int:
    paths = output_paths(args)

    v1_features = _dependency(deps, "v1_features", None)
    if v1_features is None:
        v1_features = pd.read_parquet(Path(args.v1_features))
    v2_features = _dependency(deps, "v2_features", None)
    if v2_features is None:
        v2_features = pd.read_parquet(Path(args.features))

    manifest_payload = _dependency(deps, "manifest_payload", None)
    if manifest_payload is None:
        manifest_payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest = _dependency(deps, "manifest", None)
    if manifest is None:
        manifest = FeatureManifest.from_dict(manifest_payload["feature_manifest"])

    v1_walk_forward = _dependency(deps, "v1_walk_forward", walk_forward)
    nested = _dependency(deps, "nested_walk_forward", nested_walk_forward_v2)
    probabilities_fn = _dependency(deps, "probabilities", walk_forward_probabilities)
    evaluate_fn = _dependency(deps, "evaluate", None)
    ablations_fn = _dependency(deps, "ablations", remove_one_block_ablations)
    quality_fn = _dependency(deps, "quality_checks", measure_quality_checks)

    seasons = list(CALIBRATION_SEASONS)
    print(f"Ridge-v1 walk-forward over {seasons[0]}-{seasons[-1]} ...")
    v1_predictions = v1_walk_forward(v1_features, seasons)
    print(f"  {len(v1_predictions)} rows")

    print(f"Ridge-v2 nested walk-forward over {seasons[0]}-{seasons[-1]} ...")
    result = nested(v2_features, seasons, manifest)
    v2_predictions = result.predictions
    print(f"  {len(v2_predictions)} rows, {len(result.selections)} outer selection(s)")

    calibration_fits: list[dict[str, object]] = []
    v1_probabilities = probabilities_fn(v1_predictions)
    v2_probabilities = probabilities_fn(
        v2_predictions, fit_observer=lambda record: calibration_fits.append(dict(record))
    )

    quality_evidence: dict[str, object] = {}
    quality_checks = quality_fn(
        v1_predictions=v1_predictions,
        v2_predictions=v2_predictions,
        v2_probabilities=v2_probabilities,
        recompute_probabilities=probabilities_fn,
        features=v2_features,
        manifest_payload=manifest_payload,
        determinism_probe=_determinism_probe(nested, v2_features, manifest, result),
        evidence=quality_evidence,
    )
    print("quality checks: " + ", ".join(f"{k}={v}" for k, v in quality_checks.items()))

    if evaluate_fn is None:
        report = evaluate_v2(
            v1_predictions,
            v2_predictions,
            v1_probabilities,
            v2_probabilities,
            quality_checks=quality_checks,
        )
    else:
        report = evaluate_fn(
            v1_predictions=v1_predictions,
            v2_predictions=v2_predictions,
            v1_probabilities=v1_probabilities,
            v2_probabilities=v2_probabilities,
            quality_checks=quality_checks,
        )

    ablation = ablations_fn(
        features=v2_features,
        manifest=manifest,
        selections=result.selections,
        predictions=v2_predictions,
    )
    decision = research_gate_decision(report)
    print_gate_report(
        report,
        decision,
        selections=result.selections,
        ablation=ablation,
        shadow_status="PENDING",
    )

    artifacts = build_experiment_artifacts(
        predictions=v2_predictions,
        probabilities=v2_probabilities,
        report=report,
        decision=decision,
        selections=result.selections,
        ablation=ablation,
        calibration_fits=calibration_fits,
        quality_evidence=quality_evidence,
        manifest_payload=manifest_payload,
    )

    if args.write:
        write_experiment_artifacts_atomic(
            artifacts,
            paths,
            write_parquet=_dependency(deps, "write_parquet", None),
            replace_file=_dependency(deps, "replace_file", None),
        )
        print("\nwrite complete:")
        for label, path in paths.items():
            print(f"  {label}: {path}")
    else:
        print("\ndry-run: no artifacts written")

    return int(bool(args.require_research_gates) and not decision.approved)


def _determinism_probe(nested, features, manifest, result):
    """Re-run the last outer season's nested selection and compare it with this run."""
    if not getattr(result, "selections", ()):  # nothing selected: nothing to reproduce
        return None
    last = max(int(selection.season) for selection in result.selections)

    def probe() -> tuple[pd.DataFrame, pd.DataFrame]:
        again = nested(features, [last], manifest)
        columns = ["game_id", "model_margin", "model_total"]
        first = (
            result.predictions.loc[result.predictions["season"] == last, columns]
            .sort_values("game_id")
            .reset_index(drop=True)
        )
        second = again.predictions[columns].sort_values("game_id").reset_index(drop=True)
        return first, second

    return probe


if __name__ == "__main__":
    sys.exit(main())
