"""E1-research: does FTN charting add anything to the best core Ridge-v2 candidate?

Research only. This script never writes a Task 13-17 artifact, never touches Ridge v1, the
tracker or the web package, and cannot promote anything. It exists to measure one question
and record the answer.

**Design.** Task 14's nested selection chose C0 -- the exact Ridge-v1 schema -- in every
evaluation season, so C0 *is* the best core candidate and the paired arms are:

    core : the C0 schema
    E1   : the C0 schema plus the FTN charting features

Both arms train on the **identical rows** and predict the identical outer season, so the only
difference between them is the FTN block. That is the control this comparison needs: an arm
measured against a differently-trained baseline would confound the block with its training set.

**Why the fit does not route through the manifest.** E1 is not a manifest candidate, and the
rating-variant contract requires any non-C0 schema to declare canonical rating columns -- the
same wall the Task 14 C1 ablation hit. `fit_research_ridge` therefore mirrors
`fit_target_ridge` directly, reusing the very same `RobustStandardScaler` + `Ridge` pipeline
so the preprocessing cannot silently diverge from production.

**Eligibility.** FTN charting begins in 2022, so an outer season needs at least one complete
prior FTN season: 2023, 2024 and 2025 qualify and no earlier season can.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from nfl_game.data.nfl import load_ftn_charting, load_pbp
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.v2 import RobustStandardScaler
from nfl_game.paths import (
    PROCESSED_DIR,
    V2_ABLATION_PATH,
    V2_CALIBRATION_PATH,
    V2_EVALUATION_PATH,
    V2_FEATURES_PATH,
    V2_MANIFEST_PATH,
    V2_OUTER_PREDICTIONS_PATH,
    V2_TRACKER_LEDGER_PATH,
)
from nfl_game.pipeline.build_v2 import PROTECTED_V1_ARTIFACT_NAMES
from nfl_game.ratings.ftn import ftn_features_for_targets, ftn_game_features, team_game_ftn

RESEARCH_LABEL = "E1-research"
FTN_FIRST_SEASON = 2022
DEFAULT_REPORT_PATH = PROCESSED_DIR / "ridge_v2_e1_ftn_research.json"

# No output of this script may collide with an artifact produced by Tasks 13-17, or with a
# frozen Ridge-v1 file. Research must not be able to overwrite the core decision's evidence.
FORBIDDEN_OUTPUT_NAMES = frozenset(
    {
        *PROTECTED_V1_ARTIFACT_NAMES,
        V2_FEATURES_PATH.name,
        V2_MANIFEST_PATH.name,
        V2_OUTER_PREDICTIONS_PATH.name,
        V2_EVALUATION_PATH.name,
        V2_ABLATION_PATH.name,
        V2_CALIBRATION_PATH.name,
        V2_TRACKER_LEDGER_PATH.name,
    }
)

_TARGETS = ("margin", "total_points")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1-research: FTN charting on top of C0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report without writing (default)")
    mode.add_argument("--write", action="store_true", help="write the research report")
    parser.add_argument("--features", type=Path, default=V2_FEATURES_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--first-outer-season",
        type=int,
        default=FTN_FIRST_SEASON + 1,
        help="earliest outer season; must leave at least one prior FTN season",
    )
    return parser


def eligible_outer_seasons(features: pd.DataFrame, first_outer_season: int) -> list[int]:
    """Outer seasons the 2022+ FTN history can support, and no others."""
    if first_outer_season <= FTN_FIRST_SEASON:
        raise ValueError(
            f"outer season {first_outer_season} has no prior FTN history; "
            f"charting begins in {FTN_FIRST_SEASON}"
        )
    seasons = sorted({int(value) for value in features["season"].unique()})
    return [season for season in seasons if season >= first_outer_season]


def fit_research_ridge(
    train: pd.DataFrame, target: str, columns: Sequence[str], alpha: float = 1.0
):
    """Mirror `fit_target_ridge` for an explicit column list, with the same preprocessing."""
    usable = train.loc[train[target].notna()]
    matrix = usable[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"E1 training matrix for {target!r} contains non-finite values")
    response = usable[target].to_numpy(dtype=float)
    if not np.isfinite(response).all():
        raise ValueError(f"E1 training target {target!r} contains non-finite values")
    pipeline = make_pipeline(RobustStandardScaler(), Ridge(alpha=alpha))
    pipeline.fit(matrix, response)
    return pipeline


def paired_arm_results(
    features: pd.DataFrame,
    ftn_columns: Sequence[str],
    outer_seasons: Sequence[int],
    *,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Core vs E1 on identical training rows and identical outer rows, season by season."""
    core_columns = list(FEATURE_COLS)
    e1_columns = [*core_columns, *ftn_columns]
    rows: list[dict[str, object]] = []
    for season in sorted(int(value) for value in outer_seasons):
        train = features.loc[
            (features["season"] >= FTN_FIRST_SEASON) & (features["season"] < season)
        ]
        test = features.loc[features["season"] == season]
        if train.empty or test.empty:
            continue
        usable = train.dropna(subset=e1_columns)
        test_usable = test.dropna(subset=e1_columns)
        for target in _TARGETS:
            actual = test_usable[target].to_numpy(dtype=float)
            arm_scores: dict[str, float] = {}
            for arm, columns in (("core", core_columns), ("E1", e1_columns)):
                model = fit_research_ridge(usable, target, columns, alpha=alpha)
                predicted = np.asarray(
                    model.predict(test_usable[list(columns)].to_numpy(dtype=float)), dtype=float
                )
                arm_scores[arm] = float(np.abs(predicted - actual).mean())
            rows.append(
                {
                    "label": RESEARCH_LABEL,
                    "outer_season": season,
                    "target": target,
                    "n_train": len(usable),
                    "n_test": len(test_usable),
                    "train_seasons": f"{FTN_FIRST_SEASON}-{season - 1}",
                    "core_mae": arm_scores["core"],
                    "e1_mae": arm_scores["E1"],
                    # Positive means FTN helped: the core arm's error was higher.
                    "ftn_contribution": arm_scores["core"] - arm_scores["E1"],
                }
            )
    return pd.DataFrame(rows)


def build_ftn_game_features(
    features: pd.DataFrame,
    *,
    loaders: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, list[str], int]:
    """FTN game-level features aligned to the feature artifact's game rows."""
    loaders = loaders or {}
    seasons = sorted(
        {int(value) for value in features["season"].unique() if int(value) >= FTN_FIRST_SEASON}
    )
    load_ftn = loaders.get("load_ftn_charting", load_ftn_charting)
    load_plays = loaders.get("load_pbp", load_pbp)
    ftn = load_ftn(seasons)
    pbp = load_plays(seasons)
    team_games = team_game_ftn(ftn, pbp)
    targets = sorted(
        {
            (int(season), int(week))
            for season, week in zip(features["season"], features["week"], strict=True)
            if int(season) >= FTN_FIRST_SEASON
        }
    )
    team_features = ftn_features_for_targets(team_games, targets)
    game_features = ftn_game_features(features, team_features)
    columns = [name for name in game_features.columns if name != "game_id"]
    merged = features.merge(game_features, on="game_id", how="left", validate="one_to_one")
    return merged, columns, len(team_games)


def print_research_report(results: pd.DataFrame, *, coverage: Mapping[str, object]) -> None:
    print(f"\n=== {RESEARCH_LABEL}: FTN charting on top of the C0 core ===")
    print(
        f"charted team-games: {coverage['charted_team_games']}   game rows: "
        f"{coverage['game_rows']}   FTN feature columns: {coverage['n_ftn_columns']}   "
        f"games with an imputed side: {coverage['imputed_share']:.4f}"
    )
    if results.empty:
        print("no eligible outer season: FTN history cannot support this evaluation window")
        return
    print(
        f"\n{'season':<8}{'target':<15}{'n_train':>9}{'n_test':>8}{'core MAE':>11}"
        f"{'E1 MAE':>10}{'FTN':>10}"
    )
    for _, row in results.iterrows():
        print(
            f"{row['outer_season']:<8}{row['target']:<15}{row['n_train']:>9}{row['n_test']:>8}"
            f"{row['core_mae']:>11.4f}{row['e1_mae']:>10.4f}{row['ftn_contribution']:>+10.4f}"
        )
    print("\n--- pooled by target (positive FTN = charting helped) ---")
    for target, group in results.groupby("target"):
        mean = group["ftn_contribution"].mean()
        wins = int((group["ftn_contribution"] > 0).sum())
        print(
            f"  {target:<14} mean {mean:+.4f} MAE over {len(group)} season(s); "
            f"FTN better in {wins} of {len(group)}"
        )
    print(
        "\nThis is research only. It cannot promote anything, and Ridge v1 remains official "
        "regardless of what this table shows."
    )


def main(
    argv: Sequence[str] | None = None, dependencies: Mapping[str, object] | None = None
) -> int:
    args = _parser().parse_args(argv)
    deps = dependencies or {}
    report_path = Path(args.report)
    if report_path.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(
            f"refusing to write E1 research over the protected artifact {report_path.name!r}"
        )

    features = deps.get("features")
    if features is None:
        features = pd.read_parquet(Path(args.features))
    outer_seasons = eligible_outer_seasons(features, args.first_outer_season)

    merged, ftn_columns, team_games = build_ftn_game_features(features, loaders=deps.get("loaders"))
    coverage = {
        "charted_team_games": team_games,
        "game_rows": len(merged),
        "n_ftn_columns": len(ftn_columns),
        "imputed_share": float(
            pd.to_numeric(merged.get("ftn_imputed_any"), errors="coerce")
            .loc[merged["season"] >= FTN_FIRST_SEASON]
            .fillna(1.0)
            .mean()
        ),
    }
    results = paired_arm_results(merged, ftn_columns, outer_seasons)
    print_research_report(results, coverage=coverage)

    if args.write:
        payload = {
            "label": RESEARCH_LABEL,
            "production_eligible": False,
            "ftn_first_season": FTN_FIRST_SEASON,
            "outer_seasons": outer_seasons,
            "ftn_columns": list(ftn_columns),
            "coverage": coverage,
            "results": results.to_dict(orient="records"),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(f"\nresearch report written: {report_path}")
    else:
        print("\ndry-run: no report written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
