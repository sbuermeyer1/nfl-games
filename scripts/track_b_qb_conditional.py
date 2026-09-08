"""Track B: does the QB block help on the games where it fires?

Runs the pre-registered test in
`docs/superpowers/specs/2026-09-08-track-b-qb-conditional-prereg.md`, which was committed
before this file existed. The rule below is transcribed from it and must not be edited to
match an outcome.

Why this reimplements walk-forward instead of calling `backtest.walk_forward`: `GameModel`
hardcodes `FEATURE_COLS` (`model/predict.py:164`), so it cannot fit an alternative schema.
The estimator and preprocessing are taken from production rather than rebuilt --
`ESTIMATORS["ridge"]` at `model/predict.py:108` -- so the only thing that differs between
the arms is the column set.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import DEFAULT_ALPHA, ESTIMATORS
from nfl_game.paths import PROCESSED_DIR

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
SUBSET_FLAG = "qb_new_starter_any"
TARGETS = ("margin", "total_points")

C2_BLOCK = [
    "qb_epa_diff",
    "qb_cpoe_diff",
    "qb_sack_rate_diff",
    "qb_int_rate_diff",
    "qb_change_epa_diff",
    "qb_new_starter_any",
    "qb_rookie_any",
    "qb_uncertain_any",
]

ARMS = {
    "A0": list(FEATURE_COLS),
    "A1": [*FEATURE_COLS, "qb_change_epa_diff"],
    "A2": [*FEATURE_COLS, *C2_BLOCK],
}

# Pre-registered guard: a feature that helps where it fires must not pay for it everywhere.
POOLED_TOLERANCE = 0.010
# One-sided 90% lower bound on a paired mean over 5 seasons: t(0.90, df=4).
T90_DF4 = 1.533


def _lower90(values: np.ndarray) -> float:
    """One-sided 90% lower bound on the mean of a small paired sample."""
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    return float(np.mean(values) - T90_DF4 * se)


def _common_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Every arm is scored on identical games.

    A row is kept only if every column ANY arm uses is finite and both targets are
    present. Letting one arm quietly see rows another could not is the unequal-set
    defect that produced a false Ridge-v2 conclusion.
    """
    used = sorted({column for columns in ARMS.values() for column in columns})
    missing = [column for column in [*used, SUBSET_FLAG, *TARGETS] if column not in frame]
    if missing:
        raise SystemExit(f"feature artifact is missing required columns: {missing}")
    numeric = frame[used].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    for target in TARGETS:
        keep &= frame[target].notna().to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def _walk_forward(frame: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    """Predict each test season from a model trained only on strictly earlier seasons."""
    out = []
    for season in TEST_SEASONS:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        if train.empty or test.empty:
            continue
        model = ESTIMATORS["ridge"](DEFAULT_ALPHA)
        model.fit(train[columns].to_numpy(dtype=float), train[target].to_numpy(dtype=float))
        predicted = model.predict(test[columns].to_numpy(dtype=float))
        out.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "predicted": predicted,
                    "actual": test[target].to_numpy(dtype=float),
                    SUBSET_FLAG: test[SUBSET_FLAG].to_numpy(),
                }
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _mae(frame: pd.DataFrame) -> float:
    return float(np.mean(np.abs(frame["predicted"] - frame["actual"])))


def _per_season_mae(frame: pd.DataFrame) -> dict[int, float]:
    return {int(s): _mae(g) for s, g in frame.groupby("season")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        default=str(PROCESSED_DIR / "game_features_ridge_v2.parquet"),
        help="the Ridge-v2 union feature artifact (carries both C0 and C2 columns)",
    )
    args = parser.parse_args(argv)

    frame = _common_rows(pd.read_parquet(args.features))
    report: dict = {"rows_scored": len(frame), "arms": {}, "verdict": {}}

    predictions = {}
    for target in TARGETS:
        for arm, columns in ARMS.items():
            predictions[(arm, target)] = _walk_forward(frame, columns, target)

    # Common fold set: a season any arm could not score is dropped for every arm.
    scored = {
        season
        for season in TEST_SEASONS
        if all(
            not predictions[(arm, target)].empty
            and season in set(predictions[(arm, target)]["season"])
            for arm in ARMS
            for target in TARGETS
        )
    }
    report["seasons_scored"] = sorted(scored)
    report["seasons_dropped"] = sorted(set(TEST_SEASONS) - scored)

    for target in TARGETS:
        for arm in ARMS:
            p = predictions[(arm, target)]
            p = p[p["season"].isin(scored)]
            subset = p[p[SUBSET_FLAG].eq(1)]
            report["arms"].setdefault(target, {})[arm] = {
                "pooled_mae": round(_mae(p), 6),
                "pooled_n": len(p),
                "change_mae": round(_mae(subset), 6),
                "change_n": len(subset),
                "change_mae_by_season": {
                    s: round(v, 6) for s, v in _per_season_mae(subset).items()
                },
            }

    # The pre-registered rule, applied mechanically.
    for arm in ("A1", "A2"):
        base = report["arms"]["margin"]["A0"]
        cand = report["arms"]["margin"][arm]
        paired = np.array(
            [
                base["change_mae_by_season"][s] - cand["change_mae_by_season"][s]
                for s in sorted(scored)
            ]
        )
        pooled_delta = base["pooled_mae"] - cand["pooled_mae"]
        mean_improvement = float(np.mean(paired))
        lower = _lower90(paired)
        report["verdict"][arm] = {
            "change_paired_improvement_mean": round(mean_improvement, 6),
            "change_paired_improvement_lower90": round(lower, 6),
            "change_improved_seasons": f"{int((paired > 0).sum())} of {len(paired)}",
            "pooled_margin_delta": round(pooled_delta, 6),
            "cond1_mean_positive": bool(mean_improvement > 0),
            "cond2_lower90_positive": bool(lower > 0),
            "cond3_pooled_not_worse": bool(pooled_delta > -POOLED_TOLERANCE),
        }
        v = report["verdict"][arm]
        v["ADOPT"] = bool(
            v["cond1_mean_positive"] and v["cond2_lower90_positive"] and v["cond3_pooled_not_worse"]
        )

    # Secondary, reported not gating: the stated redundancy mechanism for A2's level features.
    report["redundancy"] = {
        "corr_qb_epa_diff_vs_net_rating_diff": round(
            float(frame["qb_epa_diff"].corr(frame["net_rating_diff"])), 6
        )
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
