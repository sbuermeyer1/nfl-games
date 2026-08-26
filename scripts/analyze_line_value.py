"""Does the model have information the market had not yet priced?

Two questions, against the early line reconstructed from nflverse git history:

1. **Closing-line value.** When the model disagreed with the early number, did the line
   subsequently move toward the model? This is a *continuous* measurement in points, so it has
   far more statistical power than a win/loss rate: with ~1,300 games it can resolve effects an
   ATS record would need decades to establish.
2. **ATS settled at the number you could actually have bet** -- the early line -- rather than at
   the closing number, which is the one you cannot bet and which already contains everything the
   market knows.

Sign convention, verified against the corpus: `spread_line` is from the home team's perspective
in the same units as `margin` (home minus away), so a positive number means the home team is
favoured by that many points.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.paths import PROCESSED_DIR, RAW_DIR

DEFAULT_HISTORY = RAW_DIR / "line_history" / "line_history_combined.parquet"
REPORT_SEASONS = (2021, 2022, 2023, 2024, 2025)
THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "game_features.parquet")
    return parser


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (centre - margin, centre + margin)


def ats_record(
    frame: pd.DataFrame, *, line_column: str, edge_column: str, threshold: float
) -> dict[str, object]:
    """Bet the side the model prefers, settled at `line_column` -- the number actually bet."""
    picks = frame.loc[frame[edge_column].abs() >= threshold].copy()
    # A game landing exactly on the number you bet is a push, not a loss.
    picks = picks.loc[picks["margin"] != picks[line_column]]
    home_pick = picks[edge_column] > 0
    won = np.where(
        home_pick, picks["margin"] > picks[line_column], picks["margin"] < picks[line_column]
    )
    n = len(picks)
    wins = int(won.sum())
    low, high = _wilson(wins, n)
    return {
        "threshold": threshold,
        "n": n,
        "wins": wins,
        "hit_rate": wins / n if n else float("nan"),
        "ci_low": low,
        "ci_high": high,
    }


def closing_line_value(frame: pd.DataFrame, *, threshold: float) -> dict[str, object]:
    """Signed line movement in the direction the model preferred, in points."""
    picks = frame.loc[frame["edge_early"].abs() >= threshold]
    if picks.empty:
        return {"threshold": threshold, "n": 0}
    # Bet home and the line rising is value gained; bet away and it falling is.
    signed = np.sign(picks["edge_early"]) * picks["line_move"]
    signed = signed.to_numpy(dtype=float)
    moved = signed[signed != 0]
    n = len(signed)
    mean = float(signed.mean())
    se = float(signed.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    beat = int((signed > 0).sum())
    low, high = _wilson(beat, len(moved))
    return {
        "threshold": threshold,
        "n": n,
        "mean_clv_points": mean,
        "se": se,
        "z": mean / se if se and se > 0 else float("nan"),
        "beat_close_rate": beat / len(moved) if len(moved) else float("nan"),
        "beat_ci_low": low,
        "beat_ci_high": high,
        "n_moved": len(moved),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    history = pd.read_parquet(args.history)
    features = pd.read_parquet(args.features)

    print(f"line history: {len(history)} games, "
          f"{int(history['early_spread_line'].notna().sum())} priced")
    predictions = walk_forward(features, list(REPORT_SEASONS))
    print(f"model predictions: {len(predictions)} games")

    merged = predictions.merge(
        history[["game_id", "early_spread_line", "early_total_line", "snapshot_at"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["early_spread_line"].notna()].copy()
    print(f"joined and priced: {len(merged)} games\n")

    merged["edge_early"] = merged["model_margin"] - merged["early_spread_line"]
    merged["edge_close"] = merged["model_margin"] - merged["spread_line"]
    merged["line_move"] = merged["spread_line"] - merged["early_spread_line"]

    moved = merged.loc[merged["line_move"] != 0]
    print("--- how much does the line actually move from the early number? ---")
    print(f"games where the spread moved at all: {len(moved)}/{len(merged)} "
          f"({len(moved) / len(merged):.1%})")
    print(f"mean absolute movement: {merged['line_move'].abs().mean():.2f} points, "
          f"median {merged['line_move'].abs().median():.2f}, "
          f"max {merged['line_move'].abs().max():.1f}")

    print("\n--- 1. CLOSING-LINE VALUE (positive = the line moved toward the model) ---")
    print(f"{'edge':>6}{'n':>7}{'mean CLV':>11}{'SE':>8}{'z':>8}{'beat close':>12}{'95% CI':>18}")
    for threshold in THRESHOLDS:
        row = closing_line_value(merged, threshold=threshold)
        if not row.get("n"):
            continue
        print(
            f"{threshold:>6.0f}{row['n']:>7}{row['mean_clv_points']:>+11.4f}{row['se']:>8.4f}"
            f"{row['z']:>8.2f}{row['beat_close_rate']:>12.4f}"
            f"   [{row['beat_ci_low']:.3f}, {row['beat_ci_high']:.3f}]"
        )

    print("\n--- 2. ATS settled at the EARLY line (the number you could bet) ---")
    print(f"{'edge':>6}{'n':>7}{'hit':>9}{'95% CI':>20}   vs closing-line ATS")
    for threshold in THRESHOLDS:
        early = ats_record(
            merged, line_column="early_spread_line", edge_column="edge_early", threshold=threshold
        )
        close = ats_record(
            merged, line_column="spread_line", edge_column="edge_close", threshold=threshold
        )
        if not early["n"]:
            continue
        print(
            f"{threshold:>6.0f}{early['n']:>7}{early['hit_rate']:>9.4f}"
            f"   [{early['ci_low']:.3f}, {early['ci_high']:.3f}]"
            f"      closing {close['hit_rate']:.4f} (n={close['n']})"
        )

    print("\nBreak-even at standard -110 juice is 0.5238.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
