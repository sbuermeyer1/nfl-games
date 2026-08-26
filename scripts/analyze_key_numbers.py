"""What is half a point of line shopping actually worth?

The closing-line-value work established that the model beats the *early* number by +0.267
points at edge >= 2 -- real, but only ~55% of the ~0.48 points needed to clear the vig. This
script measures the other lever, which needs no modelling at all: getting a better number by
shopping between books.

It cannot measure book-to-book dispersion -- nflverse carries a single consensus line and a
multi-book feed is paid -- so it does not guess at one. Instead it answers the question that
IS answerable from the corpus we have: **when you get half a point better than the number you
actually bet, how often does the outcome change, and what is that worth?**

The answer is concentrated on the key numbers, because NFL margins are not smooth. Everything
here is counted from real finished games, not modelled.

Sign convention follows `analyze_line_value.py`: `margin` and `spread_line` both live in
home-margin space, a home pick wins when `margin > line`, an away pick when `margin < line`,
and `margin == line` is a push.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.paths import PROCESSED_DIR

REPORT_SEASONS = (2021, 2022, 2023, 2024, 2025)
THRESHOLDS = (0.0, 1.0, 2.0, 3.0)

# Payouts per 1 unit staked at -110.
WIN_RETURN = 100.0 / 110.0
LOSS_RETURN = -1.0
PUSH_RETURN = 0.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "game_features.parquet")
    parser.add_argument(
        "--half-points",
        type=float,
        default=0.5,
        help="how much better a number to test; 0.5 is one half-point of shopping",
    )
    return parser


def _outcome(margin: np.ndarray, line: np.ndarray, home_pick: np.ndarray) -> np.ndarray:
    """WIN / PUSH / LOSS per bet, as a string array."""
    push = margin == line
    won = np.where(home_pick, margin > line, margin < line)
    return np.where(push, "PUSH", np.where(won, "WIN", "LOSS"))


def _returns(outcome: np.ndarray) -> np.ndarray:
    return np.select(
        [outcome == "WIN", outcome == "PUSH"],
        [WIN_RETURN, PUSH_RETURN],
        default=LOSS_RETURN,
    )


def margin_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    """How often each exact final margin occurs -- the reason key numbers matter."""
    counts = frame["margin"].abs().value_counts().sort_index()
    out = pd.DataFrame({"abs_margin": counts.index, "games": counts.to_numpy()})
    out["share"] = out["games"] / len(frame)
    return out


def half_point_value(frame: pd.DataFrame, *, threshold: float, step: float) -> dict[str, object]:
    """Outcome changes from getting `step` points better than the number actually bet."""
    picks = frame.loc[frame["edge_close"].abs() >= threshold].copy()
    if picks.empty:
        return {"threshold": threshold, "n": 0}

    home_pick = (picks["edge_close"] > 0).to_numpy()
    margin = picks["margin"].to_numpy(dtype=float)
    actual = picks["spread_line"].to_numpy(dtype=float)
    # A home pick needs margin > line, so a LOWER line is better; an away pick the reverse.
    better = actual - step * np.where(home_pick, 1.0, -1.0)

    before = _outcome(margin, actual, home_pick)
    after = _outcome(margin, better, home_pick)

    changed = before != after
    gain = _returns(after) - _returns(before)
    return {
        "threshold": threshold,
        "n": len(picks),
        "changed": int(changed.sum()),
        "changed_rate": float(changed.mean()),
        "loss_to_push": int(((before == "LOSS") & (after == "PUSH")).sum()),
        "push_to_win": int(((before == "PUSH") & (after == "WIN")).sum()),
        "loss_to_win": int(((before == "LOSS") & (after == "WIN")).sum()),
        "ev_gain_per_bet": float(gain.mean()),
    }


def value_by_number(frame: pd.DataFrame, *, step: float) -> pd.DataFrame:
    """Where the half point pays -- broken down by the number the bet sat on."""
    picks = frame.copy()
    home_pick = (picks["edge_close"] > 0).to_numpy()
    margin = picks["margin"].to_numpy(dtype=float)
    actual = picks["spread_line"].to_numpy(dtype=float)
    better = actual - step * np.where(home_pick, 1.0, -1.0)

    before = _outcome(margin, actual, home_pick)
    after = _outcome(margin, better, home_pick)
    picks["abs_line"] = np.abs(actual)
    picks["changed"] = before != after
    picks["gain"] = _returns(after) - _returns(before)

    grouped = picks.groupby("abs_line").agg(
        bets=("changed", "size"),
        changed=("changed", "sum"),
        ev_gain=("gain", "mean"),
    )
    grouped["changed_rate"] = grouped["changed"] / grouped["bets"]
    return grouped.loc[grouped["changed"] > 0].sort_values("changed", ascending=False)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    step = float(args.half_points)
    features = pd.read_parquet(args.features)

    predictions = walk_forward(features, list(REPORT_SEASONS))
    frame = predictions.loc[predictions["spread_line"].notna()].copy()
    frame["edge_close"] = frame["model_margin"] - frame["spread_line"]
    print(
        f"corpus: {len(frame)} games with a closing spread, {REPORT_SEASONS[0]}-{REPORT_SEASONS[-1]}\n"
    )

    print("--- 1. WHY KEY NUMBERS EXIST: how often each exact margin occurs ---")
    freq = margin_frequency(frame)
    top = freq.sort_values("games", ascending=False).head(10)
    print(f"{'margin':>7}{'games':>8}{'share':>9}")
    for _, row in top.iterrows():
        print(f"{int(row['abs_margin']):>7}{int(row['games']):>8}{row['share']:>9.2%}")
    exact = int((frame["margin"] == frame["spread_line"]).sum())
    print(
        f"\ngames landing EXACTLY on the closing spread (a push): {exact}/{len(frame)} "
        f"({exact / len(frame):.2%})"
    )

    print(f"\n--- 2. VALUE OF {step} POINTS BETTER THAN THE NUMBER YOU BET ---")
    print(
        f"{'edge':>6}{'bets':>7}{'changed':>9}{'rate':>8}"
        f"{'L->P':>7}{'P->W':>7}{'L->W':>7}{'EV/bet':>9}"
    )
    for threshold in THRESHOLDS:
        row = half_point_value(frame, threshold=threshold, step=step)
        if not row.get("n"):
            continue
        print(
            f"{threshold:>6.1f}{row['n']:>7}{row['changed']:>9}{row['changed_rate']:>8.2%}"
            f"{row['loss_to_push']:>7}{row['push_to_win']:>7}{row['loss_to_win']:>7}"
            f"{row['ev_gain_per_bet']:>9.2%}"
        )

    print(f"\n--- 3. WHERE THE {step} POINTS PAY: by the number bet ---")
    by_number = value_by_number(frame, step=step)
    print(f"{'line':>7}{'bets':>7}{'changed':>9}{'rate':>8}{'EV/bet':>9}")
    for line, row in by_number.head(12).iterrows():
        print(
            f"{line:>7.1f}{int(row['bets']):>7}{int(row['changed']):>9}"
            f"{row['changed_rate']:>8.2%}{row['ev_gain']:>9.2%}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
