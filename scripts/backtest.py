"""Walk-forward backtest report."""

import argparse

import pandas as pd

from nfl_game.backtest import ats_by_threshold, evaluate, market_comparison_regression, walk_forward
from nfl_game.paths import PROCESSED_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", default="2021-2025")
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "gbm"])
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    lo, _, hi = args.test_seasons.partition("-")
    seasons = list(range(int(lo), int(hi or lo) + 1))

    feats = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    preds = walk_forward(feats, seasons, estimator=args.estimator, alpha=args.alpha)

    m = evaluate(preds)
    print(f"\n=== {args.estimator} | test seasons {seasons[0]}-{seasons[-1]} ===")
    print(f"games:            {m['n_games']}")
    print(f"margin MAE:       {m['margin_mae']:.3f}   market: {m['market_margin_mae']:.3f}")
    print(f"total  MAE:       {m['total_mae']:.3f}   market: {m['market_total_mae']:.3f}")
    print(f"ATS hit rate:     {m['ats_hit_rate']:.4f}  (n={m['ats_n']}, break-even 0.5240)")
    print(f"O/U hit rate:     {m['ou_hit_rate']:.4f}  (n={m['ou_n']})")

    print("\n--- ATS by edge threshold ---")
    print(ats_by_threshold(preds).to_string(index=False))

    r = market_comparison_regression(preds)
    print("\n--- does the model add anything to the line? ---")
    print(f"market coef: {r['market_coef']:.4f}")
    print(f"model  coef: {r['model_coef']:.4f}   <- near zero means it adds nothing")
    print(f"r2: {r['r2']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
