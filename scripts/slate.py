"""Weekly slate report: model vs market for a given season/week."""

import argparse

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.market.compare import build_slate, slate_markdown
from nfl_game.model.calibrate import Calibrator
from nfl_game.model.predict import GameModel
from nfl_game.paths import PROCESSED_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "gbm"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--edge-threshold", type=float, default=2.0)
    args = ap.parse_args()

    feats = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")

    # Calibrate on out-of-sample predictions from every completed prior season.
    prior_seasons = sorted(s for s in feats["season"].unique() if s < args.season)
    oos = walk_forward(feats, prior_seasons[1:], estimator=args.estimator, alpha=args.alpha)
    calibrator = Calibrator().fit(oos)

    train = feats[feats["season"] < args.season]
    target = feats[(feats["season"] == args.season) & (feats["week"] == args.week)]
    if target.empty:
        raise SystemExit(f"no games found for {args.season} week {args.week}")

    model = GameModel(estimator=args.estimator, alpha=args.alpha).fit(train)
    preds = model.predict(target)
    probs = calibrator.predict(target.merge(preds, on="game_id"))

    slate = build_slate(target, preds, probs, edge_threshold=args.edge_threshold)

    csv_path = PROCESSED_DIR / f"slate_{args.season}_wk{args.week:02d}.csv"
    md_path = PROCESSED_DIR / f"slate_{args.season}_wk{args.week:02d}.md"
    slate.to_csv(csv_path, index=False)
    md_path.write_text(slate_markdown(slate), encoding="utf-8")

    print(slate_markdown(slate))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
