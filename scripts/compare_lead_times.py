"""CLV at 4, 5 and 6 days out, scored on the SAME games at every lead.

The three corpora differ in which games were priced (1,179 / 1,224 / 1,150), and scoring each
on its own set is exactly the unequal-set error that invalidated the ridge-v2 conclusion. So
everything below is restricted to games priced at all three leads.

The decision this informs: PUBLISH_BEFORE is 4 days. The vintage floor makes any longer lock
safe (it holds a game until its features are fresh), so the only question is whether a longer
lock buys enough CLV to matter. Break-even at -110 needs ~0.48 spread points.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_line_value import closing_line_value

from nfl_game.backtest import walk_forward
from nfl_game.paths import PROCESSED_DIR, RAW_DIR

LH = RAW_DIR / "line_history"
SEASONS = [2021, 2022, 2023, 2024, 2025]
BREAK_EVEN_POINTS = 0.48

sources = {
    "4 days": LH / "line_history_partial.parquet",
    "5 days": LH / "line_history_combined_d05.parquet",
    "6 days": LH / "line_history_combined_d06.parquet",
}

frames = {}
for label, path in sources.items():
    d = pd.read_parquet(path)
    frames[label] = d.loc[d["early_spread_line"].notna()]
    print(f"{label}: {len(frames[label])} priced")

matched = set.intersection(*(set(d["game_id"]) for d in frames.values()))
print(f"MATCHED (priced at all three): {len(matched)}\n")

features = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
preds = walk_forward(features, SEASONS)

results = {}
for label, hist in frames.items():
    h = hist.loc[hist["game_id"].isin(matched)]
    m = preds.merge(
        h[["game_id", "early_spread_line", "early_total_line"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    m["edge_early"] = m["model_margin"] - m["early_spread_line"]
    m["line_move"] = m["spread_line"] - m["early_spread_line"]
    results[label] = {t: closing_line_value(m, threshold=t) for t in (1.0, 2.0, 3.0)}

print("--- SPREAD CLV, same games at every lead ---")
print(f"{'lead':>8}{'edge':>6}{'n':>7}{'mean CLV':>11}{'z':>8}{'beat close':>12}{'vs 0.48':>10}")
for label in sources:
    for t in (1.0, 2.0, 3.0):
        r = results[label][t]
        if not r.get("n"):
            continue
        clv = r["mean_clv_points"]
        print(
            f"{label:>8}{t:>6.1f}{r['n']:>7}{clv:>11.4f}{r['z']:>8.2f}"
            f"{r['beat_close_rate']:>12.1%}{clv / BREAK_EVEN_POINTS:>9.0%}"
        )

print("\n--- marginal gain from each extra day (same games) ---")
for t in (1.0, 2.0, 3.0):
    a = results["4 days"][t]["mean_clv_points"]
    b = results["5 days"][t]["mean_clv_points"]
    c = results["6 days"][t]["mean_clv_points"]
    print(
        f"  edge>={t:.0f}:  4d {a:.4f}  ->  5d {b:.4f} ({b - a:+.4f})  ->  6d {c:.4f} ({c - b:+.4f})"
    )
