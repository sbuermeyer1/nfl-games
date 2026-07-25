"""Model vs market: the weekly slate.

Both model_spread and market_spread are stated as home-team margins, matching nflverse's
spread_line convention (positive = home favored). One convention end to end is what keeps
sign errors from quietly inverting every pick.

edge_flag marks disagreement above a threshold. It is a flag, not advice — v1 ships no
bet sizing, because staking is only as sound as the calibration underneath it.
"""

import pandas as pd

SLATE_COLS = [
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "model_spread",
    "market_spread",
    "spread_gap",
    "cover_prob",
    "model_total",
    "market_total",
    "total_gap",
    "over_prob",
    "edge_flag",
]


def build_slate(
    features_df: pd.DataFrame,
    preds: pd.DataFrame,
    probs: pd.DataFrame,
    edge_threshold: float = 2.0,
) -> pd.DataFrame:
    """Join predictions and probabilities onto the slate, flag disagreements."""
    df = features_df.merge(preds, on="game_id", how="inner").merge(probs, on="game_id", how="left")

    df["model_spread"] = df["model_margin"].round(2)
    df["market_spread"] = df["spread_line"]
    df["spread_gap"] = (df["model_margin"] - df["spread_line"]).round(2)
    df["model_total"] = df["model_total"].round(2)
    df["market_total"] = df["total_line"]
    df["total_gap"] = (df["model_total"] - df["total_line"]).round(2)
    df["cover_prob"] = df["cover_prob"].round(4)
    df["over_prob"] = df["over_prob"].round(4)
    df["edge_flag"] = (df["spread_gap"].abs() >= edge_threshold).astype(int)

    out = df[SLATE_COLS].copy()
    return out.reindex(out["spread_gap"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def slate_markdown(slate: pd.DataFrame) -> str:
    """Render the slate as a markdown table, edges first."""
    header = (
        "| Game | Model | Market | Gap | Cover% | Model O/U | Market O/U | Gap | Over% | Edge |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in slate.itertuples(index=False):
        cover_pct = "n/a" if pd.isna(r.cover_prob) else f"{r.cover_prob:.1%}"
        over_pct = "n/a" if pd.isna(r.over_prob) else f"{r.over_prob:.1%}"
        rows.append(
            f"| {r.away_team} @ {r.home_team} | {r.model_spread:+.1f} | {r.market_spread:+.1f} "
            f"| {r.spread_gap:+.1f} | {cover_pct} | {r.model_total:.1f} "
            f"| {r.market_total:.1f} | {r.total_gap:+.1f} | {over_pct} "
            f"| {'*' if r.edge_flag else ''} |"
        )
    return header + "\n".join(rows)
