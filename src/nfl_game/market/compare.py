"""Model vs market: the weekly slate.

Both model_spread and market_spread are stated as home-team margins, matching nflverse's
spread_line convention (positive = home favored). One convention end to end is what keeps
sign errors from quietly inverting every pick.

edge_flag marks disagreement above a threshold. It is a flag, not advice — v1 ships no
bet sizing, because staking is only as sound as the calibration underneath it.

The QB advisory columns are presentation only. They are joined AFTER prediction and
after edge_flag is computed, so their presence, absence or staleness can never move
model_spread, model_total or edge_flag -- a property tests/test_compare.py pins
directly. qb_watch is a separate marker from edge_flag on purpose: suppressing an edge
on a starter change would make one flag mean two things and silently hide edges.
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
    "home_qb",
    "away_qb",
    "qb_change_epa_home",
    "qb_change_epa_away",
    "qb_watch",
    "qb_inferred",
]

_ADVISORY_DTYPES = {
    "home_qb": "string",
    "away_qb": "string",
    "qb_change_epa_home": "float64",
    "qb_change_epa_away": "float64",
    "qb_watch": "Int64",
    "qb_inferred": "Int64",
}


def build_slate(
    features_df: pd.DataFrame,
    preds: pd.DataFrame,
    probs: pd.DataFrame,
    edge_threshold: float = 2.0,
    starters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join predictions and probabilities onto the slate, flag disagreements."""
    df = features_df.merge(preds, on="game_id", how="inner", validate="one_to_one").merge(
        probs, on="game_id", how="left", validate="one_to_one"
    )

    df["model_spread"] = df["model_margin"].round(2)
    df["market_spread"] = df["spread_line"]
    df["spread_gap"] = (df["model_margin"] - df["spread_line"]).round(2)
    # total_gap from the unrounded model_total, matching spread_gap's use of the
    # unrounded model_margin -- rounding model_total in place before taking the gap
    # would make the two gap columns inconsistent about whether they're computed
    # pre- or post-round.
    df["total_gap"] = (df["model_total"] - df["total_line"]).round(2)
    df["model_total"] = df["model_total"].round(2)
    df["market_total"] = df["total_line"]
    df["cover_prob"] = df["cover_prob"].round(4)
    df["over_prob"] = df["over_prob"].round(4)
    df["edge_flag"] = (df["spread_gap"].abs() >= edge_threshold).astype(int)

    # Joined after edge_flag on purpose: every column above this line is computed from
    # the model and the market alone, so no advisory failure mode can reach them.
    # game_id's dtype must not depend on whether `starters` was supplied -- it is a
    # pre-existing column, not one of the six advisory columns, so casting it only
    # inside this branch would make the dtype flip based on an optional argument.
    df["game_id"] = df["game_id"].astype("string")
    if starters is not None and not starters.empty:
        advisory = starters.drop_duplicates("game_id").copy()
        advisory["game_id"] = advisory["game_id"].astype("string")
        df = df.merge(advisory, on="game_id", how="left", validate="one_to_one")
    for name, dtype in _ADVISORY_DTYPES.items():
        if name not in df:
            # pd.NA is the fill for the nullable extension dtypes (string, Int64) but
            # a numpy dtype like float64 can't be constructed from a bare pd.NA scalar.
            fill = pd.NA if pd.api.types.is_extension_array_dtype(dtype) else float("nan")
            df[name] = pd.Series(fill, index=df.index, dtype=dtype)
        else:
            df[name] = df[name].astype(dtype)

    out = df[SLATE_COLS].copy()
    return out.reindex(out["spread_gap"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def _fmt(value: float, spec: str) -> str:
    """Format a number, or "n/a" for a missing value.

    A NaN reads as a data-quality bug to anyone consuming this report -- indistinguishable
    from "the pipeline broke" -- when the actual meaning is "no line was available for
    this side yet, so nothing could be computed." That applies equally to every column
    derived from a possibly-missing line (market_spread, spread_gap, market_total,
    total_gap, cover_prob, over_prob), not just the probability columns.
    """
    return "n/a" if pd.isna(value) else format(value, spec)


def _qb_cell(row) -> str:
    """Render the advisory for one game, in one of four distinct ways.

    `qb_watch` null: the advisory could not be built at all -- "n/a". A blank cell
    would be indistinguishable from a confirmed "no change".

    `qb_watch == 0` and `qb_inferred == 1`: no depth chart has published yet for one
    or both sides, so the starter shown is inferred from last week rather than read
    from a chart -- "unconfirmed". This must NOT render the same as a genuine
    no-change (blank), which is what let an unpublished chart pass as "no starter
    changes" before a reader ever saw a real one.

    `qb_watch == 0` and `qb_inferred == 0`: both charts published and there is
    genuinely no change -- blank.

    `qb_watch == 1`: a change was detected -- the existing description, with its
    "(inferred)" suffix when the change itself rests on an inferred side.
    """
    if pd.isna(row.qb_watch):
        return "n/a"
    inferred = not pd.isna(row.qb_inferred) and row.qb_inferred == 1
    if row.qb_watch == 0:
        return "unconfirmed" if inferred else ""
    parts = []
    for name, delta in (
        (row.home_qb, row.qb_change_epa_home),
        (row.away_qb, row.qb_change_epa_away),
    ):
        if pd.isna(delta) or delta == 0.0:
            continue
        label = "unknown" if pd.isna(name) else name
        parts.append(f"{label} {delta:+.2f}")
    suffix = " (inferred)" if inferred else ""
    return ("; ".join(parts) or "change") + suffix


def slate_markdown(slate: pd.DataFrame) -> str:
    """Render the slate as a markdown table, edges first."""
    header = (
        "| Game | Model | Market | Gap | Cover% | Model O/U | Market O/U | Gap | Over% "
        "| Edge | QB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in slate.itertuples(index=False):
        market_spread = _fmt(r.market_spread, "+.1f")
        spread_gap = _fmt(r.spread_gap, "+.1f")
        cover_pct = "n/a" if pd.isna(r.cover_prob) else f"{r.cover_prob:.1%}"
        market_total = _fmt(r.market_total, ".1f")
        total_gap = _fmt(r.total_gap, "+.1f")
        over_pct = "n/a" if pd.isna(r.over_prob) else f"{r.over_prob:.1%}"
        rows.append(
            f"| {r.away_team} @ {r.home_team} | {r.model_spread:+.1f} | {market_spread} "
            f"| {spread_gap} | {cover_pct} | {r.model_total:.1f} "
            f"| {market_total} | {total_gap} | {over_pct} "
            f"| {'*' if r.edge_flag else ''} "
            f"| {_qb_cell(r)} |"
        )
    return header + "\n".join(rows)
