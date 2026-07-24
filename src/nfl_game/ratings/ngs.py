"""Next Gen Stats aggregated to team-weeks (offense only; NGS has no defensive table).

CPOE, rush yards over expected, and separation stabilize faster than box-score yardage,
which is where this layer earns its place: it identifies real team quality earlier in a
season than results alone do.

NGS applies qualifier thresholds, so coverage is incomplete — measured on 2024, passing
covers 539 of 544 team-games but rushing only 468. Missing values are imputed with the
league-week mean and flagged, so the model is told when a number is a guess.
"""

import pandas as pd

PASSING_MAP = {
    "completion_percentage_above_expectation": "cpoe",
    "avg_time_to_throw": "time_to_throw",
    "avg_air_yards_to_sticks": "air_yards_to_sticks",
    "aggressiveness": "aggressiveness",
}
RUSHING_MAP = {
    "rush_yards_over_expected_per_att": "ryoe_per_att",
    "percent_attempts_gte_eight_defenders": "pct_eight_defenders",
}
RECEIVING_MAP = {
    "avg_separation": "separation",
    "avg_yac_above_expectation": "yac_oe",
}

NGS_METRICS = list(PASSING_MAP.values()) + list(RUSHING_MAP.values()) + list(RECEIVING_MAP.values())


def _weighted_team_week(df: pd.DataFrame, mapping: dict[str, str], weight_col: str) -> pd.DataFrame:
    """Collapse player rows to one volume-weighted row per team-week."""
    d = df[(df["season_type"] == "REG") & (df["week"] > 0)].copy()
    if d.empty:
        return pd.DataFrame(columns=["season", "week", "team", *mapping.values()])

    d = d.rename(columns={"team_abbr": "team"})
    d["_w"] = d[weight_col].fillna(0.0)

    out = []
    for (season, week, team), g in d.groupby(["season", "week", "team"]):
        row = {"season": season, "week": week, "team": team}
        total = g["_w"].sum()
        for src, dest in mapping.items():
            if total > 0 and g[src].notna().any():
                valid = g[g[src].notna()]
                vw = valid["_w"].sum()
                row[dest] = (valid[src] * valid["_w"]).sum() / vw if vw > 0 else None
            else:
                row[dest] = None
        out.append(row)
    return pd.DataFrame(out)


def team_week_ngs(
    passing: pd.DataFrame, rushing: pd.DataFrame, receiving: pd.DataFrame
) -> pd.DataFrame:
    """One row per team-week with all eight NGS metrics, imputed and flagged."""
    p = _weighted_team_week(passing, PASSING_MAP, "attempts")
    r = _weighted_team_week(rushing, RUSHING_MAP, "rush_attempts")
    c = _weighted_team_week(receiving, RECEIVING_MAP, "targets")

    keys = ["season", "week", "team"]
    out = p
    for other in (r, c):
        out = out.merge(other, on=keys, how="outer")

    for metric in NGS_METRICS:
        if metric not in out.columns:
            out[metric] = None
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_imputed"] = out[metric].isna().astype(int)
        league_mean = out.groupby(["season", "week"])[metric].transform("mean")
        out[metric] = out[metric].fillna(league_mean).fillna(out[metric].mean()).fillna(0.0)

    ordered = keys + NGS_METRICS + [f"{m}_imputed" for m in NGS_METRICS]
    return out[ordered].sort_values(keys).reset_index(drop=True)
