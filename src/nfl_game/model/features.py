"""Game-level feature assembly.

Joins as-of team ratings and trailing NGS onto each scheduled game. Games that have not
been played yet are kept with null targets, so the same function serves both training
and prediction of an upcoming slate.
"""

import numpy as np
import pandas as pd

from nfl_game.ratings.ngs import NGS_METRICS

FEATURE_COLS = [
    "off_pass_edge_home",
    "off_rush_edge_home",
    "off_pass_edge_away",
    "off_rush_edge_away",
    "net_rating_diff",
    "rest_diff",
    "is_dome",
    "temp_outdoor",
    "wind_outdoor",
    "div_game",
    "cpoe_diff",
    "ryoe_diff",
    "separation_diff",
    "ngs_imputed_any",
]

TARGET_COLS = ["margin", "total_points"]

DOME_ROOFS = {"dome", "closed"}


def _trailing_ngs(ngs: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Decay-weighted mean of each team's NGS over weeks strictly before each week."""
    frames = []
    for (season, team), g in ngs.groupby(["season", "team"]):
        g = g.sort_values("week")
        weeks = g["week"].to_numpy()
        for i, week in enumerate(weeks):
            prior = g.iloc[:i]
            row = {"season": season, "team": team, "week": week}
            if prior.empty:
                for m in NGS_METRICS:
                    row[f"trail_{m}"] = np.nan
                row["trail_imputed_any"] = 1
            else:
                age = week - prior["week"].to_numpy()
                w = 0.5 ** (age / halflife)
                for m in NGS_METRICS:
                    row[f"trail_{m}"] = float(np.average(prior[m].to_numpy(), weights=w))
                flags = [f"{m}_imputed" for m in NGS_METRICS if f"{m}_imputed" in prior.columns]
                row["trail_imputed_any"] = int(prior[flags].to_numpy().max()) if flags else 0
            frames.append(row)
    out = pd.DataFrame(frames)
    for m in NGS_METRICS:
        out[f"trail_{m}"] = out[f"trail_{m}"].fillna(0.0)
    return out


def build_game_features(
    schedules: pd.DataFrame,
    ratings: pd.DataFrame,
    ngs: pd.DataFrame,
    ngs_halflife: float = 4.0,
) -> pd.DataFrame:
    """One row per regular-season game with model features and targets."""
    g = schedules[schedules["game_type"] == "REG"].copy()

    trail = _trailing_ngs(ngs, ngs_halflife)

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        r = ratings.rename(columns={"team": team_col})
        r = r.rename(
            columns={c: f"{side}_{c}" for c in r.columns if c.startswith(("off_", "def_"))}
        )
        g = g.merge(r, on=["season", "week", team_col], how="left")

        t = trail.rename(columns={"team": team_col})
        t = t.rename(columns={c: f"{side}_{c}" for c in t.columns if c.startswith("trail_")})
        g = g.merge(t, on=["season", "week", team_col], how="left")

    g["off_pass_edge_home"] = g["home_off_rating_pass"] - g["away_def_rating_pass"]
    g["off_rush_edge_home"] = g["home_off_rating_rush"] - g["away_def_rating_rush"]
    g["off_pass_edge_away"] = g["away_off_rating_pass"] - g["home_def_rating_pass"]
    g["off_rush_edge_away"] = g["away_off_rating_rush"] - g["home_def_rating_rush"]
    g["net_rating_diff"] = (g["home_off_rating"] + g["home_def_rating"]) - (
        g["away_off_rating"] + g["away_def_rating"]
    )

    g["rest_diff"] = g["home_rest"] - g["away_rest"]
    g["is_dome"] = g["roof"].isin(DOME_ROOFS).astype(int)
    g["temp_outdoor"] = np.where(g["is_dome"] == 1, 0.0, g["temp"].fillna(60.0))
    g["wind_outdoor"] = np.where(g["is_dome"] == 1, 0.0, g["wind"].fillna(0.0))
    g["div_game"] = g["div_game"].fillna(0).astype(int)

    g["cpoe_diff"] = g["home_trail_cpoe"] - g["away_trail_cpoe"]
    g["ryoe_diff"] = g["home_trail_ryoe_per_att"] - g["away_trail_ryoe_per_att"]
    g["separation_diff"] = g["home_trail_separation"] - g["away_trail_separation"]
    g["ngs_imputed_any"] = (
        g[["home_trail_imputed_any", "away_trail_imputed_any"]].fillna(1).max(axis=1).astype(int)
    )

    g["margin"] = g["result"]
    g["total_points"] = g["total"]

    keep = [
        "game_id", "season", "week", "home_team", "away_team",
        "spread_line", "total_line", *FEATURE_COLS, *TARGET_COLS,
    ]
    out = g[keep].copy()
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)
    return out.reset_index(drop=True)
