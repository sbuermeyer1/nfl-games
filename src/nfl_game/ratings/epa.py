"""Team strength from play-by-play EPA.

team_game_epa reduces raw plays to one row per offense per game. fit_ratings (Task 4)
turns those rows into opponent-adjusted offensive and defensive ratings.
"""

import pandas as pd

TEAM_GAME_COLS = [
    "game_id",
    "season",
    "week",
    "team",
    "opponent",
    "is_home",
    "epa_play",
    "epa_pass",
    "epa_rush",
    "success_rate",
    "n_pass",
    "n_rush",
]


def team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense) with that offense's EPA per play.

    Keeps regular-season scrimmage plays with a non-null EPA. The pass/rush split uses
    nflverse's `pass`/`rush` indicators rather than `play_type`, so scrambles and sacks
    count as dropbacks.
    """
    df = pbp[
        (pbp["season_type"] == "REG")
        & pbp["posteam"].notna()
        & pbp["epa"].notna()
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
    ].copy()

    df["is_pass"] = df["pass"] == 1
    df["is_rush"] = df["rush"] == 1

    grouped = df.groupby(["game_id", "season", "week", "posteam", "defteam"], dropna=True)

    out = grouped.apply(
        lambda g: pd.Series(
            {
                "epa_play": g["epa"].mean(),
                "epa_pass": g.loc[g["is_pass"], "epa"].mean(),
                "epa_rush": g.loc[g["is_rush"], "epa"].mean(),
                "success_rate": g["success"].mean(),
                "n_pass": int(g["is_pass"].sum()),
                "n_rush": int(g["is_rush"].sum()),
            }
        ),
        include_groups=False,
    ).reset_index()

    out = out.rename(columns={"posteam": "team", "defteam": "opponent"})

    if len(out) > 0:
        home = pbp[["game_id", "home_team"]].dropna().drop_duplicates("game_id")
        out = out.merge(home, on="game_id", how="left")
        out["is_home"] = (out["team"] == out["home_team"]).astype(int)
    else:
        out["is_home"] = pd.Series(dtype=int)

    if len(out) > 0:
        out["n_pass"] = out["n_pass"].astype(int)
        out["n_rush"] = out["n_rush"].astype(int)
        return out[TEAM_GAME_COLS].sort_values(["season", "week", "team"]).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=TEAM_GAME_COLS)
