"""Team strength from play-by-play EPA.

team_game_epa reduces raw plays to one row per offense per game. fit_ratings turns those
rows into opponent-adjusted offensive and defensive ratings via ridge regression on
offense-team and defense-team dummies.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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


def fit_ratings(
    team_games: pd.DataFrame,
    target: str = "epa_play",
    alpha: float = 1.0,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Opponent-adjusted offensive and defensive ratings via ridge regression.

    Regresses each team-game's `target` on offense-team and defense-team indicators.
    This separates a team's own quality from the quality of who it happened to play —
    without it, ratings mostly measure schedule luck.

    Ridge shrinkage pulls thin-sample teams toward the league mean, which is what gives
    early-season ratings a sane prior.

    Returns one row per team with `off_rating` and `def_rating`, both oriented so that
    **higher is better**. The raw defense coefficient means "EPA allowed", so it is
    negated here.

    The league mean (the fitted intercept) is on `.attrs["league_mean"]`.

    `weights`, when given, is expected to be aligned to `team_games` (i.e. pre-filter,
    one entry per input row) -- it is filtered along with the rows dropped for a null
    `target` before being passed to the model.
    """
    mask = team_games[target].notna()
    df = team_games[mask].copy()
    if df.empty:
        raise ValueError(f"no rows with non-null {target!r}")

    teams = sorted(set(df["team"]) | set(df["opponent"]))
    off = pd.get_dummies(pd.Categorical(df["team"], categories=teams), prefix="off")
    dfn = pd.get_dummies(pd.Categorical(df["opponent"], categories=teams), prefix="def")
    X = pd.concat([off, dfn], axis=1).astype(float).to_numpy()
    y = df[target].to_numpy(dtype=float)

    if weights is not None:
        weights = np.asarray(weights)[mask.to_numpy()]

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=weights)

    n = len(teams)
    out = pd.DataFrame(
        {
            "team": teams,
            "off_rating": model.coef_[:n],
            "def_rating": -model.coef_[n:],
        }
    )
    out.attrs["league_mean"] = float(model.intercept_)
    return out
