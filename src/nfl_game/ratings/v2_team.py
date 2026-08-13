"""Situational, opponent-adjusted team ratings for Ridge v2."""

import numpy as np
import pandas as pd

from nfl_game.ratings.build import decay_weights
from nfl_game.ratings.epa import fit_ratings

V2_RATING_TARGETS = (
    "epa_play",
    "epa_pass",
    "epa_rush",
    "success_rate",
    "early_down_epa",
    "neutral_epa",
    "explosive_pass_rate",
    "explosive_rush_rate",
    "sack_rate",
)

_KEY_COLUMNS = ["game_id", "season", "week", "team", "opponent", "is_home"]
_TEAM_GAME_COLUMNS = _KEY_COLUMNS + list(V2_RATING_TARGETS)


def team_game_v2(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regular-season scrimmage plays into one offensive row per game/team."""
    plays = pbp[
        (pbp["season_type"] == "REG")
        & pbp["posteam"].notna()
        & pbp["epa"].notna()
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
    ].copy()
    if plays.empty:
        return pd.DataFrame(columns=_TEAM_GAME_COLUMNS)

    plays["is_pass"] = plays["pass"] == 1
    plays["is_rush"] = plays["rush"] == 1
    plays["is_sack"] = plays["sack"] == 1
    plays["is_pass_attempt"] = plays["is_pass"] & ~plays["is_sack"]
    plays["is_early_down"] = plays["down"].isin((1, 2))
    score_column = next(
        (
            column
            for column in ("score_differential", "posteam_score_differential")
            if column in plays
        ),
        None,
    )
    score_differential = (
        pd.to_numeric(plays[score_column], errors="coerce")
        if score_column is not None
        else pd.Series(np.nan, index=plays.index, dtype=float)
    )
    plays["is_neutral"] = plays["qtr"].between(1, 3) & score_differential.abs().le(8)
    dropbacks = plays["qb_dropback"] if "qb_dropback" in plays else plays["pass"]
    plays["is_dropback"] = dropbacks == 1

    grouped = plays.groupby(["game_id", "season", "week", "posteam", "defteam"], dropna=True)

    def aggregate(game: pd.DataFrame) -> pd.Series:
        pass_attempts = game["is_pass_attempt"]
        rushes = game["is_rush"]
        dropbacks = game["is_dropback"]
        return pd.Series(
            {
                "epa_play": game["epa"].mean(),
                "epa_pass": game.loc[game["is_pass"], "epa"].mean(),
                "epa_rush": game.loc[rushes, "epa"].mean(),
                "success_rate": game["success"].mean(),
                "early_down_epa": game.loc[game["is_early_down"], "epa"].mean(),
                "neutral_epa": game.loc[game["is_neutral"], "epa"].mean(),
                "explosive_pass_rate": (
                    (game.loc[pass_attempts, "yards_gained"] >= 20).mean()
                    if pass_attempts.any()
                    else np.nan
                ),
                "explosive_rush_rate": (
                    (game.loc[rushes, "yards_gained"] >= 10).mean() if rushes.any() else np.nan
                ),
                "sack_rate": game.loc[dropbacks, "is_sack"].mean() if dropbacks.any() else np.nan,
            }
        )

    out = grouped.apply(aggregate, include_groups=False).reset_index()
    out = out.rename(columns={"posteam": "team", "defteam": "opponent"})
    home = pbp[["game_id", "home_team"]].dropna().drop_duplicates("game_id")
    out = out.merge(home, on="game_id", how="left")
    out["is_home"] = (out["team"] == out["home_team"]).astype(int)
    return out[_TEAM_GAME_COLUMNS].sort_values(["season", "week", "team"]).reset_index(drop=True)


def v2_team_ratings(
    team_games: pd.DataFrame,
    targets: list[tuple[int, int]],
    short_halflife: float,
    long_halflife: float,
    prior_season_weight: float,
) -> pd.DataFrame:
    """Build leak-free short- and long-window ratings for requested season-week targets."""
    frames = []
    windows = (("short", short_halflife), ("long", long_halflife))
    for season, week in sorted(set(targets)):
        ratings = None
        for window, halflife in windows:
            weights = decay_weights(
                team_games,
                int(season),
                int(week),
                halflife_games=halflife,
                season_penalty=prior_season_weight,
            )
            used = team_games.loc[weights > 0].reset_index(drop=True)
            used_weights = weights[weights > 0]
            if used.empty:
                raise ValueError(f"no games before season {season} week {week}")
            for target in V2_RATING_TARGETS:
                fitted = fit_ratings(used, target=target, weights=used_weights).rename(
                    columns={
                        "off_rating": f"{window}_off_{target}",
                        "def_rating": f"{window}_def_{target}",
                    }
                )
                ratings = (
                    fitted if ratings is None else ratings.merge(fitted, on="team", how="outer")
                )
        ratings.insert(0, "week", int(week))
        ratings.insert(0, "season", int(season))
        frames.append(ratings)
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team"])
    return pd.concat(frames, ignore_index=True)
