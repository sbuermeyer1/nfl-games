"""As-of team ratings: recency-weighted, opponent-adjusted, leak-free.

Every function here takes an (asof_season, asof_week) cutoff and uses only games
strictly before it. That property is what makes the backtest honest, so it is tested
directly in tests/test_build_ratings.py rather than assumed.
"""

import numpy as np
import pandas as pd

from nfl_game.ratings.epa import fit_ratings

RATING_TARGETS = {
    "epa_play": ("off_rating", "def_rating"),
    "epa_pass": ("off_rating_pass", "def_rating_pass"),
    "epa_rush": ("off_rating_rush", "def_rating_rush"),
}


def decay_weights(
    team_games: pd.DataFrame,
    asof_season: int,
    asof_week: int,
    halflife_games: float = 10.0,
    season_penalty: float = 0.6,
) -> np.ndarray:
    """Exponential recency weights; zero for anything at or after the cutoff.

    Weight falls by half every `halflife_games` weeks of age, and is multiplied by
    `season_penalty` for each completed season of distance. The season penalty is what
    makes Week 1 ratings lean on last year without treating it as current.
    """
    season = team_games["season"].to_numpy()
    week = team_games["week"].to_numpy()

    is_past = (season < asof_season) | ((season == asof_season) & (week < asof_week))

    seasons_back = np.maximum(asof_season - season, 0)
    weeks_back = np.where(season == asof_season, asof_week - week, asof_week + (18 - week))
    weeks_back = np.maximum(weeks_back, 0)

    w = 0.5 ** (weeks_back / halflife_games) * (season_penalty**seasons_back)
    return np.where(is_past, w, 0.0)


def build_ratings(
    team_games: pd.DataFrame,
    asof_season: int,
    asof_week: int,
    halflife_games: float = 10.0,
    season_penalty: float = 0.6,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Opponent-adjusted ratings as of a point in time, for overall/pass/rush EPA."""
    w = decay_weights(team_games, asof_season, asof_week, halflife_games, season_penalty)
    if not (w > 0).any():
        raise ValueError(f"no games before season {asof_season} week {asof_week}")

    used = team_games[w > 0].reset_index(drop=True)
    used_w = w[w > 0]

    out = None
    for target, (off_col, def_col) in RATING_TARGETS.items():
        fitted = fit_ratings(used, target=target, alpha=alpha, weights=used_w)
        fitted = fitted.rename(columns={"off_rating": off_col, "def_rating": def_col})
        out = fitted if out is None else out.merge(fitted, on="team", how="outer")

    return out.reset_index(drop=True)


def ratings_by_week(team_games: pd.DataFrame, seasons: list[int], **kwargs) -> pd.DataFrame:
    """Stack build_ratings across every week of every requested season."""
    frames = []
    for season in seasons:
        weeks = sorted(team_games.loc[team_games["season"] == season, "week"].unique())
        for week in weeks:
            r = build_ratings(team_games, asof_season=season, asof_week=int(week), **kwargs)
            r.insert(0, "week", int(week))
            r.insert(0, "season", season)
            frames.append(r)
    return pd.concat(frames, ignore_index=True)
