"""NFL data ingestion via nflreadpy (nflverse public data releases).

nflreadpy returns Polars DataFrames; everything here converts to pandas so nothing
downstream of this module has to know Polars exists.
"""

import nflreadpy
import pandas as pd

from nfl_game.data.teams import normalize_team_codes
from nfl_game.paths import RAW_DIR

NGS_STAT_TYPES = ("passing", "rushing", "receiving")

# Team-code columns per source. Normalising here, at the single boundary every feed
# passes through, is what makes the downstream joins line up; see teams.py for the
# three ways the feeds disagreed and what a missed join silently cost.
SCHEDULE_TEAM_COLS = ["home_team", "away_team"]
PBP_TEAM_COLS = ["posteam", "defteam", "home_team", "away_team"]
NGS_TEAM_COLS = ["team_abbr"]


def _seasons_label(seasons: list[int]) -> str:
    seasons = sorted(seasons)
    return f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])


def load_schedules(seasons: list[int] | None = None, save: bool = True) -> pd.DataFrame:
    """Game schedule, results, and closing betting lines.

    Passing seasons=None loads every season (1999+), including future games whose
    result/total are null but whose spread_line/total_line may already be posted.
    """
    requested = True if seasons is None else seasons
    df = nflreadpy.load_schedules(requested).to_pandas()
    df = normalize_team_codes(df, SCHEDULE_TEAM_COLS)
    if seasons is not None:
        df = df[df["season"].isin(seasons)].reset_index(drop=True)
    if save:
        label = _seasons_label(seasons) if seasons else "all"
        df.to_parquet(RAW_DIR / f"schedules_{label}.parquet")
    return df


def load_pbp(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Play-by-play with EPA. Large: roughly 50k rows and 372 columns per season."""
    df = nflreadpy.load_pbp(seasons).to_pandas()
    df = normalize_team_codes(df, PBP_TEAM_COLS)
    if save:
        df.to_parquet(RAW_DIR / f"pbp_{_seasons_label(seasons)}.parquet")
    return df


def load_ngs(seasons: list[int], stat_type: str, save: bool = True) -> pd.DataFrame:
    """Next Gen Stats, 2016+ only. stat_type is one of passing/rushing/receiving.

    Note: rows with week == 0 are season aggregates, not week-zero games. Callers
    doing weekly joins must filter them out.
    """
    if stat_type not in NGS_STAT_TYPES:
        raise ValueError(f"stat_type must be one of {NGS_STAT_TYPES}, got {stat_type!r}")
    df = nflreadpy.load_nextgen_stats(seasons=seasons, stat_type=stat_type).to_pandas()
    df = normalize_team_codes(df, NGS_TEAM_COLS)
    if save:
        df.to_parquet(RAW_DIR / f"ngs_{stat_type}_{_seasons_label(seasons)}.parquet")
    return df
