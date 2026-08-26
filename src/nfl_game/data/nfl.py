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


PLAYER_STATS_TEAM_COLS = ["team", "opponent_team"]
ROSTER_TEAM_COLS = ["team"]
DEPTH_CHART_TEAM_COLS = ["team"]
SNAP_TEAM_COLS = ["team", "opponent"]
PFR_TEAM_COLS = ["team", "opponent"]
PFR_STAT_TYPES = ("pass", "rush", "rec", "def")


def load_player_stats(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Weekly player and team statistics."""
    df = nflreadpy.load_player_stats(seasons, summary_level="week").to_pandas()
    df = normalize_team_codes(df, PLAYER_STATS_TEAM_COLS)
    if save:
        df.to_parquet(RAW_DIR / f"player_stats_{_seasons_label(seasons)}.parquet")
    return df


def load_players(save: bool = True) -> pd.DataFrame:
    """Player identity crosswalk without a season restriction."""
    df = nflreadpy.load_players().to_pandas()
    if save:
        df.to_parquet(RAW_DIR / "players.parquet")
    return df


def load_rosters_weekly(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Weekly roster snapshots."""
    df = nflreadpy.load_rosters_weekly(seasons).to_pandas()
    df = normalize_team_codes(df, ROSTER_TEAM_COLS)
    if save:
        df.to_parquet(RAW_DIR / f"rosters_weekly_{_seasons_label(seasons)}.parquet")
    return df


def load_depth_charts(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Point-in-time depth charts with UTC timestamps."""
    df = nflreadpy.load_depth_charts(seasons).to_pandas()
    df = normalize_team_codes(df, DEPTH_CHART_TEAM_COLS)
    if "dt" in df:
        df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="raise")
    if save:
        df.to_parquet(RAW_DIR / f"depth_charts_{_seasons_label(seasons)}.parquet")
    return df


def load_snap_counts(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Weekly offensive and defensive snap counts."""
    df = nflreadpy.load_snap_counts(seasons).to_pandas()
    df = normalize_team_codes(df, SNAP_TEAM_COLS)
    if save:
        df.to_parquet(RAW_DIR / f"snap_counts_{_seasons_label(seasons)}.parquet")
    return df


def load_pfr_advstats(
    seasons: list[int], stat_type: str, save: bool = True
) -> pd.DataFrame:
    """Weekly Pro Football Reference advanced statistics."""
    if stat_type not in PFR_STAT_TYPES:
        raise ValueError(f"stat_type must be one of {PFR_STAT_TYPES}, got {stat_type!r}")
    df = nflreadpy.load_pfr_advstats(seasons, stat_type, "week").to_pandas()
    df = normalize_team_codes(df, PFR_TEAM_COLS)
    if save:
        df.to_parquet(RAW_DIR / f"pfr_{stat_type}_{_seasons_label(seasons)}.parquet")
    return df


def load_ftn_charting(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """FTN charting data, available from the 2022 season onward."""
    df = nflreadpy.load_ftn_charting(seasons).to_pandas()
    if save:
        df.to_parquet(RAW_DIR / f"ftn_charting_{_seasons_label(seasons)}.parquet")
    return df
