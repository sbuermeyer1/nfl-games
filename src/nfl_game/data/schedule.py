from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_game.data.teams import normalize_team_codes

FINALIZATION_DELAY = timedelta(hours=6)
EASTERN = ZoneInfo("America/New_York")
REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
    "result",
    "total",
    "spread_line",
    "total_line",
}


class ScheduleSchemaError(ValueError):
    pass


def _kickoffs(rows):
    text = rows["gameday"].astype(str) + " " + rows["gametime"].astype(str)
    parsed = pd.to_datetime(text, errors="coerce")
    if parsed.isna().any():
        raise ScheduleSchemaError("schedule contains invalid kickoff date or time")
    return (
        parsed.dt.tz_localize(EASTERN, ambiguous="raise", nonexistent="raise")
        .dt.tz_convert("UTC")
        .astype("datetime64[ns, UTC]")
    )


def normalize_schedule(rows, season):
    missing = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ScheduleSchemaError(f"schedule missing columns: {missing}")
    out = rows.loc[(rows["season"] == season) & (rows["game_type"] == "REG")].copy()
    out = normalize_team_codes(out, ["home_team", "away_team"])
    out["kickoff_at"] = _kickoffs(out)
    if out["game_id"].duplicated().any():
        raise ScheduleSchemaError("schedule contains duplicate game_id values")
    for column in ("spread_line", "total_line", "result", "total"):
        numeric = pd.to_numeric(out[column], errors="coerce")
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ScheduleSchemaError(f"schedule column {column} contains infinite values")
        out[column] = numeric
    return out.sort_values(["kickoff_at", "game_id"]).reset_index(drop=True)


def is_final_game(row, now):
    kickoff = row["kickoff_at"].to_pydatetime()
    return (
        pd.notna(row["result"]) and pd.notna(row["total"]) and now >= kickoff + FINALIZATION_DELAY
    )


def active_prediction_weeks(schedule, now):
    unplayed = schedule.loc[[not is_final_game(row, now) for _, row in schedule.iterrows()]]
    return sorted(int(week) for week in unplayed["week"].unique())[:2]
