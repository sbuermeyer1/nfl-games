"""Leak-free quarterback context and expected-starter features."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_game.ratings.depth import (
    chart_as_of,
    empty_like,
    group_by_team,
    normalize_depth_charts,
)

QB_FEATURE_COLS = (
    "qb_epa_per_db",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_int_rate",
    "qb_change_epa",
    "qb_new_starter",
    "qb_rookie",
    "qb_uncertain",
)
QB_PRIOR_DROPBACKS = 200
ROOKIE_DROPBACK_LIMIT = 100

_KEY = ["season", "week", "team", "player_id"]


def qb_week_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Return one passing-stat row per quarterback, regular season, and team-week."""
    columns = _KEY + [
        "dropbacks",
        "passing_epa",
        "passing_cpoe",
        "passing_interceptions",
        "sacks_suffered",
        "epa_per_db",
    ]
    if player_stats.empty:
        return pd.DataFrame(columns=columns)
    rows = player_stats.copy()
    if "position" in rows:
        rows = rows[rows["position"].eq("QB")]
    required = {"season", "week", "team", "player_id", "attempts", "sacks_suffered"}
    if not required.issubset(rows.columns):
        return pd.DataFrame(columns=columns)
    if "season_type" in rows:
        rows = rows[rows["season_type"].eq("REG")]
    rows["week"] = pd.to_numeric(rows["week"], errors="coerce")
    rows = rows[rows["week"].gt(0)]
    for name in ("passing_epa", "passing_cpoe", "passing_interceptions"):
        rows[name] = pd.to_numeric(rows.get(name, 0), errors="coerce").fillna(0.0)
    rows["attempts"] = pd.to_numeric(rows["attempts"], errors="coerce").fillna(0.0)
    rows["sacks_suffered"] = pd.to_numeric(rows["sacks_suffered"], errors="coerce").fillna(0.0)
    rows["dropbacks"] = rows["attempts"] + rows["sacks_suffered"]
    rows["cpoe_total"] = rows["passing_cpoe"] * rows["dropbacks"]
    out = (
        rows.groupby(_KEY, as_index=False)[
            ["dropbacks", "passing_epa", "cpoe_total", "passing_interceptions", "sacks_suffered"]
        ]
        .sum()
        .sort_values(_KEY)
        .reset_index(drop=True)
    )
    out["passing_cpoe"] = np.divide(
        out.pop("cpoe_total"), out["dropbacks"], out=np.zeros(len(out)), where=out["dropbacks"] > 0
    )
    out["epa_per_db"] = np.divide(
        out["passing_epa"], out["dropbacks"], out=np.zeros(len(out)), where=out["dropbacks"] > 0
    )
    return out[columns]


def normalize_depth_chart_history(depth_charts: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Return the quarterback rows of the canonical depth-chart frame.

    Normalization is NOT done here. The feed arrives as two disjoint schemas and a
    second, partial normalizer is precisely how this path came to see nothing at all:
    every pre-2025 row was dropped for a null `team` (its identity is in `club_code`),
    and the rows that survived carried a null `season`, which no era branch could match.
    """
    columns = ["season", "week", "team", "player_id", "rank", "dt"]
    if depth_charts.empty:
        return pd.DataFrame(columns=columns)
    normalized = normalize_depth_charts(depth_charts)
    position = normalized["position"].astype("string").str.upper()
    # A null position means the frame was already normalized and filtered, so its rows
    # are quarterbacks already. Both live eras DO carry a position, so this tolerance
    # never re-opens the 2025-era hole where every position was kept.
    return normalized.loc[position.isna() | position.eq("QB"), columns].reset_index(drop=True)


def _prior(rows: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return rows[(rows["season"] < season) | ((rows["season"] == season) & (rows["week"] < week))]


def _targets_from_schedule(schedules: pd.DataFrame, targets: list[tuple[int, int]]) -> pd.DataFrame:
    requested = pd.DataFrame(sorted(set(targets)), columns=["season", "week"])
    games = schedules.merge(requested, on=["season", "week"], how="inner")
    pieces = []
    for side in ("home_team", "away_team"):
        if side in games:
            pieces.append(games[["season", "week", side, *[c for c in ("gameday", "gametime", "kickoff_at") if c in games]]].rename(columns={side: "team"}))
    if not pieces:
        return pd.DataFrame(columns=["season", "week", "team", "cutoff"])
    out = pd.concat(pieces, ignore_index=True).drop_duplicates(["season", "week", "team"])
    if "kickoff_at" in out:
        cutoff = pd.to_datetime(out["kickoff_at"], utc=True, errors="coerce")
    else:
        text = out.get("gameday", "").astype(str) + " " + out.get("gametime", "").astype(str)
        cutoff = pd.to_datetime(text, errors="coerce")
        cutoff = cutoff.dt.tz_localize(ZoneInfo("America/New_York"), ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    out["cutoff"] = cutoff
    return out[["season", "week", "team", "cutoff"]].sort_values(["season", "week", "team"])


def _rates(rows: pd.DataFrame) -> dict[str, float]:
    db = rows["dropbacks"].sum()
    if db == 0:
        return {"epa": 0.0, "cpoe": 0.0, "sack": 0.0, "interception": 0.0}
    return {
        "epa": rows["passing_epa"].sum() / db,
        "cpoe": (rows["passing_cpoe"] * rows["dropbacks"]).sum() / db,
        "sack": rows["sacks_suffered"].sum() / db,
        "interception": rows["passing_interceptions"].sum() / db,
    }


def _starter(depth: pd.DataFrame, season: int, week: int, team: str, cutoff: pd.Timestamp) -> str | None:
    """Return the expected starting quarterback under this team's era availability rule."""
    chart = chart_as_of(depth, team, season, week, cutoff)
    if chart.empty:
        return None
    return str(chart.sort_values(["rank", "player_id"]).iloc[0]["player_id"])


def qb_features_for_targets(
    qb_weeks: pd.DataFrame,
    depth_history: pd.DataFrame,
    schedules: pd.DataFrame,
    targets: list[tuple[int, int]],
) -> pd.DataFrame:
    """Build as-of QB features for both teams in each requested scheduled game."""
    games = _targets_from_schedule(schedules, targets)
    columns = ["season", "week", "team", "expected_starter_id", *QB_FEATURE_COLS]
    if games.empty:
        return pd.DataFrame(columns=columns)
    weeks = qb_weeks.copy() if not qb_weeks.empty else pd.DataFrame(columns=qb_week_stats(pd.DataFrame()).columns)
    depth = normalize_depth_chart_history(depth_history, schedules)
    depth_by_team = group_by_team(depth)
    no_depth = empty_like(depth)
    results = []
    for row in games.itertuples(index=False):
        prior = _prior(weeks, int(row.season), int(row.week)) if not weeks.empty else weeks
        league = _rates(prior) if not prior.empty else _rates(pd.DataFrame({"dropbacks": [], "passing_epa": [], "passing_cpoe": [], "sacks_suffered": [], "passing_interceptions": []}))
        recent_team = prior[prior["team"].eq(row.team)] if not prior.empty else prior
        recent_starter = None
        if not recent_team.empty:
            latest = recent_team[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).iloc[-1]
            latest_rows = recent_team[(recent_team["season"] == latest["season"]) & (recent_team["week"] == latest["week"])]
            recent_starter = str(latest_rows.sort_values(["dropbacks", "player_id"], ascending=[False, True]).iloc[0]["player_id"])
        expected = _starter(
            depth_by_team.get(row.team, no_depth),
            int(row.season),
            int(row.week),
            row.team,
            row.cutoff,
        )
        uncertain = int(expected is None)
        if expected is None:
            expected = recent_starter
        player_rows = prior[prior["player_id"].eq(expected)] if expected is not None else prior.iloc[0:0]
        player_db = player_rows["dropbacks"].sum() if not player_rows.empty else 0.0
        player_rates = _rates(player_rows) if not player_rows.empty else league
        weight = player_db + QB_PRIOR_DROPBACKS
        recent_rates = _rates(prior[prior["player_id"].eq(recent_starter)]) if recent_starter is not None else league
        results.append(
            {
                "season": int(row.season), "week": int(row.week), "team": row.team, "expected_starter_id": expected,
                "qb_epa_per_db": (player_rates["epa"] * player_db + league["epa"] * QB_PRIOR_DROPBACKS) / weight,
                "qb_cpoe": (player_rates["cpoe"] * player_db + league["cpoe"] * QB_PRIOR_DROPBACKS) / weight,
                "qb_sack_rate": (player_rates["sack"] * player_db + league["sack"] * QB_PRIOR_DROPBACKS) / weight,
                "qb_int_rate": (player_rates["interception"] * player_db + league["interception"] * QB_PRIOR_DROPBACKS) / weight,
                "qb_change_epa": player_rates["epa"] - recent_rates["epa"],
                "qb_new_starter": int(expected is not None and expected != recent_starter),
                "qb_rookie": int(expected is not None and player_db < ROOKIE_DROPBACK_LIMIT),
                "qb_uncertain": uncertain,
            }
        )
    return pd.DataFrame(results, columns=columns)
