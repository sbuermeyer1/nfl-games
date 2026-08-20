"""Leak-free roster, depth-chart, and snap-share continuity features."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from nfl_game.ratings.depth import depth_change_rate, normalize_depth_charts

PERSONNEL_FEATURE_COLS = (
    "off_returning_share", "def_returning_share", "off_snap_hhi", "def_snap_hhi",
    "depth_chart_change_rate", "roster_churn", "id_coverage", "personnel_imputed",
)
_KEY = ["season", "week", "team"]


def player_id_map(players: pd.DataFrame) -> pd.DataFrame:
    """Return the public PFR-to-GSIS identity crosswalk, never name-matched."""
    columns = ["pfr_player_id", "player_id"]
    pfr_column = next((name for name in ("pfr_id", "pfr_player_id") if name in players), None)
    if players.empty or pfr_column is None or "gsis_id" not in players:
        return pd.DataFrame(columns=columns)
    out = players[[pfr_column, "gsis_id"]].rename(columns={pfr_column: "pfr_player_id", "gsis_id": "player_id"})
    return out.dropna().drop_duplicates("pfr_player_id").sort_values("pfr_player_id").reset_index(drop=True)


def normalize_snap_counts(snaps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach GSIS IDs to public PFR snap rows; rows without an ID are excluded."""
    columns = [*_KEY, "player_id", "offense_snaps", "defense_snaps"]
    if snaps.empty or not set(_KEY + ["pfr_player_id"]).issubset(snaps):
        return pd.DataFrame(columns=columns)
    rows = snaps.copy()
    for column in ("offense_snaps", "defense_snaps"):
        rows[column] = pd.to_numeric(rows.get(column, 0), errors="coerce").fillna(0.0)
    out = rows.merge(player_id_map(players), on="pfr_player_id", how="inner")
    return out[columns].reset_index(drop=True)


def _targets_from_schedule(schedules: pd.DataFrame, targets: list[tuple[int, int]]) -> pd.DataFrame:
    requested = pd.DataFrame(sorted(set(targets)), columns=["season", "week"])
    games = schedules.merge(requested, on=["season", "week"], how="inner")
    pieces = []
    for side in ("home_team", "away_team"):
        if side in games:
            columns = ["season", "week", side, *[c for c in ("kickoff_at", "gameday", "gametime") if c in games]]
            pieces.append(games[columns].rename(columns={side: "team"}))
    if not pieces:
        return pd.DataFrame(columns=[*_KEY, "cutoff"])
    out = pd.concat(pieces, ignore_index=True).drop_duplicates(_KEY)
    if "kickoff_at" in out:
        cutoff = pd.to_datetime(out["kickoff_at"], utc=True, errors="coerce")
    else:
        text = out.get("gameday", "").astype(str) + " " + out.get("gametime", "").astype(str)
        cutoff = pd.to_datetime(text, errors="coerce").dt.tz_localize(ZoneInfo("America/New_York")).dt.tz_convert("UTC")
    out["cutoff"] = cutoff
    return out[[*_KEY, "cutoff"]].sort_values(_KEY).reset_index(drop=True)


def _timestamped_seasons(rows: pd.DataFrame) -> frozenset[int]:
    """Return the seasons this feed addresses by timestamp rather than by week label.

    The feeds differ in SHAPE, not just in schema, and the shape is what the rule turns
    on. Every season of rosters_weekly and every pre-2025 depth chart is week-labelled
    and carries no `dt`; the 2025-era depth feed carries `dt` and no week label at all.
    Keying the rule to `season >= 2025` therefore empties every 2025 roster snapshot,
    and because the resulting zeros are non-null the block still reports 1.000000
    coverage - completeness, not correctness.
    """
    if rows.empty or "dt" not in rows or "week" not in rows:
        return frozenset()
    labelled = rows.groupby(rows["season"], dropna=True).agg(
        has_dt=("dt", lambda values: bool(values.notna().any())),
        has_week=("week", lambda values: bool(values.notna().any())),
    )
    eligible = labelled[labelled["has_dt"] & ~labelled["has_week"]]
    return frozenset(int(season) for season in eligible.index)


def _snapshot(
    rows: pd.DataFrame,
    season: int,
    week: int,
    team: str,
    cutoff: pd.Timestamp,
    timestamped: bool,
) -> pd.DataFrame:
    """Select the current roster snapshot under the source-era availability rule."""
    team_rows = rows[(rows["season"] == season) & rows["team"].eq(team)]
    if not timestamped:
        labeled = team_rows[team_rows["week"] == week]
        return labeled[labeled["dt"].isna() | (labeled["dt"] <= cutoff)]
    eligible = team_rows[team_rows["dt"].notna() & (team_rows["dt"] <= cutoff)]
    if eligible.empty:
        return eligible
    return eligible[eligible["dt"] == eligible["dt"].max()]


def _utc_dt(frame: pd.DataFrame) -> pd.Series:
    """Coerce `dt` to UTC.

    An ABSENT column is the live case, not a corner case: rosters_weekly never carries
    `dt` and depth charts only carry it from 2025. `frame.get("dt")` would hand
    `pd.to_datetime` a None, which returns a scalar NaT and broadcasts a tz-NAIVE column
    that then cannot be compared to the tz-aware kickoff cutoff.
    """
    values = (
        frame["dt"]
        if "dt" in frame
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    )
    return pd.to_datetime(values, utc=True, errors="coerce")


def _prepared_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    columns = [*_KEY, "player_id", "dt"]
    if rosters.empty or not set(_KEY + ["gsis_id"]).issubset(rosters):
        return pd.DataFrame(columns=columns)
    out = rosters.copy().rename(columns={"gsis_id": "player_id"})
    out["dt"] = _utc_dt(out)
    return out[columns].dropna(subset=["player_id"])


def _snap_features(rows: pd.DataFrame, unit: str, roster: set[str]) -> tuple[float, float, float]:
    total = float(rows[unit].sum())
    mapped = rows.dropna(subset=["player_id"])
    mapped_total = float(mapped[unit].sum())
    coverage = mapped_total / total if total else 1.0
    if not mapped_total:
        return 0.0, 0.0, coverage
    shares = mapped.groupby("player_id")[unit].sum() / mapped_total
    returning = float(shares[shares.index.isin(roster)].sum())
    hhi = float((shares**2).sum())
    return returning, hhi, coverage


def personnel_features_for_targets(
    snaps: pd.DataFrame,
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    players: pd.DataFrame,
    schedules: pd.DataFrame,
    targets: list[tuple[int, int]],
) -> pd.DataFrame:
    """Build roster-continuity features using only information available at kickoff."""
    games = _targets_from_schedule(schedules, targets)
    columns = [*_KEY, *PERSONNEL_FEATURE_COLS]
    if games.empty:
        return pd.DataFrame(columns=columns)
    raw = snaps.copy()
    for unit in ("offense_snaps", "defense_snaps"):
        raw[unit] = pd.to_numeric(raw.get(unit, 0), errors="coerce").fillna(0.0)
    mapping = player_id_map(players)
    raw = raw.merge(mapping, on="pfr_player_id", how="left")
    roster_rows = _prepared_rosters(rosters)
    depth = normalize_depth_charts(depth_charts)
    roster_timestamped = _timestamped_seasons(roster_rows)
    results = []
    for target in games.itertuples(index=False):
        snapshot = _snapshot(
            roster_rows,
            target.season,
            target.week,
            target.team,
            target.cutoff,
            target.season in roster_timestamped,
        )
        roster = set(snapshot["player_id"])
        if target.week == 1:
            history = raw[(raw["season"] == target.season - 1) & raw["team"].eq(target.team)]
        else:
            history = raw[(raw["season"] == target.season) & (raw["week"] < target.week) & raw["team"].eq(target.team)]
        off_returning, off_hhi, off_coverage = _snap_features(history, "offense_snaps", roster)
        def_returning, def_hhi, _def_coverage = _snap_features(history, "defense_snaps", roster)
        total_snap_mass = float(history[["offense_snaps", "defense_snaps"]].sum().sum())
        mapped_snap_mass = float(history.dropna(subset=["player_id"])[["offense_snaps", "defense_snaps"]].sum().sum())
        coverage = mapped_snap_mass / total_snap_mass if total_snap_mass else 1.0
        results.append({
            "season": target.season, "week": target.week, "team": target.team,
            "off_returning_share": off_returning if target.week == 1 else 0.0,
            "def_returning_share": def_returning if target.week == 1 else 0.0,
            "off_snap_hhi": 0.0 if target.week == 1 else off_hhi,
            "def_snap_hhi": 0.0 if target.week == 1 else def_hhi,
            "depth_chart_change_rate": depth_change_rate(
                depth, target.team, target.season, target.week, target.cutoff
            ),
            "roster_churn": 1.0 - off_returning if target.week == 1 and history["offense_snaps"].sum() and off_coverage > 0 else 0.0,
            "id_coverage": coverage,
            "personnel_imputed": int(coverage < 0.9),
        })
    return pd.DataFrame(results, columns=columns)
