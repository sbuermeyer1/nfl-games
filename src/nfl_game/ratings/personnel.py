"""Leak-free roster, depth-chart, and snap-share continuity features."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

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


def _snapshot(rows: pd.DataFrame, season: int, week: int, team: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Select the current roster snapshot under the source-era availability rule."""
    team_rows = rows[(rows["season"] == season) & rows["team"].eq(team)]
    if season < 2025:
        labeled = team_rows[team_rows["week"] == week]
        return labeled[labeled["dt"].isna() | (labeled["dt"] <= cutoff)]
    eligible = team_rows[team_rows["dt"].notna() & (team_rows["dt"] <= cutoff)]
    if eligible.empty:
        return eligible
    return eligible[eligible["dt"] == eligible["dt"].max()]


def _prepared_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    columns = [*_KEY, "player_id", "dt"]
    if rosters.empty or not set(_KEY + ["gsis_id"]).issubset(rosters):
        return pd.DataFrame(columns=columns)
    out = rosters.copy().rename(columns={"gsis_id": "player_id"})
    out["dt"] = pd.to_datetime(out.get("dt"), utc=True, errors="coerce")
    return out[columns].dropna(subset=["player_id"])


def _prepared_depth(depth_charts: pd.DataFrame) -> pd.DataFrame:
    columns = [*_KEY, "player_id", "slot", "dt"]
    if depth_charts.empty or not set(_KEY + ["gsis_id"]).issubset(depth_charts):
        return pd.DataFrame(columns=columns)
    slot = next((name for name in ("depth_chart_position", "position", "slot") if name in depth_charts), None)
    if slot is None:
        return pd.DataFrame(columns=columns)
    out = depth_charts.copy().rename(columns={"gsis_id": "player_id", slot: "slot"})
    out["dt"] = pd.to_datetime(out.get("dt"), utc=True, errors="coerce")
    return out[columns].dropna(subset=["player_id", "slot"])


def _depth_change(depth: pd.DataFrame, season: int, week: int, team: str, cutoff: pd.Timestamp) -> float:
    rows = depth[(depth["season"] == season) & depth["team"].eq(team)]
    if season < 2025:
        rows = rows[(rows["week"] <= week) & (rows["dt"].isna() | (rows["dt"] <= cutoff))]
        moments = sorted(rows["week"].dropna().unique())[-2:]
        snapshots = [rows[rows["week"] == moment] for moment in moments]
    else:
        rows = rows[rows["dt"].notna() & (rows["dt"] <= cutoff)]
        moments = sorted(rows["dt"].unique())[-2:]
        snapshots = [rows[rows["dt"] == moment] for moment in moments]
    if len(snapshots) < 2:
        return 0.0
    prior = snapshots[0].drop_duplicates("slot").set_index("slot")["player_id"]
    current = snapshots[1].drop_duplicates("slot").set_index("slot")["player_id"]
    slots = prior.index.union(current.index)
    return float(sum(prior.get(slot) != current.get(slot) for slot in slots) / len(slots)) if len(slots) else 0.0


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
    depth = _prepared_depth(depth_charts)
    results = []
    for target in games.itertuples(index=False):
        roster = set(_snapshot(roster_rows, target.season, target.week, target.team, target.cutoff)["player_id"])
        if target.week == 1:
            history = raw[(raw["season"] == target.season - 1) & raw["team"].eq(target.team)]
        else:
            history = raw[(raw["season"] == target.season) & (raw["week"] < target.week) & raw["team"].eq(target.team)]
        off_returning, off_hhi, off_coverage = _snap_features(history, "offense_snaps", roster)
        def_returning, def_hhi, def_coverage = _snap_features(history, "defense_snaps", roster)
        coverage = min(off_coverage, def_coverage)
        results.append({
            "season": target.season, "week": target.week, "team": target.team,
            "off_returning_share": off_returning if target.week == 1 else 0.0,
            "def_returning_share": def_returning if target.week == 1 else 0.0,
            "off_snap_hhi": 0.0 if target.week == 1 else off_hhi,
            "def_snap_hhi": 0.0 if target.week == 1 else def_hhi,
            "depth_chart_change_rate": _depth_change(depth, target.season, target.week, target.team, target.cutoff),
            "roster_churn": 1.0 - off_returning if target.week == 1 and history["offense_snaps"].sum() else 0.0,
            "id_coverage": coverage,
            "personnel_imputed": int(coverage < 0.9),
        })
    return pd.DataFrame(results, columns=columns)
