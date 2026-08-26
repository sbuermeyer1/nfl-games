"""Leak-free candidate features from public nflverse/PFR advanced weekly data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce

import numpy as np
import pandas as pd

from nfl_game.data.nfl import PFR_STAT_TYPES
from nfl_game.data.source_manifest import SourceContractError, require_coverage

PFR_IDENTITY_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_type",
    "team",
    "opponent",
    "pfr_player_id",
)
PFR_PASS_INPUTS = (
    "passing_drops",
    "passing_bad_throws",
    "times_sacked",
    "times_blitzed",
    "times_hurried",
    "times_hit",
    "times_pressured",
    "times_pressured_pct",
    "def_times_blitzed",
    "def_times_hurried",
    "def_times_hitqb",
)

# Pinned from the live 2025 weekly schemas. Logical names describe how each real
# source field participates in aggregation; tests construct fixtures through this map.
PFR_SOURCE_CONTRACT = {
    "rush": {
        "attempts": "carries",
        "yards_before_contact": "rushing_yards_before_contact",
        "yards_after_contact": "rushing_yards_after_contact",
        "broken_tackles": "rushing_broken_tackles",
    },
    "rec": {"drops": "receiving_drop", "drop_rate": "receiving_drop_pct"},
    "def": {
        "pressure_opportunities": "def_targets",
        "pressures": "def_pressures",
        "missed_tackles": "def_missed_tackles",
        "missed_tackle_rate": "def_missed_tackle_pct",
    },
}
PFR_REQUIRED_COLUMNS = {
    "pass": (*PFR_IDENTITY_COLUMNS, *PFR_PASS_INPUTS),
    **{
        stat_type: (*PFR_IDENTITY_COLUMNS, *contract.values())
        for stat_type, contract in PFR_SOURCE_CONTRACT.items()
    },
}
PFR_FEATURE_COLS = (
    "pfr_pressure_rate",
    "pfr_hurry_rate",
    "pfr_hit_rate",
    "pfr_bad_throw_rate",
    "pfr_drop_rate",
    "pfr_sack_rate",
    "pfr_rush_ybc",
    "pfr_rush_yac",
    "pfr_broken_tackle_rate",
    "pfr_rec_drop_rate",
    "pfr_def_missed_tackle_rate",
    "pfr_def_pressure_rate",
    "pfr_imputed",
)
PFR_OUTPUT_COLS = PFR_FEATURE_COLS[:-1]
PFR_REQUIRED_NUMERIC_COLUMNS = PFR_OUTPUT_COLS

_KEY = ["season", "week", "team"]
_NORMALIZED_COLUMNS = {
    "pass": PFR_OUTPUT_COLS[:6],
    "rush": PFR_OUTPUT_COLS[6:9],
    "rec": PFR_OUTPUT_COLS[9:10],
    "def": PFR_OUTPUT_COLS[10:12],
}


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide by positive denominators only, leaving unavailable rates missing."""
    return numerator.div(denominator.where(denominator.gt(0)))


def _numeric(frame: pd.DataFrame, columns: Sequence[str], stat_type: str) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        numeric = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column].notna() & numeric.isna()
        if invalid.any() or not np.isfinite(numeric.dropna()).all():
            raise SourceContractError(
                f"non-numeric or non-finite values in PFR {stat_type} {column}"
            )
        out[column] = numeric
    return out


def _sum_by_team_week(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return rows.groupby(_KEY, dropna=True)[list(columns)].sum(min_count=1).reset_index()


def _normalize_pass(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = (
        "passing_drops",
        "passing_bad_throws",
        "times_sacked",
        "times_hurried",
        "times_hit",
        "times_pressured",
        "times_pressured_pct",
    )
    rows = _numeric(rows, numeric_columns, "pass")
    rows["_pass_opportunities"] = _safe_rate(rows["times_pressured"], rows["times_pressured_pct"])
    counts = _sum_by_team_week(rows, (*numeric_columns[:-1], "_pass_opportunities"))
    denominator = counts["_pass_opportunities"]
    counts["pfr_pressure_rate"] = _safe_rate(counts["times_pressured"], denominator)
    counts["pfr_hurry_rate"] = _safe_rate(counts["times_hurried"], denominator)
    counts["pfr_hit_rate"] = _safe_rate(counts["times_hit"], denominator)
    counts["pfr_bad_throw_rate"] = _safe_rate(counts["passing_bad_throws"], denominator)
    counts["pfr_drop_rate"] = _safe_rate(counts["passing_drops"], denominator)
    counts["pfr_sack_rate"] = _safe_rate(counts["times_sacked"], denominator)
    return counts[[*_KEY, *_NORMALIZED_COLUMNS["pass"]]]


def _normalize_rush(rows: pd.DataFrame) -> pd.DataFrame:
    contract = PFR_SOURCE_CONTRACT["rush"]
    rows = _numeric(rows, tuple(contract.values()), "rush")
    counts = _sum_by_team_week(rows, tuple(contract.values()))
    attempts = counts[contract["attempts"]]
    counts["pfr_rush_ybc"] = _safe_rate(counts[contract["yards_before_contact"]], attempts)
    counts["pfr_rush_yac"] = _safe_rate(counts[contract["yards_after_contact"]], attempts)
    counts["pfr_broken_tackle_rate"] = _safe_rate(counts[contract["broken_tackles"]], attempts)
    return counts[[*_KEY, *_NORMALIZED_COLUMNS["rush"]]]


def _normalize_rec(rows: pd.DataFrame) -> pd.DataFrame:
    contract = PFR_SOURCE_CONTRACT["rec"]
    rows = _numeric(rows, tuple(contract.values()), "rec")
    rows["_targets"] = _safe_rate(rows[contract["drops"]], rows[contract["drop_rate"]])
    counts = _sum_by_team_week(rows, (contract["drops"], "_targets"))
    counts["pfr_rec_drop_rate"] = _safe_rate(counts[contract["drops"]], counts["_targets"])
    return counts[[*_KEY, *_NORMALIZED_COLUMNS["rec"]]]


def _normalize_def(rows: pd.DataFrame) -> pd.DataFrame:
    contract = PFR_SOURCE_CONTRACT["def"]
    rows = _numeric(rows, tuple(contract.values()), "def")
    rows["_tackle_opportunities"] = _safe_rate(
        rows[contract["missed_tackles"]], rows[contract["missed_tackle_rate"]]
    )
    counts = _sum_by_team_week(
        rows,
        (
            contract["pressure_opportunities"],
            contract["pressures"],
            contract["missed_tackles"],
            "_tackle_opportunities",
        ),
    )
    counts["pfr_def_missed_tackle_rate"] = _safe_rate(
        counts[contract["missed_tackles"]], counts["_tackle_opportunities"]
    )
    counts["pfr_def_pressure_rate"] = _safe_rate(
        counts[contract["pressures"]], counts[contract["pressure_opportunities"]]
    )
    return counts[[*_KEY, *_NORMALIZED_COLUMNS["def"]]]


_NORMALIZERS = {
    "pass": _normalize_pass,
    "rush": _normalize_rush,
    "rec": _normalize_rec,
    "def": _normalize_def,
}


def normalize_pfr_frame(stat_type: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and aggregate one PFR player-level frame to one row per team-week."""
    if stat_type not in PFR_STAT_TYPES:
        raise SourceContractError(f"unknown PFR stat type: {stat_type!r}")
    missing = sorted(set(PFR_REQUIRED_COLUMNS[stat_type]).difference(frame.columns))
    if missing:
        raise SourceContractError(f"missing PFR {stat_type} columns: {missing}")
    regular = frame.loc[frame["game_type"].eq("REG")].copy()
    if regular.empty:
        return pd.DataFrame(columns=[*_KEY, *_NORMALIZED_COLUMNS[stat_type]])
    return _NORMALIZERS[stat_type](regular)


def team_week_pfr(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return count-first PFR aggregates after validating all four source frames."""
    missing = sorted(set(PFR_STAT_TYPES).difference(frames))
    if missing:
        raise SourceContractError(f"missing PFR stat types: {missing}")
    normalized = [normalize_pfr_frame(stat_type, frames[stat_type]) for stat_type in PFR_STAT_TYPES]
    out = reduce(
        lambda left, right: left.merge(right, on=_KEY, how="outer", validate="one_to_one"),
        normalized,
    )
    return out[[*_KEY, *PFR_OUTPUT_COLS]].sort_values(_KEY).reset_index(drop=True)


def _team_history(prior: pd.DataFrame, team: str, halflife: float) -> pd.DataFrame:
    history = prior.loc[prior["team"].eq(team)].sort_values(_KEY, kind="stable").tail(8).copy()
    if history.empty:
        history["_weight"] = pd.Series(dtype=float)
        return history
    ages = np.arange(len(history), 0, -1, dtype=float)
    history["_weight"] = 0.5 ** (ages / halflife)
    return history


def _weighted_average(rows: pd.DataFrame, column: str, default: float) -> float:
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
    weights = rows["_weight"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return default
    return float(np.average(values[valid], weights=weights[valid]))


def trailing_pfr_features(
    team_weeks: pd.DataFrame,
    targets: Sequence[tuple[int, int]],
    output_columns: Sequence[str],
    halflife: float,
) -> pd.DataFrame:
    """Build game-ordinal, eight-game, strictly pre-target PFR features."""
    columns = [*_KEY, *output_columns, "pfr_imputed"]
    if not targets:
        return pd.DataFrame(columns=columns)
    if halflife <= 0:
        raise ValueError("halflife must be positive")

    rows = team_weeks.copy()
    results = []
    for season, week in sorted(set(targets)):
        visible = (rows["season"] < season) | (rows["season"].eq(season) & rows["week"].le(week))
        teams = sorted(rows.loc[visible, "team"].dropna().unique())
        eligible = (rows["season"] < season) | (rows["season"].eq(season) & rows["week"].lt(week))
        prior = rows.loc[eligible].copy()
        histories = {team: _team_history(prior, team, halflife) for team in teams}
        populated = [history for history in histories.values() if not history.empty]
        league_rows = (
            pd.concat(populated, ignore_index=True)
            if populated
            else prior.assign(_weight=pd.Series(dtype=float))
        )
        league = {column: _weighted_average(league_rows, column, 0.0) for column in output_columns}
        for team in teams:
            history = histories[team]
            imputed = int(history.empty or history[list(output_columns)].isna().any(axis=None))
            results.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "team": team,
                    **{
                        column: _weighted_average(history, column, league[column])
                        for column in output_columns
                    },
                    "pfr_imputed": imputed,
                }
            )
    return pd.DataFrame(results, columns=columns)


def pfr_features_for_targets(
    team_weeks: pd.DataFrame,
    targets: Sequence[tuple[int, int]],
    halflife: float = 8.0,
) -> pd.DataFrame:
    """Return production-gated PFR features for requested season-week targets."""
    require_coverage(team_weeks, PFR_REQUIRED_NUMERIC_COLUMNS, minimum=0.90)
    return trailing_pfr_features(team_weeks, targets, PFR_OUTPUT_COLS, halflife)
