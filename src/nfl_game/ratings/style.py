"""Leak-free team style, turnover, field-position, and special-teams features."""

from __future__ import annotations

import numpy as np
import pandas as pd

STYLE_FEATURE_COLS = (
    "neutral_pass_rate",
    "pace_seconds",
    "turnover_rate",
    "explosive_play_rate",
    "starting_field_position",
    "special_teams_epa",
    "style_imputed",
)
TURNOVER_PRIOR_PLAYS = 200

_KEY = ["game_id", "season", "week", "team"]
_GAME_COLS = _KEY + [
    "opponent",
    "is_home",
    "neutral_pass_rate",
    "pace_seconds",
    "turnover_rate",
    "explosive_play_rate",
    "starting_field_position",
    "special_teams_epa",
    "n_turnovers",
    "n_scrimmage_plays",
]
_STYLE_VALUES = list(STYLE_FEATURE_COLS[:-1])
_EMPTY_DEFAULTS = {
    "neutral_pass_rate": 0.0,
    "pace_seconds": 0.0,
    "turnover_rate": 0.0,
    "explosive_play_rate": 0.0,
    "starting_field_position": 0.0,
    "special_teams_epa": 0.0,
}


def _indicator(rows: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean nflverse indicator, safely defaulting absent columns to false."""
    if column not in rows:
        return pd.Series(False, index=rows.index)
    return rows[column].eq(1)


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _ordered_drive(drive: pd.DataFrame) -> pd.DataFrame:
    """Put a drive into deterministic chronological order, independent of source row order."""
    ordered = drive.copy()
    play_id = _numeric(ordered, "play_id")
    if play_id.notna().all() and play_id.is_unique:
        return ordered.assign(_play_order=play_id).sort_values("_play_order", kind="stable")

    canonical_columns = sorted(ordered.columns)
    canonical = ordered[canonical_columns].apply(
        lambda row: "\x1f".join(str(value) for value in row), axis=1
    )
    return ordered.assign(_partial_play_id=play_id, _canonical_order=canonical).sort_values(
        ["_qtr", "_seconds", "_partial_play_id", "_canonical_order"],
        ascending=[True, False, True, True],
        na_position="last",
        kind="stable",
    )


def team_game_style(pbp: pd.DataFrame) -> pd.DataFrame:
    """Reduce regular-season play-by-play to one offensive style row per team-game."""
    if pbp.empty:
        return pd.DataFrame(columns=_GAME_COLS)
    rows = pbp.copy()
    required = {"season_type", "posteam", "game_id", "season", "week"}
    if not required.issubset(rows.columns):
        return pd.DataFrame(columns=_GAME_COLS)
    rows = rows[rows["season_type"].eq("REG") & rows["posteam"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(columns=_GAME_COLS)

    rows["_pass"] = _indicator(rows, "pass")
    rows["_rush"] = _indicator(rows, "rush")
    rows["_scrimmage"] = rows["_pass"] | rows["_rush"]
    rows["_dropback"] = _indicator(rows, "qb_dropback")
    if "qb_dropback" not in rows:
        rows["_dropback"] = rows["_pass"]
    rows["_yards"] = _numeric(rows, "yards_gained")
    rows["_turnovers"] = (
        _indicator(rows, "interception") | _indicator(rows, "fumble_lost")
    ).astype(int)
    rows["_special_teams"] = _indicator(rows, "special_teams_play")
    rows["_epa"] = _numeric(rows, "epa")
    rows["_yardline"] = _numeric(rows, "yardline_100")
    rows["_seconds"] = _numeric(rows, "game_seconds_remaining")
    rows["_qtr"] = _numeric(rows, "qtr")
    score_column = next(
        (
            column
            for column in ("score_differential", "posteam_score_differential")
            if column in rows
        ),
        "score_differential",
    )
    rows["_score_diff"] = _numeric(rows, score_column)

    output = []
    for key, game in rows.groupby(["game_id", "season", "week", "posteam"], dropna=True):
        scrimmage = game[game["_scrimmage"]]
        neutral = scrimmage[scrimmage["_qtr"].between(1, 3) & scrimmage["_score_diff"].abs().le(8)]
        neutral_denominator = int(neutral["_dropback"].sum() + neutral["_rush"].sum())
        neutral_pass_rate = (
            float(neutral["_dropback"].sum()) / neutral_denominator
            if neutral_denominator
            else np.nan
        )
        n_scrimmage = len(scrimmage)
        n_turnovers = int(scrimmage["_turnovers"].sum())

        pace_values = []
        if "drive" in scrimmage:
            paced = scrimmage[scrimmage["drive"].notna()].copy()
            for _, drive in paced.groupby("drive", dropna=True):
                seconds = _ordered_drive(drive)["_seconds"].dropna().to_numpy(dtype=float)
                deltas = seconds[:-1] - seconds[1:]
                pace_values.extend(deltas[(deltas > 0) & (deltas <= 60)])
        pace = float(np.median(pace_values)) if pace_values else np.nan

        explosive = (scrimmage["_pass"] & scrimmage["_yards"].ge(20)) | (
            scrimmage["_rush"] & scrimmage["_yards"].ge(10)
        )
        starting_positions = []
        if "drive" in scrimmage:
            starts = scrimmage[scrimmage["drive"].notna() & scrimmage["_yardline"].notna()]
            for _, drive in starts.groupby("drive", dropna=True):
                first_play = _ordered_drive(drive).iloc[0]
                starting_positions.append(100.0 - float(first_play["_yardline"]))

        special_teams = game.loc[game["_special_teams"], "_epa"].dropna()
        opponent = (
            game["defteam"].dropna().iloc[0]
            if "defteam" in game and game["defteam"].notna().any()
            else np.nan
        )
        home_team = (
            game["home_team"].dropna().iloc[0]
            if "home_team" in game and game["home_team"].notna().any()
            else np.nan
        )
        output.append(
            {
                "game_id": key[0],
                "season": key[1],
                "week": key[2],
                "team": key[3],
                "opponent": opponent,
                "is_home": int(key[3] == home_team) if pd.notna(home_team) else np.nan,
                "neutral_pass_rate": neutral_pass_rate,
                "pace_seconds": pace,
                "turnover_rate": n_turnovers / n_scrimmage if n_scrimmage else np.nan,
                "explosive_play_rate": float(explosive.mean()) if n_scrimmage else np.nan,
                "starting_field_position": float(np.mean(starting_positions))
                if starting_positions
                else np.nan,
                "special_teams_epa": float(special_teams.mean())
                if not special_teams.empty
                else np.nan,
                "n_turnovers": n_turnovers,
                "n_scrimmage_plays": n_scrimmage,
            }
        )
    return pd.DataFrame(output, columns=_GAME_COLS).sort_values(_KEY).reset_index(drop=True)


def _weighted_average(
    rows: pd.DataFrame, weights: np.ndarray, column: str, default: float
) -> float:
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return default
    return float(np.average(values[valid], weights=weights[valid]))


def _team_history(prior: pd.DataFrame, team: str, halflife: float) -> pd.DataFrame:
    """Return a team's eight latest eligible games with game-ordinal exponential weights."""
    history = prior[prior["team"].eq(team)].copy()
    if history.empty:
        return history.assign(_weight=pd.Series(dtype=float))
    ordering = ["season", "week"] + (["game_id"] if "game_id" in history else [])
    history = history.sort_values(ordering, kind="stable").tail(8).copy()
    ages = np.arange(len(history), 0, -1, dtype=float)
    history["_weight"] = 0.5 ** (ages / halflife)
    return history


def style_features_for_targets(
    team_games: pd.DataFrame, targets: list[tuple[int, int]], halflife: float = 8.0
) -> pd.DataFrame:
    """Return eight-game, game-ordinal, strictly pre-target style features for each team."""
    columns = ["season", "week", "team", "raw_turnover_rate", *STYLE_FEATURE_COLS]
    if team_games.empty or not targets:
        return pd.DataFrame(columns=columns)
    games = team_games.copy()
    required = {"season", "week", "team"}
    if not required.issubset(games.columns):
        return pd.DataFrame(columns=columns)
    results = []
    for season, week in sorted(set(targets)):
        is_before_or_target = (games["season"] < season) | (
            games["season"].eq(season) & games["week"].le(week)
        )
        teams = sorted(games.loc[is_before_or_target, "team"].dropna().unique())
        prior_mask = (games["season"] < season) | (
            games["season"].eq(season) & games["week"].lt(week)
        )
        prior = games.loc[prior_mask].copy()
        histories = {
            team: _team_history(prior, team, halflife) for team in prior["team"].dropna().unique()
        }
        league_rows = pd.concat(histories.values(), ignore_index=True) if histories else prior
        league_weights = league_rows.get("_weight", pd.Series(dtype=float)).to_numpy(dtype=float)
        league = {
            name: _weighted_average(league_rows, league_weights, name, _EMPTY_DEFAULTS[name])
            for name in _STYLE_VALUES
            if name != "turnover_rate"
        }
        if {"n_turnovers", "n_scrimmage_plays"}.issubset(league_rows.columns):
            turnover_counts = (
                pd.to_numeric(league_rows["n_turnovers"], errors="coerce").fillna(0).to_numpy()
            )
            play_counts = (
                pd.to_numeric(league_rows["n_scrimmage_plays"], errors="coerce")
                .fillna(0)
                .to_numpy()
            )
            weighted_plays = float(np.dot(league_weights, play_counts))
            league_turnover = (
                float(np.dot(league_weights, turnover_counts) / weighted_plays)
                if weighted_plays
                else 0.0
            )
        else:
            league_turnover = _weighted_average(league_rows, league_weights, "turnover_rate", 0.0)
        for team in teams:
            history = histories.get(team, _team_history(prior, team, halflife))
            team_weights = history.get("_weight", pd.Series(dtype=float)).to_numpy(dtype=float)
            imputed = int(history.empty)
            values = {
                name: _weighted_average(history, team_weights, name, league[name])
                if not history.empty
                else league[name]
                for name in league
            }
            if not history.empty and {"n_turnovers", "n_scrimmage_plays"}.issubset(history.columns):
                turnovers = float(
                    np.dot(
                        team_weights,
                        pd.to_numeric(history["n_turnovers"], errors="coerce").fillna(0),
                    )
                )
                plays = float(
                    np.dot(
                        team_weights,
                        pd.to_numeric(history["n_scrimmage_plays"], errors="coerce").fillna(0),
                    )
                )
                raw_turnover = turnovers / plays if plays else league_turnover
                turnover_rate = (turnovers + TURNOVER_PRIOR_PLAYS * league_turnover) / (
                    plays + TURNOVER_PRIOR_PLAYS
                )
            else:
                raw_turnover = league_turnover
                turnover_rate = league_turnover
            results.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "team": team,
                    "raw_turnover_rate": raw_turnover,
                    **values,
                    "turnover_rate": turnover_rate,
                    "style_imputed": imputed,
                }
            )
    return pd.DataFrame(results, columns=columns)
