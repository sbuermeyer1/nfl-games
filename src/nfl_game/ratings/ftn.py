"""FTN charting aggregates for the research-only Ridge-v2 E1 candidate.

FTN charting begins in 2022, which is too short a history for the production candidate, so
nothing here feeds a production artifact. Two properties of the live feed drive this module:

* **There is no team column.** FTN rows carry `nflverse_game_id` and `nflverse_play_id` only,
  so offensive attribution requires a join to play-by-play for `posteam`. A charted play that
  does not join is dropped rather than guessed at.
* **`date_pulled` is not an availability time.** It is the archive's snapshot timestamp -- 2022
  rows carry 2024 pull dates -- so the as-of rule comes from week ordering, exactly as the other
  blocks do, and never from that column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FTN_FEATURE_COLS = (
    "ftn_motion_rate",
    "ftn_play_action_rate",
    "ftn_rpo_rate",
    "ftn_screen_rate",
    "ftn_out_of_pocket_rate",
    "ftn_int_worthy_rate",
    "ftn_catchable_rate",
    "ftn_drop_rate",
    "ftn_blitzers_mean",
    "ftn_pass_rushers_mean",
    "ftn_qb_fault_sack_rate",
    "ftn_imputed",
)

# Every column the aggregation reads. An absent one raises: these all vanish downstream, and
# tolerating absence is what turns a schema break into a block that reports full coverage
# while emitting a constant.
FTN_REQUIRED_COLUMNS = (
    "nflverse_game_id",
    "nflverse_play_id",
    "season",
    "week",
    "is_motion",
    "is_play_action",
    "is_rpo",
    "is_screen_pass",
    "is_qb_out_of_pocket",
    "is_interception_worthy",
    "is_catchable_ball",
    "is_drop",
    "is_qb_fault_sack",
    "n_blitzers",
    "n_pass_rushers",
)

PBP_REQUIRED_COLUMNS = ("game_id", "play_id", "posteam", "season", "week")

_TEAM_GAME_COLS = (
    "game_id",
    "season",
    "week",
    "team",
    "n_charted_plays",
    *(name for name in FTN_FEATURE_COLS if name != "ftn_imputed"),
)

# League fallbacks for a team with no prior charted history. Each is flagged with
# ftn_imputed=1 so a fill can never masquerade as a measurement.
_EMPTY_DEFAULTS: dict[str, float] = {
    "ftn_motion_rate": 0.0,
    "ftn_play_action_rate": 0.0,
    "ftn_rpo_rate": 0.0,
    "ftn_screen_rate": 0.0,
    "ftn_out_of_pocket_rate": 0.0,
    "ftn_int_worthy_rate": 0.0,
    "ftn_catchable_rate": 0.0,
    "ftn_drop_rate": 0.0,
    "ftn_blitzers_mean": 0.0,
    "ftn_pass_rushers_mean": 0.0,
    "ftn_qb_fault_sack_rate": 0.0,
}

_RATE_VALUES = tuple(_EMPTY_DEFAULTS)


def _boolean(rows: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a charted flag to boolean.

    `is_trick_play` arrives as object dtype on the live feed, so nothing here may assume the
    declared bool dtype holds for every flag.
    """
    values = rows[column]
    if values.dtype == bool:
        return values
    return values.map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def team_game_ftn(ftn: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Reduce charted plays to one offensive row per team-game.

    The FTN feed has no team, so `posteam` comes from play-by-play. Charted plays that do not
    join are dropped: a play whose offense is unknown cannot be attributed to either side.
    """
    missing = sorted(set(FTN_REQUIRED_COLUMNS).difference(ftn.columns))
    if missing:
        raise ValueError(f"FTN charting frame is missing required column(s) {missing}")
    missing_pbp = sorted(set(PBP_REQUIRED_COLUMNS).difference(pbp.columns))
    if missing_pbp:
        raise ValueError(f"play-by-play frame is missing required column(s) {missing_pbp}")
    if ftn.empty or pbp.empty:
        return pd.DataFrame(columns=list(_TEAM_GAME_COLS))

    plays = pbp.loc[pbp["posteam"].notna(), ["game_id", "play_id", "posteam"]].copy()
    plays["play_id"] = _numeric(plays, "play_id")
    charted = ftn.copy()
    charted["nflverse_play_id"] = _numeric(charted, "nflverse_play_id")
    joined = charted.merge(
        plays,
        left_on=["nflverse_game_id", "nflverse_play_id"],
        right_on=["game_id", "play_id"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        return pd.DataFrame(columns=list(_TEAM_GAME_COLS))

    flags = {
        "ftn_motion_rate": "is_motion",
        "ftn_play_action_rate": "is_play_action",
        "ftn_rpo_rate": "is_rpo",
        "ftn_screen_rate": "is_screen_pass",
        "ftn_out_of_pocket_rate": "is_qb_out_of_pocket",
        "ftn_int_worthy_rate": "is_interception_worthy",
        "ftn_catchable_rate": "is_catchable_ball",
        "ftn_drop_rate": "is_drop",
        "ftn_qb_fault_sack_rate": "is_qb_fault_sack",
    }
    for name, column in flags.items():
        joined[name] = _boolean(joined, column).astype(float)
    joined["ftn_blitzers_mean"] = _numeric(joined, "n_blitzers")
    joined["ftn_pass_rushers_mean"] = _numeric(joined, "n_pass_rushers")

    grouped = joined.groupby(["game_id", "season", "week", "posteam"], dropna=True)
    aggregated = grouped[list(_RATE_VALUES)].mean().reset_index()
    aggregated["n_charted_plays"] = grouped.size().to_numpy()
    aggregated = aggregated.rename(columns={"posteam": "team"})
    return aggregated[list(_TEAM_GAME_COLS)].sort_values(
        ["season", "week", "team"], kind="stable", ignore_index=True
    )


def _team_history(prior: pd.DataFrame, team: str, halflife: float) -> pd.DataFrame:
    history = prior[prior["team"].eq(team)]
    if history.empty:
        return history.assign(_weight=pd.Series(dtype=float))
    ordering = ["season", "week"] + (["game_id"] if "game_id" in history else [])
    history = history.sort_values(ordering, kind="stable").tail(8).copy()
    ages = np.arange(len(history), 0, -1, dtype=float)
    history["_weight"] = 0.5 ** (ages / halflife)
    return history


def _weighted_average(rows: pd.DataFrame, column: str, default: float) -> float:
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
    weights = rows["_weight"].to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return default
    return float(np.average(values[valid], weights=weights[valid]))


def ftn_features_for_targets(
    team_games: pd.DataFrame,
    targets: list[tuple[int, int]],
    halflife: float = 8.0,
) -> pd.DataFrame:
    """Eight-game, strictly pre-target FTN features for every team seen up to each target.

    Strictly pre-target is the whole point: a team's row for (season, week) is built only from
    games played before that week, so a later result can never move an earlier feature.
    """
    columns = ["season", "week", "team", *FTN_FEATURE_COLS]
    if team_games.empty or not targets:
        return pd.DataFrame(columns=columns)
    required = {"season", "week", "team"}
    if not required.issubset(team_games.columns):
        raise ValueError(f"FTN team-game frame is missing required column(s) {sorted(required)}")

    games = team_games.copy()
    results: list[dict[str, object]] = []
    for season, week in sorted(set(targets)):
        seen = (games["season"] < season) | (games["season"].eq(season) & games["week"].le(week))
        teams = sorted(games.loc[seen, "team"].dropna().unique())
        prior_mask = (games["season"] < season) | (
            games["season"].eq(season) & games["week"].lt(week)
        )
        prior = games.loc[prior_mask]
        histories = {
            team: _team_history(prior, team, halflife) for team in prior["team"].dropna().unique()
        }
        pooled = (
            pd.concat([frame for frame in histories.values() if not frame.empty])
            if any(not frame.empty for frame in histories.values())
            else pd.DataFrame(columns=[*games.columns, "_weight"])
        )
        league = {
            name: (
                _weighted_average(pooled, name, _EMPTY_DEFAULTS[name])
                if not pooled.empty
                else _EMPTY_DEFAULTS[name]
            )
            for name in _RATE_VALUES
        }
        for team in teams:
            history = histories.get(team)
            imputed = history is None or history.empty
            row: dict[str, object] = {"season": season, "week": week, "team": team}
            for name in _RATE_VALUES:
                row[name] = (
                    league[name] if imputed else _weighted_average(history, name, league[name])
                )
            row["ftn_imputed"] = 1.0 if imputed else 0.0
            results.append(row)
    return pd.DataFrame(results, columns=columns)


def ftn_game_features(games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away FTN features to game rows as margin diffs and total sums."""
    required = {"game_id", "season", "week", "home_team", "away_team"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise ValueError(f"game frame is missing required column(s) {missing}")
    value_columns = [name for name in FTN_FEATURE_COLS if name != "ftn_imputed"]
    output_columns = (
        ["game_id"]
        + [f"{name}_diff" for name in value_columns]
        + [f"{name}_sum" for name in value_columns]
        + ["ftn_imputed_any"]
    )
    if games.empty or team_features.empty:
        return pd.DataFrame(columns=output_columns)

    keys = ["season", "week", "team"]
    home = team_features.rename(columns={name: f"home_{name}" for name in FTN_FEATURE_COLS})
    away = team_features.rename(columns={name: f"away_{name}" for name in FTN_FEATURE_COLS})
    merged = games.merge(
        home, left_on=["season", "week", "home_team"], right_on=keys, how="left"
    ).merge(
        away,
        left_on=["season", "week", "away_team"],
        right_on=keys,
        how="left",
        suffixes=("", "_away_key"),
    )
    out = pd.DataFrame({"game_id": merged["game_id"].to_numpy()})
    for name in value_columns:
        home_values = pd.to_numeric(merged[f"home_{name}"], errors="coerce")
        away_values = pd.to_numeric(merged[f"away_{name}"], errors="coerce")
        out[f"{name}_diff"] = (home_values - away_values).to_numpy(dtype=float)
        out[f"{name}_sum"] = (home_values + away_values).to_numpy(dtype=float)
    home_flag = pd.to_numeric(merged["home_ftn_imputed"], errors="coerce")
    away_flag = pd.to_numeric(merged["away_ftn_imputed"], errors="coerce")
    # A missing join is itself an imputation: the game had no FTN feature on that side.
    out["ftn_imputed_any"] = (
        ((home_flag.fillna(1.0) > 0) | (away_flag.fillna(1.0) > 0)).astype(float).to_numpy()
    )
    return out[output_columns]
