"""Game-level expected-starter advisory.

This is presentation, not a model input. Nothing here reaches `FEATURE_COLS`, and the
advisory is joined onto a slate AFTER prediction, so its presence, absence or staleness
can never move `model_margin` or `model_total`.
"""

from __future__ import annotations

import pandas as pd

from nfl_game.ratings.qb import CutoffPolicy, qb_features_for_targets

ADVISORY_COLS = [
    "game_id",
    "home_qb",
    "away_qb",
    "qb_change_epa_home",
    "qb_change_epa_away",
    "qb_watch",
    "qb_inferred",
]

_NAME_SOURCES = ("gsis_id", "player_id")
_DISPLAY_SOURCES = ("display_name", "football_name", "full_name")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype="string"),
            "home_qb": pd.Series(dtype="string"),
            "away_qb": pd.Series(dtype="string"),
            "qb_change_epa_home": pd.Series(dtype="float64"),
            "qb_change_epa_away": pd.Series(dtype="float64"),
            "qb_watch": pd.Series(dtype="Int64"),
            "qb_inferred": pd.Series(dtype="Int64"),
        }
    )


def player_display_names(players: pd.DataFrame) -> dict[str, str]:
    """Map player id to display name, tolerating the crosswalk's column spellings.

    An id with no matching row yields no entry rather than falling back to the raw id:
    a slate cell reading "00-0034796" looks like a rendering bug, while an empty cell
    correctly reads as "not resolved".
    """
    if players.empty:
        return {}
    id_col = next((name for name in _NAME_SOURCES if name in players), None)
    name_col = next((name for name in _DISPLAY_SOURCES if name in players), None)
    if id_col is None or name_col is None:
        return {}
    rows = players[[id_col, name_col]].dropna()
    return {str(pid): str(name) for pid, name in rows.itertuples(index=False)}


def starter_advisory(
    qb_weeks: pd.DataFrame,
    depth_history: pd.DataFrame,
    schedules: pd.DataFrame,
    players: pd.DataFrame,
    targets: list[tuple[int, int]],
    cutoff: CutoffPolicy = None,
) -> pd.DataFrame:
    """One advisory row per scheduled game in `targets`."""
    if not targets:
        return _empty()
    per_team = qb_features_for_targets(qb_weeks, depth_history, schedules, targets, cutoff)
    if per_team.empty:
        return _empty()

    requested = pd.DataFrame(sorted(set(targets)), columns=["season", "week"])
    games = schedules.merge(requested, on=["season", "week"], how="inner")
    games = games[["game_id", "season", "week", "home_team", "away_team"]].drop_duplicates(
        "game_id"
    )
    if games.empty:
        return _empty()

    names = player_display_names(players)
    out = games[["game_id"]].copy()
    out["game_id"] = out["game_id"].astype("string")
    for side in ("home", "away"):
        merged = games.merge(
            per_team,
            left_on=["season", "week", f"{side}_team"],
            right_on=["season", "week", "team"],
            how="left",
        )
        out[f"{side}_qb"] = (
            merged["expected_starter_id"].map(names).astype("string").to_numpy()
        )
        out[f"qb_change_epa_{side}"] = pd.to_numeric(
            merged["qb_change_epa"], errors="coerce"
        ).to_numpy()
        out[f"_new_{side}"] = pd.to_numeric(
            merged["qb_new_starter"], errors="coerce"
        ).to_numpy()
        out[f"_unc_{side}"] = pd.to_numeric(
            merged["qb_uncertain"], errors="coerce"
        ).to_numpy()

    # qb_watch/qb_inferred are never null from this function: `per_team` is built by
    # qb_features_for_targets from the SAME schedules/targets `games` comes from, so
    # every team in `games` always has a matching per-team row, and qb_new_starter /
    # qb_uncertain are always plain 0/1 ints (never NaN) even when neither side's
    # starter resolves. A null qb_watch/qb_inferred can still happen downstream, in
    # build_slate, when no `starters` frame is supplied at all or a game_id has no
    # matching advisory row -- that is a different, real "unavailable" case, not this
    # per-side one.
    out["qb_watch"] = out[["_new_home", "_new_away"]].max(axis=1).astype("Int64")
    out["qb_inferred"] = out[["_unc_home", "_unc_away"]].max(axis=1).astype("Int64")
    return out[ADVISORY_COLS].reset_index(drop=True)
