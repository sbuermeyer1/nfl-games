"""One normalization for the depth-chart feed, which arrives as two disjoint schemas.

`load_depth_charts` concatenates two sources that share almost no columns, and every
consumer that assumed one shape silently saw nothing of the other:

    column          pre-2025 (332,174 rows)   2025-era (554,215 rows)
    season, week    present                   all null
    team            all null (`club_code`)    present
    position        present                   all null (`pos_abb`)
    depth_team      present, {'1','2','3'}    all null (`pos_rank`, 1.0-12.0)
    dt              all null                  present, daily snapshots

Consumers must take the canonical frame from here rather than reach into the raw feed.
"""

from __future__ import annotations

import pandas as pd

from nfl_game.data.teams import normalize_team_codes

DEPTH_COLUMNS = ("season", "week", "team", "player_id", "position", "rank", "dt")

_TEAM_SOURCES = ("team", "club_code")
_POSITION_SOURCES = ("position", "pos_abb")
_PLAYER_SOURCES = ("player_id", "gsis_id")
_RANK_SOURCES = ("rank", "depth_team", "pos_rank")

#: Both eras agree on the elapsed window a change is measured over. The 2025-era feed
#: publishes DAILY snapshots while the older one publishes one chart per week, so
#: comparing "the last two snapshots" would measure a one-day diff against a seven-day
#: one, with the discontinuity landing inside the evaluation window.
CHANGE_WINDOW = pd.Timedelta(days=7)


def _coalesce_optional(frame: pd.DataFrame, sources: tuple[str, ...]) -> pd.Series:
    """Coalesce a field whose total absence is a legitimate state, not a schema break.

    Only `position` qualifies: a frame that has already been normalized and filtered to
    one position no longer carries it, and re-normalizing such a frame must not fail.
    Absence is tolerated; a feed that carries the column but leaves it null for an
    entire era is not, because the coalesce falls through to the era that does carry it.
    """
    if not any(name in frame for name in sources):
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    return _coalesce(frame, sources, "position")


def _coalesce(frame: pd.DataFrame, sources: tuple[str, ...], field: str) -> pd.Series:
    """Take the first non-null value across the era-specific spellings of one field.

    An entirely absent field RAISES rather than yielding nulls: every null is dropped
    downstream, so tolerating it turns a schema break into a block that emits a constant
    and still reports 1.000000 coverage, because the constant is non-null.
    """
    present = [name for name in sources if name in frame]
    if not present:
        raise ValueError(f"depth charts carry no {field} column; expected one of {sources}")
    out = frame[present[0]].astype("object")
    for name in present[1:]:
        out = out.where(out.notna(), frame[name].astype("object"))
    return out


def normalize_depth_charts(depth_charts: pd.DataFrame) -> pd.DataFrame:
    """Return both depth-chart eras in one canonical schema.

    `season`/`week` are populated only for the week-labelled era and `dt` only for the
    timestamped era; that split is the era marker every consumer keys on.
    """
    if depth_charts.empty:
        return pd.DataFrame(columns=list(DEPTH_COLUMNS))
    rows = depth_charts.copy()
    out = pd.DataFrame(index=rows.index)
    for name in ("season", "week"):
        out[name] = pd.to_numeric(rows[name], errors="coerce") if name in rows else pd.NA
    out["team"] = _coalesce(rows, _TEAM_SOURCES, "team")
    out["player_id"] = _coalesce(rows, _PLAYER_SOURCES, "player identity")
    out["position"] = _coalesce_optional(rows, _POSITION_SOURCES)
    out["rank"] = pd.to_numeric(_coalesce(rows, _RANK_SOURCES, "rank"), errors="coerce")
    out["dt"] = (
        pd.to_datetime(rows["dt"], utc=True, errors="coerce")
        if "dt" in rows
        else pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    )
    out = out.dropna(subset=["team", "player_id", "rank"])
    out = normalize_team_codes(out, ["team"])
    return out[list(DEPTH_COLUMNS)].reset_index(drop=True)


def chart_as_of(
    depth: pd.DataFrame, team: str, season: int, week: int, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Select one team's effective chart under its own era's availability rule."""
    team_rows = depth[depth["team"].eq(team)]
    timestamped = team_rows[team_rows["dt"].notna() & (team_rows["dt"] <= cutoff)]
    if not timestamped.empty:
        return timestamped[timestamped["dt"] == timestamped["dt"].max()]
    labelled = team_rows[team_rows["season"].eq(season) & team_rows["week"].eq(week)]
    return labelled[labelled["dt"].isna()]


def starters_as_of(
    depth: pd.DataFrame, team: str, season: int, week: int, cutoff: pd.Timestamp
) -> set[str]:
    """Return the player IDs listed first at their position as of the cutoff.

    `rank == 1` is the only depth level that means the same thing in both eras: the
    older feed ranks {1,2,3} while the newer one ranks 1-12.
    """
    chart = chart_as_of(depth, team, season, week, cutoff)
    return set(chart.loc[chart["rank"].eq(1.0), "player_id"])


def _previous_starters(
    depth: pd.DataFrame, team: str, season: int, week: int, cutoff: pd.Timestamp
) -> set[str] | None:
    """Return the starters one CHANGE_WINDOW earlier, or None if no prior chart exists.

    The distinction matters: treating an ABSENT prior chart as an empty starter set
    scores every week-one game as complete turnover, which is a constant masquerading
    as a measurement.
    """
    team_rows = depth[depth["team"].eq(team)]
    if (team_rows["dt"].notna() & (team_rows["dt"] <= cutoff)).any():
        earlier = cutoff - CHANGE_WINDOW
        if not (team_rows["dt"].notna() & (team_rows["dt"] <= earlier)).any():
            return None
        return starters_as_of(depth, team, season, week, earlier)
    if week <= 1:
        return None
    prior = team_rows[team_rows["season"].eq(season) & team_rows["week"].eq(week - 1)]
    if prior.empty:
        return None
    return starters_as_of(depth, team, season, week - 1, cutoff)


def depth_change_rate(
    depth: pd.DataFrame, team: str, season: int, week: int, cutoff: pd.Timestamp
) -> float:
    """Fraction of a team's projected starters that changed over the last seven days.

    Keyed on the starter SET rather than on a slot, because the two eras do not share a
    slot vocabulary: only 67.63% of pre-2025 rows carry a `depth_position` the 2025 feed
    also uses, and the newer feed is side-specific (LCB/RCB) where the older one is
    often generic (CB), so a slot-keyed rate would score a side swap as a change.
    """
    previous = _previous_starters(depth, team, season, week, cutoff)
    if previous is None:
        return 0.0
    current = starters_as_of(depth, team, season, week, cutoff)
    union = current | previous
    if not union:
        return 0.0
    return float(len(current ^ previous) / len(union))
