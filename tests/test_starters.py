import pandas as pd

from nfl_game.ratings.qb import qb_week_stats
from nfl_game.ratings.starters import (
    ADVISORY_COLS,
    player_display_names,
    starter_advisory,
)


def _players():
    return pd.DataFrame(
        {
            "gsis_id": ["LAMAR", "HUNT", "BURROW"],
            "display_name": ["Lamar Jackson", "Tyler Huntley", "Joe Burrow"],
        }
    )


def _fixture():
    schedules = pd.DataFrame(
        {
            "game_id": ["2025_05_CIN_BAL"],
            "season": [2025],
            "week": [5],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2025-10-05 17:00", tz="UTC")],
        }
    )
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "BAL", "CIN"],
            "gsis_id": ["HUNT", "LAMAR", "BURROW"],
            "pos_abb": ["QB", "QB", "QB"],
            "pos_rank": [1.0, 2.0, 1.0],
            "dt": pd.to_datetime(["2025-10-04 12:00"] * 3, utc=True),
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025] * 3,
            "week": [4, 4, 4],
            "team": ["BAL", "BAL", "CIN"],
            "player_id": ["LAMAR", "HUNT", "BURROW"],
            "position": ["QB"] * 3,
            "season_type": ["REG"] * 3,
            "attempts": [30.0, 0.0, 30.0],
            "sacks_suffered": [2.0, 0.0, 2.0],
            "passing_epa": [12.0, 0.0, 8.0],
            "passing_cpoe": [3.0, 0.0, 1.0],
            "passing_interceptions": [0.0, 0.0, 1.0],
        }
    )
    return qb_week_stats(stats), depth, schedules


def test_display_names_map_ids_to_names():
    assert player_display_names(_players())["HUNT"] == "Tyler Huntley"


def test_advisory_reports_one_row_per_game_with_the_expected_columns():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(weeks, depth, schedules, _players(), [(2025, 5)], cutoff=None)
    assert list(out.columns) == ADVISORY_COLS
    assert len(out) == 1


def test_advisory_names_the_changed_starter_and_flags_the_game():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(
        weeks, depth, schedules, _players(), [(2025, 5)], cutoff=None
    ).iloc[0]
    assert out["home_qb"] == "Tyler Huntley"
    assert out["away_qb"] == "Joe Burrow"
    assert out["qb_watch"] == 1
    # Huntley has no dropbacks, so he shrinks fully to the league prior, which is
    # below Lamar's own rate. The delta must therefore be negative, not zero.
    assert out["qb_change_epa_home"] < 0.0
    assert out["qb_change_epa_away"] == 0.0


def test_qb_watch_is_zero_when_neither_side_changed():
    weeks, depth, schedules = _fixture()
    unchanged = depth.copy()
    unchanged.loc[unchanged["gsis_id"].eq("HUNT"), "pos_rank"] = 2.0
    unchanged.loc[unchanged["gsis_id"].eq("LAMAR"), "pos_rank"] = 1.0
    out = starter_advisory(
        weeks, unchanged, schedules, _players(), [(2025, 5)], cutoff=None
    ).iloc[0]
    assert out["qb_watch"] == 0
    assert out["qb_inferred"] == 0


def test_missing_chart_marks_the_row_inferred_rather_than_guessing_silently():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(
        weeks, depth.iloc[0:0], schedules, _players(), [(2025, 5)], cutoff=None
    ).iloc[0]
    assert out["qb_inferred"] == 1
    # The fallback still names last week's starter -- it is labelled, not suppressed.
    assert out["home_qb"] == "Lamar Jackson"


def test_unknown_player_id_yields_a_null_name_not_the_raw_id():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(
        weeks, depth, schedules, _players().iloc[0:0], [(2025, 5)], cutoff=None
    ).iloc[0]
    assert pd.isna(out["home_qb"])


def test_qb_watch_and_qb_inferred_are_never_null_for_a_scheduled_game():
    # I2/Fix 2: qb_watch/qb_inferred used to be masked null via a `known` check on
    # whether both sides' per-team rows resolved. That mask can never be False for
    # any input the builder produces -- `per_team` is built from the same
    # schedules/targets as `games`, so every scheduled team always has a per-team
    # row, and qb_new_starter/qb_uncertain are always int, never NaN, even in the
    # most degenerate case (no depth chart AND no prior stats at all, so neither
    # side's starter resolves). This pins that degenerate case to still produce
    # concrete 0/1 values, not <NA> -- a null qb_watch/qb_inferred from this
    # function would mean the unreachable mask (or an equivalent) crept back in.
    _, depth, schedules = _fixture()
    out = starter_advisory(
        pd.DataFrame(columns=["season", "week", "team", "player_id", "dropbacks", "passing_epa", "passing_cpoe", "sacks_suffered", "passing_interceptions"]),
        depth.iloc[0:0],
        schedules,
        _players(),
        [(2025, 5)],
        cutoff=None,
    ).iloc[0]
    assert not pd.isna(out["qb_watch"])
    assert not pd.isna(out["qb_inferred"])
    assert out["qb_watch"] == 0
    assert out["qb_inferred"] == 1


def test_empty_targets_yield_an_empty_frame_with_the_full_schema():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(weeks, depth, schedules, _players(), [], cutoff=None)
    assert out.empty
    assert list(out.columns) == ADVISORY_COLS


def test_a_season_week_with_no_scheduled_games_yields_the_full_schema():
    """Covers the `per_team.empty` early return: a target season/week the schedule
    has no games for. `qb_features_for_targets`'s own games-merge (schedules inner-
    joined on the requested season/week) is empty here, so `per_team` comes back
    empty and the function returns before it ever computes its own local `games`."""
    weeks, depth, schedules = _fixture()
    out = starter_advisory(weeks, depth, schedules, _players(), [(2099, 1)], cutoff=None)
    assert out.empty
    assert list(out.columns) == ADVISORY_COLS


def test_games_empty_still_yields_the_full_schema(monkeypatch):
    """Covers the `games.empty` early return in starter_advisory itself.

    In real use this line is never reached separately from `per_team.empty` above:
    both are computed from the identical schedules-inner-joined-on-targets merge, so
    whenever one is empty so is the other, and `per_team.empty` is checked first. The
    `games.empty` check below it is defensive-only under today's code -- reachable
    only by isolating it from `per_team`, as this test does by stubbing
    qb_features_for_targets to report a per-team row for a team/week the schedule was
    never asked about."""
    weeks, depth, schedules = _fixture()
    from nfl_game.ratings import starters as starters_module

    fake_per_team = pd.DataFrame({"season": [2099], "week": [1], "team": ["ZZZ"]})
    monkeypatch.setattr(
        starters_module, "qb_features_for_targets", lambda *args, **kwargs: fake_per_team
    )
    out = starter_advisory(weeks, depth, schedules, _players(), [(2099, 1)], cutoff=None)
    assert out.empty
    assert list(out.columns) == ADVISORY_COLS
