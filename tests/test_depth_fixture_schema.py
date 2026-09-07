"""Pin the hand-written depth-chart fixtures in `tests/test_qb.py` against the real feed.

This module makes NO network call. It derives the fixtures' column sets by calling the
fixture builders themselves (never a retyped constant) and compares them against the
column spellings `depth.py` actually coalesces. Step 1 of this task's brief made the one
live call this task requires and recorded the real feed's columns in the plan file; the
fixtures were corrected against that observation before this test was written.
"""

from tests.test_qb import _cutoff_fixture, _pre2025_era_fixture

from nfl_game.ratings.depth import (
    _PLAYER_SOURCES,
    _POSITION_SOURCES,
    _RANK_SOURCES,
    _TEAM_SOURCES,
)


def _timestamped_fixture_columns() -> set[str]:
    _, depth, _ = _cutoff_fixture()
    return set(depth.columns)


def _labelled_fixture_columns() -> set[str]:
    depth, _ = _pre2025_era_fixture()
    return set(depth.columns)


def test_each_fixture_column_is_one_depth_py_actually_reads():
    """A fixture column that depth.py never coalesces is decoration, not a fixture.

    The sets come from the fixture builders themselves, so renaming a column in
    tests/test_qb.py without updating depth.py fails HERE.
    """
    known = set(_TEAM_SOURCES) | set(_POSITION_SOURCES) | set(_PLAYER_SOURCES)
    known |= set(_RANK_SOURCES) | {"dt", "season", "week"}
    for columns in (_timestamped_fixture_columns(), _labelled_fixture_columns()):
        assert columns <= known, sorted(columns - known)


def test_the_two_fixture_eras_are_genuinely_disjoint_on_identity():
    """If the two fixtures share an identity spelling, one is not a separate era.

    `chart_as_of` branches on which era a row belongs to. Two fixtures that both
    spell the team `team` exercise one branch twice and leave the other unpinned.
    """
    identity = {"club_code", "team", "gsis_id", "player_id", "pos_abb", "position"}
    shared = _timestamped_fixture_columns() & _labelled_fixture_columns() & identity
    assert not shared, sorted(shared)


def test_the_fixtures_are_not_accidentally_identical():
    """A guard on the guard: if someone makes both fixtures the same, say so.

    Without this, pointing both helpers at one builder would satisfy every
    assertion above by making the intersection trivially empty of identity columns.
    """
    assert _timestamped_fixture_columns() != _labelled_fixture_columns()


def test_each_era_uses_the_team_spelling_the_live_feed_actually_carries():
    """Membership and disjointness both pass even if the two eras are SWAPPED.

    `_TEAM_SOURCES` accepts both `team` and `club_code`, and the two fixtures never
    collide on either -- so a fixture pointed at the wrong era's spelling (2025-era
    rows carrying `club_code`, pre-2025 rows carrying `team`) passes every other
    assertion in this file, and every functional test in test_qb.py, without ever
    touching a network call. Only a live inspection of the feed (recorded in this
    plan's Task 8, Step 1) can catch that swap, so it is pinned here as a literal:
    live 2025-era depth charts populate `team`, never `club_code`; live pre-2025
    charts populate `club_code`, never `team`.
    """
    _, timestamped, _ = _cutoff_fixture()
    labelled, _ = _pre2025_era_fixture()
    assert "team" in timestamped.columns and "club_code" not in timestamped.columns
    assert "club_code" in labelled.columns and "team" not in labelled.columns
