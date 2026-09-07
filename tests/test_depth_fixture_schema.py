"""Pin the hand-written depth-chart fixtures in `tests/test_qb.py` against the real feed.

This module makes NO network call. It derives the fixtures' column sets by calling the
fixture builders themselves (never a retyped constant) and compares them against the
column spellings `depth.py` actually coalesces. Step 1 of this task's brief made the one
live call this task requires and recorded the real feed's columns in the plan file; the
fixtures were corrected against that observation before this test was written.
"""

from tests.test_qb import (
    _composed_public_depth_fixture,
    _cutoff_fixture,
    _depth_history,
    _future_starter_row,
    _mixed_depth_normalized_source,
    _mixed_depth_raw_rows,
    _pre2025_era_fixture,
)

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


def _depth_history_columns() -> set[str]:
    return set(_depth_history().columns)


def _composed_public_fixture_columns() -> set[str]:
    return set(_composed_public_depth_fixture().columns)


def _mixed_depth_fixture_columns() -> set[str]:
    return set(_mixed_depth_normalized_source().columns) | set(_mixed_depth_raw_rows().columns)


def _future_starter_row_columns() -> set[str]:
    return set(_future_starter_row().columns)


def test_each_fixture_column_is_one_depth_py_actually_reads():
    """A fixture column that depth.py never coalesces is decoration, not a fixture.

    The sets come from the fixture builders themselves, so renaming a column in
    tests/test_qb.py without updating depth.py fails HERE. `_depth_history()` is
    included alongside the two era fixtures -- it predates this task, is not
    exclusive to either era (it deliberately concatenates both live shapes into one
    frame), and was never checked against the live feed until now. The three
    composed/public-schema fixtures below (Fix 4, post-launch review) were inline
    literals in tests/test_qb.py, unguarded by this drift check, until they were
    named and added here.
    """
    known = set(_TEAM_SOURCES) | set(_POSITION_SOURCES) | set(_PLAYER_SOURCES)
    known |= set(_RANK_SOURCES) | {"dt", "season", "week"}
    fixtures = (
        _timestamped_fixture_columns(),
        _labelled_fixture_columns(),
        _depth_history_columns(),
        _composed_public_fixture_columns(),
        _mixed_depth_fixture_columns(),
        _future_starter_row_columns(),
    )
    for columns in fixtures:
        assert columns <= known, sorted(columns - known)


def test_the_two_fixture_eras_are_genuinely_disjoint_on_identity():
    """If the two fixtures share a team/position spelling, one is not a separate era.

    `chart_as_of` branches on which era a row belongs to. Two fixtures that both
    spell the team `team` exercise one branch twice and leave the other unpinned.

    `gsis_id` is deliberately NOT in this set. The original draft of this test
    included it alongside `player_id`, on the assumption that player identity
    splits by era the same way team identity does. Step 1's live fetch disproves
    that: `gsis_id` is the one column the real 2019 and 2025 feeds both populate
    (every other overlap-candidate column -- club_code/team, position/pos_abb,
    depth_team/pos_rank, season/week/dt -- is genuinely era-exclusive). Keeping
    `gsis_id` in this set would force both fixtures to disagree on a column the
    live feed agrees on, which is exactly the kind of invented-schema mismatch
    this task exists to prevent, not enforce.
    """
    identity = {"club_code", "team", "pos_abb", "position"}
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


def test_each_era_uses_the_player_identity_spelling_the_live_feed_actually_carries():
    """The same swap risk as the team spelling above, for player identity.

    `_PLAYER_SOURCES` accepts both `player_id` and `gsis_id`, and both fixtures feed
    into the same coalesce, so a fixture using the wrong spelling for its era passes
    the membership and disjointness checks above and every functional test in
    test_qb.py -- neither observes which raw spelling fed the canonical `player_id`
    column downstream. Only a live inspection of the feed (recorded in this plan's
    Task 8, Step 1) shows that NEITHER era ever populates `player_id`: both the 2019
    and the 2025 feed carry `gsis_id`, so both fixtures are pinned to it here.
    """
    _, timestamped, _ = _cutoff_fixture()
    labelled, _ = _pre2025_era_fixture()
    assert "gsis_id" in timestamped.columns and "player_id" not in timestamped.columns
    assert "gsis_id" in labelled.columns and "player_id" not in labelled.columns


def test_depth_history_uses_the_player_identity_spelling_the_live_feed_actually_carries():
    """A third, independent instance of the same defect class, in a fixture that
    predates this task and is not built by `_cutoff_fixture()`/`_pre2025_era_fixture()`.

    `_depth_history()` deliberately concatenates both live shapes into one frame (its
    own docstring says so), and gets the TEAM spelling right for each block -- `club_code`
    for the week-labelled rows, `team` for the timestamped rows -- which is why it was
    usable as corroborating evidence for the other two fixtures' team-column fix. But it
    used `player_id` in both blocks, which neither live era populates. This is invisible
    to every one of the six pre-existing tests that consume it, for the same coalescing
    reason as the other two instances, and is caught only by pinning the live-observed
    spelling as a literal, exactly like the two tests above.
    """
    columns = _depth_history_columns()
    assert "gsis_id" in columns and "player_id" not in columns
