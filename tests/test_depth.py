"""The depth-chart feed arrives as two feeds with disjoint schemas; normalize them once."""

import pandas as pd
import pytest

from nfl_game.ratings.depth import (
    DEPTH_COLUMNS,
    depth_change_rate,
    normalize_depth_charts,
    starters_as_of,
)


def week_labelled_rows():
    """The pre-2025 feed: season/week labels, team identity in `club_code`, no `dt`."""
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "club_code": "BUF",
                "gsis_id": "qb-1",
                "position": "QB",
                "depth_position": "QB",
                "depth_team": "1",
            },
            {
                "season": 2024,
                "week": 1,
                "club_code": "BUF",
                "gsis_id": "qb-2",
                "position": "QB",
                "depth_position": "QB",
                "depth_team": "2",
            },
            {
                "season": 2024,
                "week": 1,
                "club_code": "BUF",
                "gsis_id": "rb-1",
                "position": "RB",
                "depth_position": "HB",
                "depth_team": "1",
            },
            {
                "season": 2024,
                "week": 2,
                "club_code": "BUF",
                "gsis_id": "qb-2",
                "position": "QB",
                "depth_position": "QB",
                "depth_team": "1",
            },
            {
                "season": 2024,
                "week": 2,
                "club_code": "BUF",
                "gsis_id": "rb-1",
                "position": "RB",
                "depth_position": "HB",
                "depth_team": "1",
            },
        ]
    )


def timestamped_rows():
    """The 2025-era feed: daily `dt` snapshots, no season/week, slot in `pos_abb`."""
    rows = []
    for moment, starter in (("2025-09-01T10:00:00Z", "qb-1"), ("2025-09-08T10:00:00Z", "qb-2")):
        rows.append(
            {"team": "BUF", "gsis_id": starter, "pos_abb": "QB", "pos_rank": 1.0, "dt": moment}
        )
        rows.append(
            {"team": "BUF", "gsis_id": "rb-1", "pos_abb": "RB", "pos_rank": 1.0, "dt": moment}
        )
        rows.append(
            {"team": "BUF", "gsis_id": "qb-9", "pos_abb": "QB", "pos_rank": 2.0, "dt": moment}
        )
    return pd.DataFrame(rows)


def test_normalize_carries_both_eras_into_one_schema():
    out = normalize_depth_charts(
        pd.concat([week_labelled_rows(), timestamped_rows()], ignore_index=True)
    )

    assert list(out.columns) == list(DEPTH_COLUMNS)
    assert len(out) == len(week_labelled_rows()) + len(timestamped_rows())
    assert set(out["team"].unique()) == {"BUF"}
    assert out["player_id"].notna().all()
    assert out["rank"].notna().all()


def test_normalize_resolves_team_identity_from_either_column():
    """Pre-2025 identity lives in `club_code` and is null in `team`; 2025 is the reverse."""
    out = normalize_depth_charts(
        pd.concat([week_labelled_rows(), timestamped_rows()], ignore_index=True)
    )

    assert out["team"].isna().sum() == 0


def test_normalize_maps_relocated_franchise_codes_to_canonical():
    """The pre-2025 feed carries LA, OAK and SD; the 2025 feed carries LAR."""
    rows = week_labelled_rows().assign(club_code=["LA", "OAK", "SD", "LA", "OAK"])

    out = normalize_depth_charts(rows)

    assert set(out["team"].unique()) == {"LAR", "LV", "LAC"}


def test_normalize_reads_rank_from_either_era_scale():
    """`depth_team` is the string set {1,2,3}; `pos_rank` is float 1.0-12.0."""
    out = normalize_depth_charts(
        pd.concat([week_labelled_rows(), timestamped_rows()], ignore_index=True)
    )

    week_ranks = out[out["week"].notna()]["rank"]
    stamp_ranks = out[out["dt"].notna()]["rank"]
    assert sorted(week_ranks.unique()) == [1.0, 2.0]
    assert sorted(stamp_ranks.unique()) == [1.0, 2.0]


def test_starters_as_of_uses_the_week_label_when_the_feed_has_no_timestamp():
    depth = normalize_depth_charts(week_labelled_rows())
    cutoff = pd.Timestamp("2024-09-12T17:00:00Z")

    assert starters_as_of(depth, "BUF", season=2024, week=1, cutoff=cutoff) == {"qb-1", "rb-1"}
    assert starters_as_of(depth, "BUF", season=2024, week=2, cutoff=cutoff) == {"qb-2", "rb-1"}


def test_starters_as_of_takes_the_latest_snapshot_at_or_before_the_cutoff():
    depth = normalize_depth_charts(timestamped_rows())

    early = starters_as_of(
        depth, "BUF", season=2025, week=2, cutoff=pd.Timestamp("2025-09-05T00:00:00Z")
    )
    late = starters_as_of(
        depth, "BUF", season=2025, week=2, cutoff=pd.Timestamp("2025-09-14T00:00:00Z")
    )

    assert early == {"qb-1", "rb-1"}
    assert late == {"qb-2", "rb-1"}


def test_starters_as_of_never_reads_a_snapshot_published_after_the_cutoff():
    depth = normalize_depth_charts(timestamped_rows())

    assert (
        starters_as_of(
            depth, "BUF", season=2025, week=1, cutoff=pd.Timestamp("2025-08-30T00:00:00Z")
        )
        == set()
    )


def test_starters_as_of_excludes_backups():
    """rank == 1 is the only depth level that means the same thing in both eras."""
    depth = normalize_depth_charts(timestamped_rows())

    starters = starters_as_of(
        depth, "BUF", season=2025, week=2, cutoff=pd.Timestamp("2025-09-14T00:00:00Z")
    )

    assert "qb-9" not in starters


def daily_rows():
    """Daily snapshots where the starter changed a week ago, not yesterday."""
    rows = []
    for day in range(1, 15):
        moment = f"2025-09-{day:02d}T10:00:00Z"
        starter = "qb-1" if day <= 7 else "qb-2"
        rows.append(
            {"team": "BUF", "gsis_id": starter, "pos_abb": "QB", "pos_rank": 1.0, "dt": moment}
        )
        rows.append(
            {"team": "BUF", "gsis_id": "rb-1", "pos_abb": "RB", "pos_rank": 1.0, "dt": moment}
        )
    return pd.DataFrame(rows)


def test_change_rate_is_starter_set_turnover_between_weekly_charts():
    depth = normalize_depth_charts(week_labelled_rows())

    rate = depth_change_rate(
        depth, "BUF", season=2024, week=2, cutoff=pd.Timestamp("2024-09-12T17:00:00Z")
    )

    assert rate == pytest.approx(2 / 3)


def test_change_rate_is_zero_in_week_one_with_no_prior_chart():
    depth = normalize_depth_charts(week_labelled_rows())

    rate = depth_change_rate(
        depth, "BUF", season=2024, week=1, cutoff=pd.Timestamp("2024-09-05T17:00:00Z")
    )

    assert rate == pytest.approx(0.0)


def test_change_rate_compares_a_seven_day_window_not_the_last_two_snapshots():
    """The 2025 feed is daily; "the last two snapshots" would measure a one-day diff.

    The starter changed seven days before the cutoff, so a one-day comparison reports
    no change at all while the seven-day window reports the real turnover.
    """
    depth = normalize_depth_charts(daily_rows())
    cutoff = pd.Timestamp("2025-09-14T17:00:00Z")

    rate = depth_change_rate(depth, "BUF", season=2025, week=3, cutoff=cutoff)

    yesterday = starters_as_of(
        depth, "BUF", season=2025, week=3, cutoff=pd.Timestamp("2025-09-13T17:00:00Z")
    )
    today = starters_as_of(depth, "BUF", season=2025, week=3, cutoff=cutoff)
    assert yesterday == today, "fixture must be quiet day-over-day for this to discriminate"
    assert rate == pytest.approx(2 / 3)


def test_change_rate_is_zero_when_no_chart_is_available_either_side():
    depth = normalize_depth_charts(timestamped_rows())

    rate = depth_change_rate(
        depth, "BUF", season=2025, week=1, cutoff=pd.Timestamp("2025-08-01T00:00:00Z")
    )

    assert rate == pytest.approx(0.0)


def test_normalize_keeps_dt_timezone_aware_when_the_column_is_absent():
    """An absent `dt` must not broadcast a tz-naive column (the Task 13 crash)."""
    out = normalize_depth_charts(week_labelled_rows())

    assert out["dt"].dt.tz is not None


def test_normalize_drops_rows_with_no_usable_identity_or_rank():
    rows = pd.concat(
        [
            week_labelled_rows(),
            pd.DataFrame(
                [
                    {
                        "season": 2024,
                        "week": 1,
                        "club_code": "BUF",
                        "gsis_id": None,
                        "position": "QB",
                        "depth_team": "1",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "season": 2024,
                        "week": 1,
                        "club_code": "BUF",
                        "gsis_id": "x-1",
                        "position": "QB",
                        "depth_team": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    out = normalize_depth_charts(rows)

    assert len(out) == len(week_labelled_rows())


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [
        (["team", "club_code"], "team"),
        (["gsis_id", "player_id"], "player"),
        (["depth_team", "pos_rank", "rank"], "rank"),
    ],
)
def test_normalize_raises_when_a_whole_field_is_absent_from_the_feed(dropped, expected):
    """A schema break must be loud. Silently yielding NaN here is what made the feed inert.

    Every such column was dropped downstream, so the block emitted a constant and the
    90% coverage gate read 1.000000 because the constant was non-null.
    """
    rows = week_labelled_rows().drop(columns=[c for c in dropped if c in week_labelled_rows()])

    with pytest.raises(ValueError, match=expected):
        normalize_depth_charts(rows)


def test_normalize_is_idempotent_on_its_own_output():
    """Call sites pass either a raw feed or an already-normalized frame."""
    once = normalize_depth_charts(
        pd.concat([week_labelled_rows(), timestamped_rows()], ignore_index=True)
    )

    twice = normalize_depth_charts(once)

    pd.testing.assert_frame_equal(once, twice)
