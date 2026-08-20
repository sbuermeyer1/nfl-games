import pandas as pd
import pytest

from nfl_game.ratings.personnel import (
    PERSONNEL_FEATURE_COLS,
    _prepared_rosters,
    normalize_snap_counts,
    personnel_features_for_targets,
    player_id_map,
)


def personnel_inputs(targets=None):
    return {
        "snaps": pd.DataFrame(
            [
                {"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "pfr-a", "offense_snaps": 50, "defense_snaps": 0},
                {"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "pfr-b", "offense_snaps": 25, "defense_snaps": 0},
                {"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "pfr-c", "offense_snaps": 25, "defense_snaps": 0},
                {"season": 2025, "week": 1, "team": "BUF", "pfr_player_id": "pfr-a", "offense_snaps": 50, "defense_snaps": 50},
                {"season": 2025, "week": 1, "team": "BUF", "pfr_player_id": "pfr-b", "offense_snaps": 30, "defense_snaps": 30},
                {"season": 2025, "week": 1, "team": "BUF", "pfr_player_id": "pfr-d", "offense_snaps": 20, "defense_snaps": 20},
            ]
        ),
        "rosters": pd.DataFrame(
            [
                {"season": 2025, "week": 1, "team": "BUF", "gsis_id": "gsis-a", "dt": "2025-09-06T12:00:00Z"},
                {"season": 2025, "week": 1, "team": "BUF", "gsis_id": "gsis-b", "dt": "2025-09-06T12:00:00Z"},
                {"season": 2025, "week": 1, "team": "BUF", "gsis_id": "gsis-d", "dt": "2025-09-06T12:00:00Z"},
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-a", "dt": "2025-09-13T12:00:00Z"},
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-b", "dt": "2025-09-13T12:00:00Z"},
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-d", "dt": "2025-09-13T12:00:00Z"},
            ]
        ),
        "depth_charts": pd.DataFrame(
            [
                {"team": "BUF", "gsis_id": "gsis-a", "pos_abb": "QB", "pos_rank": 1.0, "dt": "2025-09-05T12:00:00Z"},
                {"team": "BUF", "gsis_id": "gsis-b", "pos_abb": "RB", "pos_rank": 1.0, "dt": "2025-09-05T12:00:00Z"},
                {"team": "BUF", "gsis_id": "gsis-d", "pos_abb": "QB", "pos_rank": 1.0, "dt": "2025-09-12T12:00:00Z"},
                {"team": "BUF", "gsis_id": "gsis-b", "pos_abb": "RB", "pos_rank": 1.0, "dt": "2025-09-12T12:00:00Z"},
            ]
        ),
        "players": pd.DataFrame(
            [
                {"pfr_id": "pfr-a", "gsis_id": "gsis-a"},
                {"pfr_id": "pfr-b", "gsis_id": "gsis-b"},
                {"pfr_id": "pfr-c", "gsis_id": "gsis-c"},
                {"pfr_id": "pfr-d", "gsis_id": "gsis-d"},
            ]
        ),
        "schedules": pd.DataFrame(
            [
                {"season": 2025, "week": 1, "home_team": "BUF", "away_team": "MIA", "kickoff_at": "2025-09-07T17:00:00Z"},
                {"season": 2025, "week": 2, "home_team": "BUF", "away_team": "MIA", "kickoff_at": "2025-09-14T17:00:00Z"},
                {"season": 2024, "week": 2, "home_team": "BUF", "away_team": "MIA", "kickoff_at": "2024-09-15T17:00:00Z"},
            ]
        ),
        "targets": [(2025, 2)] if targets is None else targets,
    }


def test_player_id_map_uses_pfr_and_gsis_identifiers_not_names():
    mapping = player_id_map(personnel_inputs()["players"])
    assert mapping.to_dict("records") == [
        {"pfr_player_id": "pfr-a", "player_id": "gsis-a"},
        {"pfr_player_id": "pfr-b", "player_id": "gsis-b"},
        {"pfr_player_id": "pfr-c", "player_id": "gsis-c"},
        {"pfr_player_id": "pfr-d", "player_id": "gsis-d"},
    ]


def test_returning_snap_share_uses_current_roster_and_prior_season_snaps():
    out = personnel_features_for_targets(**personnel_inputs(targets=[(2025, 1)])).set_index("team")
    assert out.loc["BUF", "off_returning_share"] == pytest.approx(0.75)
    assert out.loc["BUF", "roster_churn"] == pytest.approx(0.25)


def test_snap_hhi_is_sum_of_squared_player_shares():
    out = personnel_features_for_targets(**personnel_inputs()).set_index("team")
    assert out.loc["BUF", "off_snap_hhi"] == pytest.approx(0.5**2 + 0.3**2 + 0.2**2)


def test_depth_chart_change_is_starter_set_turnover_over_seven_days():
    """QB1 moved gsis-a -> gsis-d a week before kickoff; RB1 held."""
    out = personnel_features_for_targets(**personnel_inputs()).set_index("team")
    assert out.loc["BUF", "depth_chart_change_rate"] == pytest.approx(2 / 3)


def test_low_identifier_coverage_is_imputed_without_counting_unmapped_as_departure():
    inputs = personnel_inputs(targets=[(2025, 1)])
    inputs["snaps"] = inputs["snaps"].query("not (season == 2024 and pfr_player_id == 'pfr-c')").copy()
    inputs["snaps"] = pd.concat([inputs["snaps"], pd.DataFrame([{"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "unmapped", "offense_snaps": 25, "defense_snaps": 0}])], ignore_index=True)
    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]
    assert out["id_coverage"] == pytest.approx(0.75)
    assert out["personnel_imputed"] == 1
    assert out["off_returning_share"] == pytest.approx(1.0)


def test_normalize_snap_counts_preserves_only_mapped_gsis_identity():
    inputs = personnel_inputs()
    out = normalize_snap_counts(inputs["snaps"], inputs["players"])
    assert set(out["player_id"]) == {"gsis-a", "gsis-b", "gsis-c", "gsis-d"}
    assert list(PERSONNEL_FEATURE_COLS) == [
        "off_returning_share", "def_returning_share", "off_snap_hhi", "def_snap_hhi",
        "depth_chart_change_rate", "roster_churn", "id_coverage", "personnel_imputed",
    ]


def test_pre_2025_depth_change_compares_the_two_most_recent_weekly_charts():
    """The older feed is week-labelled with no timestamp, so week W-1 is the anchor."""
    inputs = personnel_inputs(targets=[(2024, 2)])
    inputs["depth_charts"] = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "club_code": "BUF", "gsis_id": "gsis-a", "position": "QB", "depth_team": "1"},
            {"season": 2024, "week": 1, "club_code": "BUF", "gsis_id": "gsis-b", "position": "RB", "depth_team": "1"},
            {"season": 2024, "week": 2, "club_code": "BUF", "gsis_id": "gsis-d", "position": "QB", "depth_team": "1"},
            {"season": 2024, "week": 2, "club_code": "BUF", "gsis_id": "gsis-b", "position": "RB", "depth_team": "1"},
        ]
    )
    out = personnel_features_for_targets(**inputs).set_index("team")
    assert out.loc["BUF", "depth_chart_change_rate"] == pytest.approx(2 / 3)


def test_zero_mapped_prior_offense_is_neutral_churn_not_departure():
    inputs = personnel_inputs(targets=[(2025, 1)])
    inputs["snaps"] = inputs["snaps"].query("season != 2024").copy()
    inputs["snaps"] = pd.concat(
        [
            inputs["snaps"],
            pd.DataFrame(
                [{"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "unknown", "offense_snaps": 100, "defense_snaps": 0}]
            ),
        ],
        ignore_index=True,
    )
    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]
    assert out["off_returning_share"] == pytest.approx(0.0)
    assert out["roster_churn"] == pytest.approx(0.0)
    assert out["personnel_imputed"] == 1


def test_identifier_coverage_combines_offense_and_defense_snap_mass():
    inputs = personnel_inputs(targets=[(2025, 1)])
    inputs["snaps"] = pd.DataFrame(
        [
            {"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "pfr-a", "offense_snaps": 80, "defense_snaps": 100},
            {"season": 2024, "week": 18, "team": "BUF", "pfr_player_id": "unknown", "offense_snaps": 20, "defense_snaps": 0},
        ]
    )
    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]
    assert out["id_coverage"] == pytest.approx(0.9)
    assert out["personnel_imputed"] == 0


def test_pre_2025_depth_change_excludes_backups_from_the_starter_set():
    """Only rank 1 counts; the older feed ranks {1,2,3} and the newer one 1-12."""
    inputs = personnel_inputs(targets=[(2024, 2)])
    inputs["depth_charts"] = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "club_code": "BUF", "gsis_id": "gsis-a", "position": "QB", "depth_team": "1"},
            {"season": 2024, "week": 1, "club_code": "BUF", "gsis_id": "gsis-d", "position": "QB", "depth_team": "2"},
            {"season": 2024, "week": 2, "club_code": "BUF", "gsis_id": "gsis-a", "position": "QB", "depth_team": "1"},
            {"season": 2024, "week": 2, "club_code": "BUF", "gsis_id": "gsis-b", "position": "QB", "depth_team": "2"},
        ]
    )
    out = personnel_features_for_targets(**inputs).set_index("team")
    assert out.loc["BUF", "depth_chart_change_rate"] == pytest.approx(0.0)


def test_post_cutoff_roster_snapshot_does_not_change_week_one_continuity():
    inputs = personnel_inputs(targets=[(2025, 1)])
    before = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]
    assert before["off_returning_share"] == pytest.approx(0.75)
    assert before["roster_churn"] == pytest.approx(0.25)
    inputs["rosters"] = pd.concat(
        [
            inputs["rosters"],
            pd.DataFrame(
                [{"season": 2025, "week": 1, "team": "BUF", "gsis_id": "gsis-x", "dt": "2025-09-08T12:00:00Z"}]
            ),
        ],
        ignore_index=True,
    )
    after = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]
    pd.testing.assert_series_equal(before, after)


def test_pre_2025_roster_ignores_off_week_snapshot_for_week_one_continuity():
    inputs = personnel_inputs(targets=[(2024, 1)])
    inputs["schedules"] = pd.DataFrame(
        [{"season": 2024, "week": 1, "home_team": "BUF", "away_team": "MIA", "kickoff_at": "2024-09-08T17:00:00Z"}]
    )
    inputs["snaps"] = pd.DataFrame(
        [
            {"season": 2023, "week": 18, "team": "BUF", "pfr_player_id": "pfr-a", "offense_snaps": 75, "defense_snaps": 0},
            {"season": 2023, "week": 18, "team": "BUF", "pfr_player_id": "pfr-c", "offense_snaps": 25, "defense_snaps": 0},
        ]
    )
    inputs["rosters"] = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "team": "BUF", "gsis_id": "gsis-a", "dt": "2024-09-01T12:00:00Z"},
            {"season": 2024, "week": 2, "team": "BUF", "gsis_id": "gsis-x", "dt": "2024-09-02T12:00:00Z"},
        ]
    )
    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]
    assert out["off_returning_share"] == pytest.approx(0.75)
    assert out["roster_churn"] == pytest.approx(0.25)


def test_absent_roster_dt_column_stays_timezone_aware():
    """Live rosters_weekly carries no `dt` at all; an absent column must not go tz-naive."""
    rosters = personnel_inputs()["rosters"].drop(columns=["dt"])

    prepared = _prepared_rosters(rosters)

    assert prepared["dt"].dt.tz is not None


def test_personnel_features_build_when_neither_source_carries_dt():
    """The cutoff comparison must not raise on the real 2016-2024 source schema."""
    kwargs = personnel_inputs(targets=[(2025, 1)])
    kwargs["rosters"] = kwargs["rosters"].drop(columns=["dt"])
    kwargs["depth_charts"] = kwargs["depth_charts"].drop(columns=["dt"])

    out = personnel_features_for_targets(**kwargs)

    assert sorted(out["team"]) == ["BUF", "MIA"]
    assert set(PERSONNEL_FEATURE_COLS).issubset(out.columns)


def test_2025_roster_without_dt_column_uses_the_week_labelled_snapshot():
    """rosters_weekly has never carried `dt`, so a season-keyed timestamp rule empties it.

    Measured live: every 2025 roster snapshot returned 0 rows from a 46,849-row feed,
    which drove week-one returning shares to 0.0 and churn to 1.0 for every team while
    C4 coverage still read 1.000000.
    """
    inputs = personnel_inputs(targets=[(2025, 1)])
    inputs["rosters"] = inputs["rosters"].drop(columns=["dt"])

    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]

    assert out["off_returning_share"] == pytest.approx(0.75)
    assert out["roster_churn"] == pytest.approx(0.25)


def test_unlabelled_timestamped_season_uses_latest_pre_kickoff_snapshot():
    """A feed with timestamps and no week label is addressed by its latest snapshot.

    This is the live shape of the 2025-era depth-chart feed: 554,215 rows carrying `dt`
    with `season` and `week` both null.
    """
    inputs = personnel_inputs(targets=[(2025, 1)])
    inputs["rosters"] = pd.DataFrame(
        [
            {"season": 2025, "week": None, "team": "BUF", "gsis_id": "gsis-a", "dt": "2025-09-06T12:00:00Z"},
            {"season": 2025, "week": None, "team": "BUF", "gsis_id": "gsis-b", "dt": "2025-09-06T12:00:00Z"},
            {"season": 2025, "week": None, "team": "BUF", "gsis_id": "gsis-d", "dt": "2025-09-06T12:00:00Z"},
            {"season": 2025, "week": None, "team": "BUF", "gsis_id": "gsis-a", "dt": "2025-09-07T00:00:00Z"},
        ]
    )

    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]

    assert out["off_returning_share"] == pytest.approx(0.5)
    assert out["roster_churn"] == pytest.approx(0.5)


def test_roster_availability_rule_is_decided_per_season_not_per_frame():
    """A later timestamped season must not impose its rule on an untimestamped one."""
    inputs = personnel_inputs(targets=[(2024, 1)])
    inputs["schedules"] = pd.DataFrame(
        [{"season": 2024, "week": 1, "home_team": "BUF", "away_team": "MIA", "kickoff_at": "2024-09-08T17:00:00Z"}]
    )
    inputs["snaps"] = pd.DataFrame(
        [
            {"season": 2023, "week": 18, "team": "BUF", "pfr_player_id": "pfr-a", "offense_snaps": 75, "defense_snaps": 0},
            {"season": 2023, "week": 18, "team": "BUF", "pfr_player_id": "pfr-c", "offense_snaps": 25, "defense_snaps": 0},
        ]
    )
    inputs["rosters"] = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "team": "BUF", "gsis_id": "gsis-a", "dt": None},
            {"season": 2025, "week": None, "team": "BUF", "gsis_id": "gsis-b", "dt": "2025-09-06T12:00:00Z"},
        ]
    )

    out = personnel_features_for_targets(**inputs).set_index("team").loc["BUF"]

    assert out["off_returning_share"] == pytest.approx(0.75)
