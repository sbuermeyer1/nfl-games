import pandas as pd
import pytest

from nfl_game.ratings.personnel import (
    PERSONNEL_FEATURE_COLS,
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
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-a", "depth_chart_position": "QB1", "dt": "2025-09-08T12:00:00Z"},
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-b", "depth_chart_position": "RB1", "dt": "2025-09-08T12:00:00Z"},
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-d", "depth_chart_position": "QB1", "dt": "2025-09-12T12:00:00Z"},
                {"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-b", "depth_chart_position": "RB1", "dt": "2025-09-12T12:00:00Z"},
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


def test_depth_chart_change_compares_player_ids_in_each_slot():
    out = personnel_features_for_targets(**personnel_inputs()).set_index("team")
    assert out.loc["BUF", "depth_chart_change_rate"] == pytest.approx(0.5)


def test_post_cutoff_snapshots_do_not_change_features():
    inputs = personnel_inputs()
    before = personnel_features_for_targets(**inputs)
    inputs["rosters"] = pd.concat([inputs["rosters"], pd.DataFrame([{"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-x", "dt": "2025-09-15T00:00:00Z"}])], ignore_index=True)
    inputs["depth_charts"] = pd.concat([inputs["depth_charts"], pd.DataFrame([{"season": 2025, "week": 2, "team": "BUF", "gsis_id": "gsis-x", "depth_chart_position": "QB1", "dt": "2025-09-15T00:00:00Z"}])], ignore_index=True)
    after = personnel_features_for_targets(**inputs)
    pd.testing.assert_frame_equal(before, after)


def test_pre_2025_snapshots_are_eligible_only_for_the_labeled_week():
    inputs = personnel_inputs(targets=[(2024, 2)])
    inputs["rosters"] = pd.DataFrame(
        [{"season": 2024, "week": 1, "team": "BUF", "gsis_id": "gsis-a", "dt": "2024-09-01T12:00:00Z"}]
    )
    out = personnel_features_for_targets(**inputs).set_index("team")
    assert out.loc["BUF", "roster_churn"] == pytest.approx(0.0)


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


def test_pre_2025_labeled_snapshot_after_cutoff_is_not_eligible():
    inputs = personnel_inputs(targets=[(2024, 2)])
    inputs["rosters"] = pd.DataFrame(
        [{"season": 2024, "week": 2, "team": "BUF", "gsis_id": "gsis-a", "dt": "2024-09-16T12:00:00Z"}]
    )
    inputs["snaps"] = inputs["snaps"].query("season == 2024").copy()
    out = personnel_features_for_targets(**inputs).set_index("team")
    assert out.loc["BUF", "off_snap_hhi"] == pytest.approx(0.0)


def test_pre_2025_depth_snapshot_after_cutoff_does_not_change_chart():
    inputs = personnel_inputs(targets=[(2024, 2)])
    inputs["depth_charts"] = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "team": "BUF", "gsis_id": "gsis-a", "depth_chart_position": "QB1", "dt": "2024-09-08T12:00:00Z"},
            {"season": 2024, "week": 2, "team": "BUF", "gsis_id": "gsis-d", "depth_chart_position": "QB1", "dt": "2024-09-16T12:00:00Z"},
        ]
    )
    out = personnel_features_for_targets(**inputs).set_index("team")
    assert out.loc["BUF", "depth_chart_change_rate"] == pytest.approx(0.0)
