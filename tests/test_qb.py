import pandas as pd
import pytest

from nfl_game.ratings.qb import (
    QB_FEATURE_COLS,
    QB_PRIOR_DROPBACKS,
    ROOKIE_DROPBACK_LIMIT,
    normalize_depth_chart_history,
    qb_features_for_targets,
    qb_week_stats,
)


def _player_stats():
    return pd.DataFrame(
        [
            {"season": 2024, "week": 1, "team": "A", "player_id": "qb-a", "position": "QB", "attempts": 40, "sacks_suffered": 4, "passing_epa": 8.8, "passing_cpoe": 6.0, "passing_interceptions": 2},
            {"season": 2024, "week": 1, "team": "A", "player_id": "qb-b", "position": "QB", "attempts": 10, "sacks_suffered": 0, "passing_epa": -1.0, "passing_cpoe": -4.0, "passing_interceptions": 0},
            {"season": 2024, "week": 1, "team": "B", "player_id": "qb-c", "position": "QB", "attempts": 20, "sacks_suffered": 0, "passing_epa": 0.0, "passing_cpoe": 0.0, "passing_interceptions": 0},
        ]
    )


def _schedules():
    return pd.DataFrame(
        [
            {"season": 2024, "week": 1, "home_team": "A", "away_team": "B", "gameday": "2024-09-08", "gametime": "13:00"},
            {"season": 2024, "week": 2, "home_team": "A", "away_team": "B", "gameday": "2024-09-15", "gametime": "13:00", "home_qb_id": "leak", "away_qb_id": "leak"},
            {"season": 2025, "week": 3, "home_team": "A", "away_team": "B", "gameday": "2025-09-21", "gametime": "13:00"},
            {"season": 2025, "week": 4, "home_team": "A", "away_team": "B", "gameday": "2025-09-28", "gametime": "13:00"},
        ]
    )


def _depth_history():
    return pd.DataFrame(
        [
            {"season": 2024, "week": 2, "team": "A", "position": "QB", "depth_chart_position": 1, "player_id": "qb-a"},
            {"season": 2024, "week": 2, "team": "A", "position": "QB", "depth_chart_position": 2, "player_id": "qb-b"},
            {"season": 2024, "week": 2, "team": "B", "position": "QB", "depth_chart_position": 1, "player_id": "qb-c"},
            {"season": 2025, "week": 3, "team": "A", "position": "QB", "depth_chart_position": 1, "player_id": "qb-a", "dt": "2025-09-20T12:00:00Z"},
            {"season": 2025, "week": 3, "team": "B", "position": "QB", "depth_chart_position": 1, "player_id": "qb-c", "dt": "2025-09-20T12:00:00Z"},
        ]
    )


def test_qb_week_stats_uses_attempts_plus_sacks_as_dropbacks():
    out = qb_week_stats(_player_stats()).set_index("player_id")
    assert out.loc["qb-a", "dropbacks"] == 44
    assert out.loc["qb-a", "epa_per_db"] == pytest.approx(8.8 / 44)


def test_qb_week_stats_excludes_postseason_and_nonpositive_weeks():
    stats = pd.concat(
        [
            _player_stats().assign(season_type="REG"),
            pd.DataFrame(
                [
                    {"season": 2024, "week": 0, "team": "A", "player_id": "week-zero", "position": "QB", "season_type": "REG", "attempts": 30, "sacks_suffered": 0, "passing_epa": 30, "passing_cpoe": 0, "passing_interceptions": 0},
                    {"season": 2024, "week": 19, "team": "A", "player_id": "post-qb", "position": "QB", "season_type": "POST", "attempts": 30, "sacks_suffered": 0, "passing_epa": 30, "passing_cpoe": 0, "passing_interceptions": 0},
                ]
            ),
        ],
        ignore_index=True,
    )
    out = qb_week_stats(stats)
    assert set(out["player_id"]) == {"qb-a", "qb-b", "qb-c"}
    assert (out["week"] > 0).all()


def test_small_sample_rates_shrink_toward_league():
    weeks = qb_week_stats(_player_stats())
    out = qb_features_for_targets(weeks, _depth_history(), _schedules(), [(2024, 2)])
    league_rate = 2 / 74
    assert league_rate < out.set_index("team").loc["A", "qb_int_rate"] < 2 / 44
    assert QB_PRIOR_DROPBACKS == 200


def test_pre_2025_rank_one_chart_wins_and_schedule_qb_ids_are_not_starters():
    weeks = qb_week_stats(_player_stats())
    out = qb_features_for_targets(weeks, _depth_history(), _schedules(), [(2024, 2)])
    a = out.set_index("team").loc["A"]
    assert a["expected_starter_id"] == "qb-a"
    assert a["qb_uncertain"] == 0


def test_missing_chart_falls_back_to_prior_game_most_used_qb_and_marks_uncertain():
    weeks = qb_week_stats(_player_stats())
    depth = _depth_history().query("not (season == 2024 and week == 2 and team == 'A')")
    out = qb_features_for_targets(weeks, depth, _schedules(), [(2024, 2)])
    a = out.set_index("team").loc["A"]
    assert a["expected_starter_id"] == "qb-a"
    assert a["qb_uncertain"] == 1


def test_qb_change_rookie_and_new_starter_use_strictly_prior_dropbacks():
    weeks = qb_week_stats(_player_stats())
    depth = _depth_history().copy()
    depth.loc[(depth["season"] == 2024) & (depth["week"] == 2) & (depth["team"] == "A") & (depth["depth_chart_position"] == 1), "player_id"] = "qb-b"
    out = qb_features_for_targets(weeks, depth, _schedules(), [(2024, 2)])
    a = out.set_index("team").loc["A"]
    assert a["qb_new_starter"] == 1
    assert a["qb_rookie"] == 1
    assert a["qb_change_epa"] == pytest.approx((-1 / 10) - (8.8 / 44))
    assert ROOKIE_DROPBACK_LIMIT == 100


def test_2025_depth_history_uses_timestamp_as_of_target_kickoff():
    weeks = qb_week_stats(_player_stats())
    depth = _depth_history()
    depth.loc[(depth["season"] == 2025) & (depth["week"] == 3) & (depth["team"] == "A"), "player_id"] = "qb-b"
    history = normalize_depth_chart_history(depth, _schedules())
    out = qb_features_for_targets(weeks, history, _schedules(), [(2025, 4)])
    assert out.set_index("team").loc["A", "expected_starter_id"] == "qb-b"


def test_normalized_depth_history_keeps_rank_one_starter_when_composed_publicly():
    weeks = qb_week_stats(_player_stats())
    depth = pd.DataFrame(
        [
            {"season": 2025, "week": 3, "team": "A", "position": "QB", "depth_chart_position": 1, "player_id": "z-starter", "dt": "2025-09-20T12:00:00Z"},
            {"season": 2025, "week": 3, "team": "A", "position": "QB", "depth_chart_position": 2, "player_id": "a-backup", "dt": "2025-09-20T12:00:00Z"},
            {"season": 2025, "week": 3, "team": "B", "position": "QB", "depth_chart_position": 1, "player_id": "qb-c", "dt": "2025-09-20T12:00:00Z"},
        ]
    )
    history = normalize_depth_chart_history(depth, _schedules())
    out = qb_features_for_targets(weeks, history, _schedules(), [(2025, 4)])
    assert out.set_index("team").loc["A", "expected_starter_id"] == "z-starter"


def test_mixed_normalized_and_raw_depth_history_coalesces_per_row_ranks():
    weeks = qb_week_stats(_player_stats())
    normalized = normalize_depth_chart_history(
        pd.DataFrame(
            [{"season": 2025, "week": 3, "team": "A", "position": "QB", "depth_chart_position": 1, "player_id": "z-starter", "dt": "2025-09-20T12:00:00Z"}]
        ),
        _schedules(),
    )
    raw = pd.DataFrame(
        [
            {"season": 2025, "week": 3, "team": "A", "position": "QB", "depth_chart_position": 2, "player_id": "a-backup", "dt": "2025-09-20T12:00:00Z"},
            {"season": 2025, "week": 3, "team": "B", "position": "QB", "depth_chart_position": 1, "player_id": "qb-c", "dt": "2025-09-20T12:00:00Z"},
        ]
    )
    history = normalize_depth_chart_history(pd.concat([normalized, raw], ignore_index=True), _schedules())
    a_rows = history[history["team"].eq("A")].sort_values("rank")
    assert a_rows[["player_id", "rank"]].values.tolist() == [["z-starter", 1], ["a-backup", 2]]
    assert a_rows["dt"].tolist() == [pd.Timestamp("2025-09-20T12:00:00Z")] * 2
    out = qb_features_for_targets(weeks, history, _schedules(), [(2025, 4)])
    assert out.set_index("team").loc["A", "expected_starter_id"] == "z-starter"


def test_future_depth_snapshot_cannot_change_expected_starter():
    weeks = qb_week_stats(_player_stats())
    before = qb_features_for_targets(weeks, _depth_history(), _schedules(), [(2025, 4)])
    future = pd.concat([_depth_history(), pd.DataFrame([{"season": 2025, "week": 4, "team": "A", "position": "QB", "depth_chart_position": 1, "player_id": "qb-z", "dt": "2025-10-01T12:00:00Z"}])], ignore_index=True)
    after = qb_features_for_targets(weeks, future, _schedules(), [(2025, 4)])
    pd.testing.assert_frame_equal(before, after)


def test_empty_history_returns_documented_numeric_feature_schema():
    out = qb_features_for_targets(pd.DataFrame(), pd.DataFrame(), _schedules(), [(2024, 2)])
    assert list(out.columns) == ["season", "week", "team", "expected_starter_id", *QB_FEATURE_COLS]
    assert set(out["team"]) == {"A", "B"}
