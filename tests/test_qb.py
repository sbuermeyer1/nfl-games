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
            {"season": 2024, "week": 1, "team": "BUF", "player_id": "qb-a", "position": "QB", "attempts": 40, "sacks_suffered": 4, "passing_epa": 8.8, "passing_cpoe": 6.0, "passing_interceptions": 2},
            {"season": 2024, "week": 1, "team": "BUF", "player_id": "qb-b", "position": "QB", "attempts": 10, "sacks_suffered": 0, "passing_epa": -1.0, "passing_cpoe": -4.0, "passing_interceptions": 0},
            {"season": 2024, "week": 1, "team": "MIA", "player_id": "qb-c", "position": "QB", "attempts": 20, "sacks_suffered": 0, "passing_epa": 0.0, "passing_cpoe": 0.0, "passing_interceptions": 0},
        ]
    )


def _schedules():
    return pd.DataFrame(
        [
            {"season": 2024, "week": 1, "home_team": "BUF", "away_team": "MIA", "gameday": "2024-09-08", "gametime": "13:00"},
            {"season": 2024, "week": 2, "home_team": "BUF", "away_team": "MIA", "gameday": "2024-09-15", "gametime": "13:00", "home_qb_id": "leak", "away_qb_id": "leak"},
            {"season": 2025, "week": 3, "home_team": "BUF", "away_team": "MIA", "gameday": "2025-09-21", "gametime": "13:00"},
            {"season": 2025, "week": 4, "home_team": "BUF", "away_team": "MIA", "gameday": "2025-09-28", "gametime": "13:00"},
        ]
    )


def _depth_history():
    """Both live shapes: the pre-2025 feed keys identity to `club_code` and rank to
    `depth_team`; the 2025-era feed keys them to `team` and `pos_rank` and carries no
    season or week at all."""
    week_labelled = pd.DataFrame(
        [
            {"season": 2024, "week": 2, "club_code": "BUF", "position": "QB", "depth_team": "1", "player_id": "qb-a"},
            {"season": 2024, "week": 2, "club_code": "BUF", "position": "QB", "depth_team": "2", "player_id": "qb-b"},
            {"season": 2024, "week": 2, "club_code": "MIA", "position": "QB", "depth_team": "1", "player_id": "qb-c"},
        ]
    )
    timestamped = pd.DataFrame(
        [
            {"team": "BUF", "pos_abb": "QB", "pos_rank": 1.0, "player_id": "qb-a", "dt": "2025-09-20T12:00:00Z"},
            {"team": "MIA", "pos_abb": "QB", "pos_rank": 1.0, "player_id": "qb-c", "dt": "2025-09-20T12:00:00Z"},
        ]
    )
    return pd.concat([week_labelled, timestamped], ignore_index=True)


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
                    {"season": 2024, "week": 0, "team": "BUF", "player_id": "week-zero", "position": "QB", "season_type": "REG", "attempts": 30, "sacks_suffered": 0, "passing_epa": 30, "passing_cpoe": 0, "passing_interceptions": 0},
                    {"season": 2024, "week": 19, "team": "BUF", "player_id": "post-qb", "position": "QB", "season_type": "POST", "attempts": 30, "sacks_suffered": 0, "passing_epa": 30, "passing_cpoe": 0, "passing_interceptions": 0},
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
    assert league_rate < out.set_index("team").loc["BUF", "qb_int_rate"] < 2 / 44
    assert QB_PRIOR_DROPBACKS == 200


def test_pre_2025_rank_one_chart_wins_and_schedule_qb_ids_are_not_starters():
    weeks = qb_week_stats(_player_stats())
    out = qb_features_for_targets(weeks, _depth_history(), _schedules(), [(2024, 2)])
    a = out.set_index("team").loc["BUF"]
    assert a["expected_starter_id"] == "qb-a"
    assert a["qb_uncertain"] == 0


def test_missing_chart_falls_back_to_prior_game_most_used_qb_and_marks_uncertain():
    weeks = qb_week_stats(_player_stats())
    depth = _depth_history().query("club_code != 'BUF' or club_code.isna()", engine="python")
    out = qb_features_for_targets(weeks, depth, _schedules(), [(2024, 2)])
    a = out.set_index("team").loc["BUF"]
    assert a["expected_starter_id"] == "qb-a"
    assert a["qb_uncertain"] == 1


def test_qb_change_rookie_and_new_starter_use_strictly_prior_dropbacks():
    weeks = qb_week_stats(_player_stats())
    depth = _depth_history().copy()
    depth.loc[depth["club_code"].eq("BUF") & depth["depth_team"].eq("1"), "player_id"] = "qb-b"
    out = qb_features_for_targets(weeks, depth, _schedules(), [(2024, 2)])
    a = out.set_index("team").loc["BUF"]
    assert a["qb_new_starter"] == 1
    assert a["qb_rookie"] == 1
    assert a["qb_change_epa"] == pytest.approx((-1 / 10) - (8.8 / 44))
    assert ROOKIE_DROPBACK_LIMIT == 100


def test_2025_depth_history_uses_timestamp_as_of_target_kickoff():
    weeks = qb_week_stats(_player_stats())
    depth = _depth_history()
    depth.loc[depth["team"].eq("BUF") & depth["dt"].notna(), "player_id"] = "qb-b"
    history = normalize_depth_chart_history(depth, _schedules())
    out = qb_features_for_targets(weeks, history, _schedules(), [(2025, 4)])
    assert out.set_index("team").loc["BUF", "expected_starter_id"] == "qb-b"


def test_normalized_depth_history_keeps_rank_one_starter_when_composed_publicly():
    weeks = qb_week_stats(_player_stats())
    depth = pd.DataFrame(
        [
            {"team": "BUF", "pos_abb": "QB", "pos_rank": 1.0, "player_id": "z-starter", "dt": "2025-09-20T12:00:00Z"},
            {"team": "BUF", "pos_abb": "QB", "pos_rank": 2.0, "player_id": "a-backup", "dt": "2025-09-20T12:00:00Z"},
            {"team": "MIA", "pos_abb": "QB", "pos_rank": 1.0, "player_id": "qb-c", "dt": "2025-09-20T12:00:00Z"},
        ]
    )
    history = normalize_depth_chart_history(depth, _schedules())
    out = qb_features_for_targets(weeks, history, _schedules(), [(2025, 4)])
    assert out.set_index("team").loc["BUF", "expected_starter_id"] == "z-starter"


def test_mixed_normalized_and_raw_depth_history_coalesces_per_row_ranks():
    weeks = qb_week_stats(_player_stats())
    normalized = normalize_depth_chart_history(
        pd.DataFrame(
            [{"team": "BUF", "pos_abb": "QB", "pos_rank": 1.0, "player_id": "z-starter", "dt": "2025-09-20T12:00:00Z"}]
        ),
        _schedules(),
    )
    raw = pd.DataFrame(
        [
            {"team": "BUF", "pos_abb": "QB", "pos_rank": 2.0, "player_id": "a-backup", "dt": "2025-09-20T12:00:00Z"},
            {"team": "MIA", "pos_abb": "QB", "pos_rank": 1.0, "player_id": "qb-c", "dt": "2025-09-20T12:00:00Z"},
        ]
    )
    history = normalize_depth_chart_history(pd.concat([normalized, raw], ignore_index=True), _schedules())
    a_rows = history[history["team"].eq("BUF")].sort_values("rank")
    assert a_rows[["player_id", "rank"]].values.tolist() == [["z-starter", 1], ["a-backup", 2]]
    assert a_rows["dt"].tolist() == [pd.Timestamp("2025-09-20T12:00:00Z")] * 2
    out = qb_features_for_targets(weeks, history, _schedules(), [(2025, 4)])
    assert out.set_index("team").loc["BUF", "expected_starter_id"] == "z-starter"


def test_future_depth_snapshot_cannot_change_expected_starter():
    weeks = qb_week_stats(_player_stats())
    before = qb_features_for_targets(weeks, _depth_history(), _schedules(), [(2025, 4)])
    future = pd.concat([_depth_history(), pd.DataFrame([{"season": 2025, "week": 4, "team": "BUF", "position": "QB", "depth_chart_position": 1, "player_id": "qb-z", "dt": "2025-10-01T12:00:00Z"}])], ignore_index=True)
    after = qb_features_for_targets(weeks, future, _schedules(), [(2025, 4)])
    pd.testing.assert_frame_equal(before, after)


def test_empty_history_returns_documented_numeric_feature_schema():
    out = qb_features_for_targets(pd.DataFrame(), pd.DataFrame(), _schedules(), [(2024, 2)])
    assert list(out.columns) == ["season", "week", "team", "expected_starter_id", *QB_FEATURE_COLS]
    assert set(out["team"]) == {"BUF", "MIA"}


def _cutoff_fixture():
    """A team whose QB1 changes between Tuesday and Sunday kickoff."""
    schedules = pd.DataFrame(
        {
            "season": [2025],
            "week": [5],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2025-10-05 17:00", tz="UTC")],
        }
    )
    # 2025-era feed: daily `dt` snapshots, `pos_rank`, `club_code`, `pos_abb`.
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "BAL", "BAL", "BAL", "CIN", "CIN"],
            "gsis_id": ["LAMAR", "HUNT", "LAMAR", "HUNT", "BURROW", "BURROW"],
            "pos_abb": ["QB", "QB", "QB", "QB", "QB", "QB"],
            "pos_rank": [1.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            "dt": pd.to_datetime(
                [
                    "2025-09-30 12:00",  # Tuesday: Lamar QB1
                    "2025-09-30 12:00",
                    "2025-10-04 12:00",  # Saturday: Huntley QB1
                    "2025-10-04 12:00",
                    "2025-09-30 12:00",
                    "2025-10-04 12:00",
                ],
                utc=True,
            ),
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025] * 3,
            "week": [4, 4, 4],
            "team": ["BAL", "BAL", "CIN"],
            "player_id": ["LAMAR", "HUNT", "BURROW"],
            "position": ["QB", "QB", "QB"],
            "season_type": ["REG"] * 3,
            "attempts": [30.0, 0.0, 30.0],
            "sacks_suffered": [2.0, 0.0, 2.0],
            "passing_epa": [12.0, 0.0, 8.0],
            "passing_cpoe": [3.0, 0.0, 1.0],
            "passing_interceptions": [0.0, 0.0, 1.0],
        }
    )
    return stats, depth, schedules


def test_cutoff_at_kickoff_sees_the_late_change():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats), depth, schedules, [(2025, 5)], cutoff=None
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "HUNT"


def test_cutoff_at_publication_lead_sees_the_earlier_chart():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats),
        depth,
        schedules,
        [(2025, 5)],
        cutoff=pd.Timedelta(days=5),
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "LAMAR"


def test_cutoff_accepts_an_absolute_instant():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats),
        depth,
        schedules,
        [(2025, 5)],
        cutoff=pd.Timestamp("2025-10-04 18:00", tz="UTC"),
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "HUNT"


def test_naive_absolute_cutoff_raises():
    stats, depth, schedules = _cutoff_fixture()
    with pytest.raises(ValueError, match="timezone-aware"):
        qb_features_for_targets(
            qb_week_stats(stats),
            depth,
            schedules,
            [(2025, 5)],
            cutoff=pd.Timestamp("2025-10-04 18:00"),
        )


def _pre2025_era_fixture():
    """The OTHER feed era: week-labelled charts, no `dt`, `team`/`position`/`depth_team`.

    The two eras share almost no columns, and `chart_as_of` takes a different branch for
    each -- the timestamped branch filters on `dt <= cutoff`, the labelled branch matches
    season/week and cannot see a cutoff at all. A test that exercises only the 2025-era
    feed leaves the entire labelled branch unpinned.
    """
    _, _, schedules = _cutoff_fixture()
    depth = pd.DataFrame(
        {
            "season": [2019, 2019, 2019, 2019],
            "week": [5, 5, 5, 5],
            "team": ["BAL", "BAL", "CIN", "CIN"],
            "player_id": ["LAMAR", "HUNT", "BURROW", "OTHER"],
            "position": ["QB", "QB", "QB", "QB"],
            "depth_team": ["1", "2", "1", "2"],
        }
    )
    schedules = schedules.assign(
        season=2019, kickoff_at=[pd.Timestamp("2019-10-06 17:00", tz="UTC")]
    )
    return depth, schedules


def test_labelled_era_resolves_a_starter_at_both_cutoffs():
    stats, _, _ = _cutoff_fixture()
    depth, schedules = _pre2025_era_fixture()
    weeks = qb_week_stats(stats.assign(season=2019, week=4))
    for cutoff in (None, pd.Timedelta(days=5)):
        out = qb_features_for_targets(
            weeks, depth, schedules, [(2019, 5)], cutoff=cutoff
        ).set_index("team")
        # The labelled era carries no timestamp, so the chart is the same at both
        # cutoffs -- but it must still RESOLVE rather than falling through to null.
        assert out.loc["BAL", "expected_starter_id"] == "LAMAR", cutoff
        assert out.loc["BAL", "qb_uncertain"] == 0, cutoff
