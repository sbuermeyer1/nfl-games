from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from nfl_game.data.schedule import (
    ScheduleSchemaError,
    active_prediction_weeks,
    is_final_game,
    normalize_schedule,
)

NOW = datetime(2026, 9, 10, 16, tzinfo=UTC)


def raw_schedule():
    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_LA_SF",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-09",
                "gametime": "20:20",
                "away_team": "LA",
                "home_team": "SF",
                "result": 3.0,
                "total": 47.0,
                "spread_line": 2.5,
                "total_line": 45.5,
            },
            {
                "game_id": "2026_01_BUF_NYJ",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "away_team": "BUF",
                "home_team": "NYJ",
                "result": np.nan,
                "total": np.nan,
                "spread_line": -3.0,
                "total_line": np.nan,
            },
            {
                "game_id": "2026_02_KC_DEN",
                "season": 2026,
                "game_type": "REG",
                "week": 2,
                "gameday": "2026-09-20",
                "gametime": "16:25",
                "away_team": "KC",
                "home_team": "DEN",
                "result": np.nan,
                "total": np.nan,
                "spread_line": np.nan,
                "total_line": 46.0,
            },
            {
                "game_id": "2026_03_PRE_X_Y",
                "season": 2026,
                "game_type": "PRE",
                "week": 3,
                "gameday": "2026-08-20",
                "gametime": "20:00",
                "away_team": "X",
                "home_team": "Y",
                "result": 1.0,
                "total": 30.0,
                "spread_line": 1.0,
                "total_line": 33.0,
            },
        ]
    )


def test_normalize_schedule_filters_regular_season_normalizes_teams_and_kickoff():
    out = normalize_schedule(raw_schedule(), 2026)
    assert list(out["game_id"]) == [
        "2026_01_LA_SF",
        "2026_01_BUF_NYJ",
        "2026_02_KC_DEN",
    ]
    assert out.iloc[0]["away_team"] == "LAR"
    assert str(out["kickoff_at"].dtype) == "datetime64[ns, UTC]"
    assert out["game_id"].is_unique


def test_active_prediction_weeks_returns_earliest_two_weeks_with_unplayed_games():
    out = normalize_schedule(raw_schedule(), 2026)
    assert active_prediction_weeks(out, NOW) == [1, 2]


def test_finalization_requires_scores_and_six_hours_after_kickoff():
    row = normalize_schedule(raw_schedule(), 2026).iloc[0]
    assert not is_final_game(row, datetime(2026, 9, 10, 1, tzinfo=UTC))
    assert is_final_game(row, NOW)


def test_normalize_schedule_rejects_duplicate_ids_and_invalid_lines():
    duplicate = pd.concat([raw_schedule(), raw_schedule().iloc[[0]]], ignore_index=True)
    with pytest.raises(ScheduleSchemaError, match="duplicate game_id"):
        normalize_schedule(duplicate, 2026)

    invalid = raw_schedule()
    invalid.loc[0, "spread_line"] = float("inf")
    with pytest.raises(ScheduleSchemaError, match="spread_line"):
        normalize_schedule(invalid, 2026)


def test_first_active_week_has_every_prior_week_final():
    """The vintage floor in advance_live_ledger depends on this property.

    If active_prediction_weeks ever stops returning a prefix of the unplayed weeks,
    the minimum week in the features artifact stops implying that its predecessors
    were complete at build time, and the floor silently admits stale predictions.
    """
    now = pd.Timestamp("2026-10-01T12:00:00Z")
    rows = []
    for week in range(1, 6):
        kickoff = pd.Timestamp("2026-09-06T17:00:00Z") + pd.Timedelta(weeks=week - 1)
        played = kickoff + pd.Timedelta(hours=6) <= now
        rows.append(
            {
                "game_id": f"2026_{week:02d}_AAA_BBB",
                "season": 2026,
                "week": week,
                "away_team": "AAA",
                "home_team": "BBB",
                "kickoff_at": kickoff,
                "result": 3.0 if played else np.nan,
                "total": 44.0 if played else np.nan,
            }
        )
    schedule = pd.DataFrame(rows)

    weeks = active_prediction_weeks(schedule, now)
    first = min(weeks)
    prior = schedule.loc[schedule["week"] < first]

    assert not prior.empty, "fixture must contain at least one completed week"
    assert all(is_final_game(row, now) for _, row in prior.iterrows())
