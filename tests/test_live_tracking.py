import numpy as np
import pandas as pd
import pytest

from nfl_game.tracking.ledger import LEDGER_COLUMNS
from nfl_game.tracking.live import LiveTrackerLifecycleError, advance_live_ledger

NOW = pd.Timestamp("2026-09-01T12:00:00Z")
GAME_ID = "2026_01_AAA_BBB"


def empty_live_ledger():
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def advance(existing, schedule, predictions, now, *, first_publishable_week=1, **kwargs):
    """Existing tests all use week-1 fixtures, so the floor defaults to 1 here.

    The production signature has no default on purpose; this default lives in the test
    file only, so the floor tests below must pass `first_publishable_week` explicitly.
    """
    return advance_live_ledger(
        existing,
        schedule,
        predictions,
        now,
        first_publishable_week=first_publishable_week,
        **kwargs,
    )


def schedule_fixture(
    kickoff,
    *,
    spread=3.0,
    total=44.0,
    result=np.nan,
    actual_total=np.nan,
    game_id=GAME_ID,
    week=1,
):
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "away_team": "AAA",
                "home_team": "BBB",
                "kickoff_at": pd.Timestamp(kickoff),
                "spread_line": spread,
                "total_line": total,
                "result": result,
                "total": actual_total,
            }
        ]
    )


def predictions_fixture(*, model_margin=6.0, model_total=47.0, game_id=GAME_ID):
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "model_margin": model_margin,
                "model_total": model_total,
            }
        ]
    )


def published_fixture(kickoff=NOW):
    return advance(
        empty_live_ledger(),
        schedule_fixture(kickoff),
        predictions_fixture(),
        pd.Timestamp(kickoff) - pd.Timedelta(hours=2),
    )


def test_publication_starts_at_exactly_24_hours_but_not_before():
    kickoff = NOW + pd.Timedelta(hours=24)

    too_soon = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff + pd.Timedelta(microseconds=1)),
        predictions_fixture(),
        NOW,
    )
    boundary = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff),
        predictions_fixture(),
        NOW,
    )

    assert too_soon.empty
    assert list(too_soon.columns) == LEDGER_COLUMNS
    assert boundary["game_id"].tolist() == [GAME_ID]


def test_first_run_inside_24_hours_freezes_prediction_and_available_markets():
    kickoff = NOW + pd.Timedelta(hours=23)

    out = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff),
        predictions_fixture(model_margin=6.0, model_total=47.0),
        NOW,
    )

    row = out.iloc[0]
    assert row["published_at"] == NOW
    assert row["kickoff_at"] == kickoff
    assert row["current_kickoff_at"] == kickoff
    assert row["model_margin"] == 6.0
    assert row["model_total"] == 47.0
    assert row["spread_publication_status"] == "published"
    assert row["total_publication_status"] == "published"
    assert row["published_spread_line"] == row["official_spread_line"] == 3.0
    assert row["published_total_line"] == row["official_total_line"] == 44.0
    assert row["published_spread_observed_at"] == NOW
    assert row["published_total_observed_at"] == NOW


def test_missing_spread_retries_but_total_freezes_independently():
    kickoff = NOW + pd.Timedelta(hours=23)
    first = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff, spread=None, total=44.0),
        predictions_fixture(),
        NOW,
    )

    second_now = NOW + pd.Timedelta(hours=21)
    second = advance(
        first,
        schedule_fixture(kickoff, spread=2.5, total=46.0),
        predictions_fixture(model_margin=99.0, model_total=99.0),
        second_now,
    )

    assert first.iloc[0]["spread_publication_status"] == "pending"
    assert first.iloc[0]["total_publication_status"] == "published"
    row = second.iloc[0]
    assert row["published_spread_line"] == row["official_spread_line"] == 2.5
    assert row["published_spread_observed_at"] == second_now
    assert row["published_total_line"] == row["official_total_line"] == 44.0
    assert row["published_total_observed_at"] == NOW
    assert row["model_margin"] == first.iloc[0]["model_margin"]
    assert row["model_total"] == first.iloc[0]["model_total"]


def test_missing_market_at_exactly_one_hour_is_excluded_forever():
    kickoff = NOW + pd.Timedelta(hours=2)
    pending = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff, spread=None),
        predictions_fixture(),
        NOW,
    )

    deadline = NOW + pd.Timedelta(hours=1)
    excluded = advance(
        pending,
        schedule_fixture(kickoff, spread=None),
        predictions_fixture(),
        deadline,
    )
    later = advance(
        excluded,
        schedule_fixture(kickoff, spread=2.0),
        predictions_fixture(model_margin=-10.0),
        deadline + pd.Timedelta(minutes=30),
    )

    assert later.iloc[0]["spread_publication_status"] == "excluded"
    assert later.iloc[0]["spread_exclusion_reason"] == "missing_line_at_deadline"
    assert pd.isna(later.iloc[0]["published_spread_line"])
    assert pd.isna(later.iloc[0]["official_spread_line"])


def test_schedule_missing_at_deadline_still_excludes_pending_market():
    kickoff = NOW + pd.Timedelta(hours=2)
    pending = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff, spread=None),
        predictions_fixture(),
        NOW,
    )
    missing_schedule = schedule_fixture(kickoff).iloc[0:0]

    out = advance(
        pending,
        missing_schedule,
        pd.DataFrame(),
        NOW + pd.Timedelta(hours=1),
    )

    row = out.iloc[0]
    assert row["spread_publication_status"] == "excluded"
    assert row["spread_exclusion_reason"] == "missing_line_at_deadline"
    assert pd.isna(row["published_spread_line"])
    assert row["total_publication_status"] == "published"
    assert row["published_total_line"] == 44.0


def test_first_run_at_deadline_excludes_both_markets_even_when_lines_exist():
    kickoff = NOW + pd.Timedelta(hours=1)

    out = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff, spread=3.0, total=44.0),
        predictions_fixture(),
        NOW,
    )

    row = out.iloc[0]
    assert row["spread_publication_status"] == "excluded"
    assert row["total_publication_status"] == "excluded"
    assert row["spread_exclusion_reason"] == "publication_window_missed"
    assert row["total_exclusion_reason"] == "publication_window_missed"
    assert pd.isna(row["published_spread_line"])
    assert pd.isna(row["published_total_line"])


def test_finalization_waits_until_exactly_six_hours_then_captures_result_and_close():
    published = published_fixture()
    final_schedule = schedule_fixture(
        NOW,
        spread=5.0,
        total=46.0,
        result=6.0,
        actual_total=47.0,
    )

    too_early = advance(
        published,
        final_schedule,
        pd.DataFrame(),
        NOW + pd.Timedelta(hours=6, microseconds=-1),
    )
    final = advance(
        published,
        final_schedule,
        pd.DataFrame(),
        NOW + pd.Timedelta(hours=6),
    )

    assert pd.isna(too_early.iloc[0]["actual_margin"])
    assert pd.isna(too_early.iloc[0]["closing_spread_line"])
    row = final.iloc[0]
    assert row["actual_margin"] == 6.0
    assert row["actual_total"] == 47.0
    assert row["closing_spread_line"] == 5.0
    assert row["closing_total_line"] == 46.0
    assert row["closing_spread_observed_at"] == NOW + pd.Timedelta(hours=6)
    assert row["closing_total_observed_at"] == NOW + pd.Timedelta(hours=6)
    assert row["spread_clv"] == 2.0


def test_repeated_calls_are_idempotent_and_never_mutate_callers():
    kickoff = NOW + pd.Timedelta(hours=23)
    existing = empty_live_ledger()
    schedule = schedule_fixture(kickoff)
    predictions = predictions_fixture()
    existing_before = existing.copy(deep=True)
    schedule_before = schedule.copy(deep=True)
    predictions_before = predictions.copy(deep=True)

    first = advance(existing, schedule, predictions, NOW)
    second = advance(first, schedule, predictions, NOW)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(existing, existing_before)
    pd.testing.assert_frame_equal(schedule, schedule_before)
    pd.testing.assert_frame_equal(predictions, predictions_before)


def test_changed_predictions_cannot_change_frozen_model_facts():
    kickoff = NOW + pd.Timedelta(hours=23)
    first = advance(
        empty_live_ledger(), schedule_fixture(kickoff), predictions_fixture(), NOW
    )

    later = advance(
        first,
        schedule_fixture(kickoff, spread=4.0, total=45.0),
        predictions_fixture(model_margin=-100.0, model_total=100.0),
        NOW + pd.Timedelta(hours=1),
        model_version="ridge-v999",
    )

    for column in (
        "model_margin",
        "model_total",
        "model_version",
        "published_at",
        "official_spread_line",
        "official_total_line",
    ):
        assert later.iloc[0][column] == first.iloc[0][column]


def test_postponement_updates_only_current_kickoff_fact():
    original_kickoff = NOW + pd.Timedelta(hours=23)
    first = advance(
        empty_live_ledger(),
        schedule_fixture(original_kickoff),
        predictions_fixture(),
        NOW,
    )
    postponed = original_kickoff + pd.Timedelta(days=2)

    later = advance(
        first,
        schedule_fixture(postponed, spread=9.0, total=50.0),
        predictions_fixture(model_margin=-1.0, model_total=-1.0),
        NOW + pd.Timedelta(hours=1),
    )

    row = later.iloc[0]
    assert row["kickoff_at"] == original_kickoff
    assert row["current_kickoff_at"] == postponed
    assert row["published_at"] == first.iloc[0]["published_at"]
    assert row["published_spread_line"] == first.iloc[0]["published_spread_line"]
    assert row["published_total_line"] == first.iloc[0]["published_total_line"]
    assert row["model_margin"] == first.iloc[0]["model_margin"]


def test_missing_closing_markets_retry_and_freeze_independently():
    published = published_fixture()
    first_close_at = NOW + pd.Timedelta(hours=6)
    partial = advance(
        published,
        schedule_fixture(NOW, spread=5.0, total=None, result=6.0, actual_total=47.0),
        pd.DataFrame(),
        first_close_at,
    )
    completed_at = first_close_at + pd.Timedelta(hours=1)
    complete = advance(
        partial,
        schedule_fixture(NOW, spread=8.0, total=46.0, result=9.0, actual_total=60.0),
        pd.DataFrame(),
        completed_at,
    )

    partial_row = partial.iloc[0]
    assert partial_row["actual_margin"] == 6.0
    assert partial_row["actual_total"] == 47.0
    assert partial_row["closing_spread_line"] == 5.0
    assert pd.isna(partial_row["closing_total_line"])
    row = complete.iloc[0]
    assert row["actual_margin"] == 6.0
    assert row["actual_total"] == 47.0
    assert row["closing_spread_line"] == 5.0
    assert row["closing_spread_observed_at"] == first_close_at
    assert row["closing_total_line"] == 46.0
    assert row["closing_total_observed_at"] == completed_at


def test_incomplete_final_record_raises_at_exactly_seven_days():
    published = published_fixture()
    incomplete = schedule_fixture(NOW, spread=None, total=None, result=6.0, actual_total=47.0)

    just_before = advance(
        published,
        incomplete,
        pd.DataFrame(),
        NOW + pd.Timedelta(days=7, microseconds=-1),
    )

    assert just_before.iloc[0]["actual_margin"] == 6.0
    assert pd.isna(just_before.iloc[0]["closing_spread_line"])
    with pytest.raises(LiveTrackerLifecycleError, match=GAME_ID):
        advance(
            just_before,
            incomplete,
            pd.DataFrame(),
            NOW + pd.Timedelta(days=7),
        )


def test_schedule_missing_at_seven_days_still_raises_for_incomplete_record():
    published = published_fixture()
    missing_schedule = schedule_fixture(NOW).iloc[0:0]

    with pytest.raises(LiveTrackerLifecycleError, match=GAME_ID):
        advance(
            published,
            missing_schedule,
            pd.DataFrame(),
            NOW + pd.Timedelta(days=7),
        )


def test_schedule_missing_retains_completed_record_unchanged_after_seven_days():
    published = published_fixture()
    complete = advance(
        published,
        schedule_fixture(
            NOW,
            spread=5.0,
            total=46.0,
            result=6.0,
            actual_total=47.0,
        ),
        pd.DataFrame(),
        NOW + pd.Timedelta(hours=6),
    )
    missing_schedule = schedule_fixture(NOW).iloc[0:0]

    retained = advance(
        complete,
        missing_schedule,
        pd.DataFrame(),
        NOW + pd.Timedelta(days=8),
    )

    pd.testing.assert_frame_equal(retained, complete)


def test_manually_voided_game_bypasses_retry_error_and_grades_both_markets_no_pick():
    voided = published_fixture()
    voided.loc[0, "void_reason"] = "cancelled"

    out = advance(
        voided,
        schedule_fixture(NOW, spread=None, total=None),
        pd.DataFrame(),
        NOW + pd.Timedelta(days=7),
    )

    row = out.iloc[0]
    assert row["spread_grade"] == "no_pick"
    assert row["total_grade"] == "no_pick"
    assert row["spread_close_grade"] == "no_pick"
    assert row["total_close_grade"] == "no_pick"


def test_output_order_is_stable_by_game_id_and_unlisted_records_are_retained():
    kickoff = NOW + pd.Timedelta(hours=23)
    first = advance(
        empty_live_ledger(),
        schedule_fixture(kickoff, game_id="2026_01_CCC_DDD"),
        predictions_fixture(game_id="2026_01_CCC_DDD"),
        NOW,
    )
    schedule = pd.concat(
        [
            schedule_fixture(kickoff, game_id="2026_01_EEE_FFF"),
            schedule_fixture(kickoff, game_id="2026_01_AAA_BBB"),
        ],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            predictions_fixture(game_id="2026_01_EEE_FFF"),
            predictions_fixture(game_id="2026_01_AAA_BBB"),
        ],
        ignore_index=True,
    )

    out = advance(first, schedule, predictions, NOW)

    assert out["game_id"].tolist() == [
        "2026_01_AAA_BBB",
        "2026_01_CCC_DDD",
        "2026_01_EEE_FFF",
    ]


def test_floor_publishes_the_first_active_week():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    assert len(advanced) == 1
    assert advanced.loc[0, "week"] == 3


def test_floor_blocks_a_week_whose_features_predate_the_prior_week():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        NOW,
        first_publishable_week=2,
    )

    assert advanced.empty


def test_floor_blocks_the_week_after_the_first_active_week():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=4),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    assert advanced.empty


def test_floor_blocks_a_week_behind_the_floor():
    """A game whose week is BEHIND the floor must not publish.

    This is reachable when a past week's game never got published (e.g. after a
    tracker outage) and the floor has since moved on to a later week.
    """
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=2),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    assert advanced.empty


def test_floor_of_none_publishes_nothing():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=1),
        predictions_fixture(),
        NOW,
        first_publishable_week=None,
    )

    assert advanced.empty


def test_floor_never_blocks_an_existing_record_from_advancing():
    kickoff = NOW + pd.Timedelta(hours=12)
    published = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    # The floor has since moved on; the existing record must still advance.
    advanced = advance_live_ledger(
        published,
        schedule_fixture(kickoff, week=3, result=7.0, actual_total=45.0),
        predictions_fixture(),
        kickoff + pd.Timedelta(hours=7),
        first_publishable_week=4,
    )

    assert len(advanced) == 1
    assert advanced.loc[0, "actual_margin"] == 7.0
