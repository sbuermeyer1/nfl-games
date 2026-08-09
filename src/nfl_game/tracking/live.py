"""Pure lifecycle transitions for official live tracker records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_game.tracking.ledger import (
    HISTORICAL_MODEL_VERSION,
    LEDGER_COLUMNS,
    OFFICIAL_ESTIMATOR,
    grade_ledger,
    validate_ledger,
)

PUBLISH_BEFORE = pd.Timedelta(hours=24)
LINE_DEADLINE = pd.Timedelta(hours=1)
FINALIZATION_DELAY = pd.Timedelta(hours=6)
FINAL_RETRY_LIMIT = pd.Timedelta(days=7)
LIVE_LEDGER_COLUMNS = LEDGER_COLUMNS


class LiveTrackerLifecycleError(RuntimeError):
    """Raised when a live record cannot safely advance without manual review."""


def _utc_timestamp(value, label):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise LiveTrackerLifecycleError(f"{label} must be a valid UTC timestamp") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise LiveTrackerLifecycleError(f"{label} must be a valid UTC timestamp")
    return timestamp


def _finite_number(value):
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _new_market(record, game, kind, now, missed_window):
    advanced = record.copy()
    status_column = f"{kind}_publication_status"
    reason_column = f"{kind}_exclusion_reason"
    official_column = f"official_{kind}_line"
    published_column = f"published_{kind}_line"
    observed_column = f"published_{kind}_observed_at"
    line = _finite_number(game.get(f"{kind}_line"))

    if missed_window:
        advanced[status_column] = "excluded"
        advanced[reason_column] = "publication_window_missed"
        advanced[official_column] = np.nan
        advanced[published_column] = np.nan
        advanced[observed_column] = pd.NaT
    elif line is None:
        advanced[status_column] = "pending"
        advanced[reason_column] = pd.NA
        advanced[official_column] = np.nan
        advanced[published_column] = np.nan
        advanced[observed_column] = pd.NaT
    else:
        advanced[status_column] = "published"
        advanced[reason_column] = pd.NA
        advanced[official_column] = line
        advanced[published_column] = line
        advanced[observed_column] = now
    return advanced


def _new_record(game, prediction, now, model_version):
    kickoff = _utc_timestamp(game.get("kickoff_at"), "kickoff_at")
    model_margin = _finite_number(prediction.get("model_margin"))
    model_total = _finite_number(prediction.get("model_total"))
    if model_margin is None or model_total is None:
        raise LiveTrackerLifecycleError(
            f"eligible game {game.get('game_id')} has a non-finite Ridge prediction"
        )

    record = {
        "record_type": "live",
        "model_version": model_version,
        "estimator": OFFICIAL_ESTIMATOR,
        "game_id": str(game["game_id"]),
        "season": game["season"],
        "week": game["week"],
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "model_margin": model_margin,
        "model_total": model_total,
        "published_spread_line": np.nan,
        "published_total_line": np.nan,
        "official_spread_line": np.nan,
        "official_total_line": np.nan,
        "closing_spread_line": np.nan,
        "closing_total_line": np.nan,
        "published_at": now,
        "kickoff_at": kickoff,
        "actual_margin": np.nan,
        "actual_total": np.nan,
        "spread_publication_status": pd.NA,
        "total_publication_status": pd.NA,
        "spread_exclusion_reason": pd.NA,
        "total_exclusion_reason": pd.NA,
        "published_spread_observed_at": pd.NaT,
        "published_total_observed_at": pd.NaT,
        "closing_spread_observed_at": pd.NaT,
        "closing_total_observed_at": pd.NaT,
        "current_kickoff_at": kickoff,
        "void_reason": pd.NA,
    }
    missed_window = now >= kickoff - LINE_DEADLINE
    record = _new_market(record, game, "spread", now, missed_window)
    return _new_market(record, game, "total", now, missed_window)


def _apply_schedule_change(record, game):
    advanced = record.copy()
    advanced["current_kickoff_at"] = _utc_timestamp(game.get("kickoff_at"), "kickoff_at")
    return advanced


def _advance_market(record, game, kind, now):
    advanced = record.copy()
    status_column = f"{kind}_publication_status"
    if advanced[status_column] != "pending":
        return advanced

    kickoff = _utc_timestamp(game.get("kickoff_at"), "kickoff_at")
    if now >= kickoff - LINE_DEADLINE:
        advanced[status_column] = "excluded"
        advanced[f"{kind}_exclusion_reason"] = "missing_line_at_deadline"
        return advanced

    line = _finite_number(game.get(f"{kind}_line"))
    if line is not None:
        advanced[status_column] = "published"
        advanced[f"official_{kind}_line"] = line
        advanced[f"published_{kind}_line"] = line
        advanced[f"published_{kind}_observed_at"] = now
    return advanced


def _is_void(record):
    reason = record.get("void_reason")
    return reason is not None and reason is not pd.NA and not pd.isna(reason)


def _capture_final(record, game, now):
    advanced = record.copy()
    if _is_void(advanced):
        return advanced

    kickoff = _utc_timestamp(game.get("kickoff_at"), "kickoff_at")
    if now < kickoff + FINALIZATION_DELAY:
        return advanced

    existing_margin = _finite_number(advanced.get("actual_margin"))
    existing_total = _finite_number(advanced.get("actual_total"))
    result = _finite_number(game.get("result"))
    actual_total = _finite_number(game.get("total"))
    if existing_margin is None and existing_total is None and result is not None and actual_total is not None:
        advanced["actual_margin"] = result
        advanced["actual_total"] = actual_total

    result_complete = (
        _finite_number(advanced.get("actual_margin")) is not None
        and _finite_number(advanced.get("actual_total")) is not None
    )
    if result_complete:
        for kind in ("spread", "total"):
            if advanced[f"{kind}_publication_status"] != "published":
                continue
            if _finite_number(advanced.get(f"closing_{kind}_line")) is not None:
                continue
            line = _finite_number(game.get(f"{kind}_line"))
            if line is not None:
                advanced[f"closing_{kind}_line"] = line
                advanced[f"closing_{kind}_observed_at"] = now

    required_closes_complete = all(
        advanced[f"{kind}_publication_status"] != "published"
        or _finite_number(advanced.get(f"closing_{kind}_line")) is not None
        for kind in ("spread", "total")
    )
    if result_complete and required_closes_complete:
        return advanced
    if now >= kickoff + FINAL_RETRY_LIMIT:
        raise LiveTrackerLifecycleError(
            f"live tracker record {advanced['game_id']} remains incomplete after seven days"
        )
    return advanced


def advance_live_ledger(
    existing_live,
    schedule,
    predictions,
    now,
    model_version=HISTORICAL_MODEL_VERSION,
):
    """Advance live facts without mutating any caller-owned frame."""
    now = _utc_timestamp(now, "now")
    schedule_rows = {
        str(row.game_id): row._asdict()
        for row in schedule.copy(deep=True).itertuples(index=False)
    }
    prediction_rows = {
        str(row.game_id): row._asdict()
        for row in predictions.copy(deep=True).itertuples(index=False)
    }
    records = {
        str(row.game_id): row._asdict()
        for row in existing_live.copy(deep=True).itertuples(index=False)
    }
    advanced = []

    for game_id in sorted(schedule_rows):
        game = schedule_rows[game_id]
        kickoff = _utc_timestamp(game.get("kickoff_at"), "kickoff_at")
        record = records.pop(game_id, None)
        if record is None:
            if now < kickoff - PUBLISH_BEFORE:
                continue
            prediction = prediction_rows.get(game_id)
            if prediction is None:
                raise LiveTrackerLifecycleError(
                    f"eligible game {game_id} has no Ridge prediction"
                )
            record = _new_record(game, prediction, now, model_version)
        else:
            record = _apply_schedule_change(record, game)

        record = _advance_market(record, game, "spread", now)
        record = _advance_market(record, game, "total", now)
        record = _capture_final(record, game, now)
        advanced.append(record)

    for game_id in sorted(records):
        record = records[game_id]
        persisted_game = {
            "kickoff_at": record["current_kickoff_at"],
            "spread_line": np.nan,
            "total_line": np.nan,
            "result": np.nan,
            "total": np.nan,
        }
        record = _advance_market(record, persisted_game, "spread", now)
        record = _advance_market(record, persisted_game, "total", now)
        record = _capture_final(record, persisted_game, now)
        advanced.append(record)
    result = pd.DataFrame.from_records(advanced, columns=LIVE_LEDGER_COLUMNS)
    result = grade_ledger(result)
    result = result.sort_values("game_id", kind="stable").reset_index(drop=True)
    if not result.empty:
        validate_ledger(result)
    return result
