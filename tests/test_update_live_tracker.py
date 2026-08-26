import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from scripts import build_tracker, update_live_tracker

from nfl_game.tracking.live import LiveTrackerLifecycleError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = pd.Timestamp("2026-09-05T17:00:00Z")
GAME_ID = "2026_01_NE_SEA"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_inside_publish_window():
    schedule = pd.read_parquet(PROJECT_ROOT / "data/processed/schedule_2026.parquet")
    row = schedule.loc[schedule["game_id"].eq(GAME_ID)].iloc[[0]].copy()
    row["gameday"] = "2026-09-06"
    row["gametime"] = "13:00"
    row["result"] = pd.NA
    row["total"] = pd.NA
    row["spread_line"] = 2.5
    row["total_line"] = 45.5
    return row


def write_artifacts(tmp_path):
    feature_path = tmp_path / "game_features.parquet"
    ledger_path = tmp_path / "tracker_ledger.parquet"
    shutil.copyfile(PROJECT_ROOT / "data/processed/game_features.parquet", feature_path)
    shutil.copyfile(PROJECT_ROOT / "data/processed/tracker_ledger.parquet", ledger_path)
    return feature_path, ledger_path


def run_cli(tmp_path, *mode, monkeypatch, schedule=None, now=NOW, voids=()):
    feature_path = tmp_path / "game_features.parquet"
    ledger_path = tmp_path / "tracker_ledger.parquet"
    prediction_calls = []

    def predictions(self, season, week, estimator="ridge"):
        prediction_calls.append((season, week, estimator))
        return pd.DataFrame({"game_id": [GAME_ID], "model_margin": [4.0], "model_total": [47.0]})

    monkeypatch.setattr(update_live_tracker.SlateService, "model_predictions", predictions)
    argv = [
        "--features",
        str(feature_path),
        "--ledger",
        str(ledger_path),
        "--season",
        "2026",
        *mode,
    ]
    for value in voids:
        argv.extend(["--void-game", value])
    result = update_live_tracker.main(
        argv,
        loader=lambda seasons, save=False: (
            schedule_inside_publish_window() if schedule is None else schedule.copy()
        ),
        now=now,
    )
    return result, prediction_calls


def test_default_dry_run_reports_change_without_writing(tmp_path, monkeypatch, capsys):
    _, ledger_path = write_artifacts(tmp_path)
    original = ledger_path.read_bytes()
    before = sorted(path.name for path in tmp_path.iterdir())

    result, calls = run_cli(tmp_path, monkeypatch=monkeypatch)

    assert result == 0
    assert ledger_path.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert calls == [(2026, 1, "ridge")]
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "changed": True,
        "historical_records": 1359,
        "live_records": 1,
        "mode": "dry-run",
        "new_live_records": 1,
        "voided_records": 0,
    }


def test_write_combines_unchanged_history_with_valid_live_rows(tmp_path, monkeypatch):
    _, ledger_path = write_artifacts(tmp_path)

    result, _ = run_cli(tmp_path, "--write", monkeypatch=monkeypatch)

    ledger = pd.read_parquet(ledger_path)
    historical = ledger.query("record_type == 'backtest'")
    assert result == 0
    assert len(historical) == 1359
    assert len(ledger.query("record_type == 'live'")) == 1
    build_tracker.assert_acceptance_baseline(historical, build_tracker.EXPECTED_BASELINE)


def test_identical_write_is_digest_no_op_and_does_not_repredict(tmp_path, monkeypatch, capsys):
    _, ledger_path = write_artifacts(tmp_path)
    first, first_calls = run_cli(tmp_path, "--write", monkeypatch=monkeypatch)
    digest = sha256_file(ledger_path)
    capsys.readouterr()

    second, second_calls = run_cli(tmp_path, "--write", monkeypatch=monkeypatch)
    summary = json.loads(capsys.readouterr().out)

    assert first == second == 0
    assert first_calls == [(2026, 1, "ridge")]
    assert second_calls == []
    assert sha256_file(ledger_path) == digest
    assert summary["changed"] is False
    assert summary["new_live_records"] == 0


@pytest.mark.parametrize("artifact", ["ledger", "features"])
def test_corrupt_input_fails_before_writing(tmp_path, monkeypatch, artifact):
    feature_path, ledger_path = write_artifacts(tmp_path)
    if artifact == "ledger":
        corrupt = pd.read_parquet(ledger_path)
        corrupt.loc[0, "estimator"] = "gbm"
        corrupt.to_parquet(ledger_path, index=False)
    else:
        corrupt = pd.read_parquet(feature_path).drop(columns="game_id")
        corrupt.to_parquet(feature_path, index=False)
    original = ledger_path.read_bytes()

    with pytest.raises(ValueError):
        run_cli(tmp_path, "--write", monkeypatch=monkeypatch)

    assert ledger_path.read_bytes() == original


def test_lifecycle_failure_does_not_replace_ledger(tmp_path, monkeypatch):
    _, ledger_path = write_artifacts(tmp_path)
    run_cli(tmp_path, "--write", monkeypatch=monkeypatch)
    original = ledger_path.read_bytes()
    overdue = schedule_inside_publish_window()
    overdue["gameday"] = "2026-08-25"

    with pytest.raises(LiveTrackerLifecycleError, match="incomplete after seven days"):
        run_cli(tmp_path, "--write", monkeypatch=monkeypatch, schedule=overdue)

    assert ledger_path.read_bytes() == original


def test_repeatable_manual_void_is_applied_before_overdue_validation(tmp_path, monkeypatch, capsys):
    _, ledger_path = write_artifacts(tmp_path)
    run_cli(tmp_path, "--write", monkeypatch=monkeypatch)
    overdue = schedule_inside_publish_window()
    overdue["gameday"] = "2026-08-25"
    capsys.readouterr()

    first, _ = run_cli(
        tmp_path,
        "--write",
        monkeypatch=monkeypatch,
        schedule=overdue,
        voids=(f"{GAME_ID}=cancelled",),
    )
    digest = sha256_file(ledger_path)
    second, _ = run_cli(
        tmp_path,
        "--write",
        monkeypatch=monkeypatch,
        schedule=overdue,
        voids=(f"{GAME_ID}=cancelled",),
    )

    ledger = pd.read_parquet(ledger_path)
    assert first == second == 0
    assert ledger.query("record_type == 'live'").iloc[0]["void_reason"] == "cancelled"
    assert sha256_file(ledger_path) == digest


def test_atomic_replace_failure_preserves_ledger_and_cleans_temporary_file(tmp_path, monkeypatch):
    _, ledger_path = write_artifacts(tmp_path)
    original = ledger_path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(update_live_tracker.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        run_cli(tmp_path, "--write", monkeypatch=monkeypatch)

    assert ledger_path.read_bytes() == original
    assert list(tmp_path.glob(f".{ledger_path.name}.update-*.tmp")) == []


def test_atomic_staging_failure_preserves_ledger_and_cleans_temporary_file(tmp_path, monkeypatch):
    _, ledger_path = write_artifacts(tmp_path)
    original = ledger_path.read_bytes()

    def fail_fsync(file_descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr(update_live_tracker.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        run_cli(tmp_path, "--write", monkeypatch=monkeypatch)

    assert ledger_path.read_bytes() == original
    assert list(tmp_path.glob(f".{ledger_path.name}.update-*.tmp")) == []


def test_dry_run_before_publication_window_handles_no_live_rows(tmp_path, monkeypatch, capsys):
    _, ledger_path = write_artifacts(tmp_path)
    original = ledger_path.read_bytes()

    result, calls = run_cli(
        tmp_path,
        "--dry-run",
        monkeypatch=monkeypatch,
        now=pd.Timestamp("2026-08-01T17:00:00Z"),
    )

    assert result == 0
    assert calls == []
    assert ledger_path.read_bytes() == original
    summary = json.loads(capsys.readouterr().out)
    assert summary["live_records"] == 0
    assert summary["new_live_records"] == 0


def invalid_schedule(case):
    schedule = schedule_inside_publish_window()
    if case == "empty":
        return schedule.iloc[0:0].copy()
    if case == "game_identity":
        schedule.loc[:, "home_team"] = "LAR"
        return schedule
    full = pd.read_parquet(PROJECT_ROOT / "data/processed/schedule_2026.parquet")
    duplicate = full.loc[full["game_id"].eq("2026_01_SF_LA")].iloc[[0]].copy()
    duplicate.loc[:, "game_id"] = "2026_01_NE_LA"
    duplicate.loc[:, "away_team"] = "NE"
    duplicate.loc[:, "gameday"] = "2026-09-06"
    duplicate.loc[:, "gametime"] = "16:00"
    duplicate.loc[:, "result"] = float("nan")
    duplicate.loc[:, "total"] = float("nan")
    return pd.concat([schedule, duplicate], ignore_index=True)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty", "no regular-season games"),
        ("game_identity", "team mismatch with game_id"),
        ("duplicate_team", "team mismatch within a week"),
    ],
)
def test_invalid_current_schedule_is_rejected_without_writing(tmp_path, monkeypatch, case, message):
    _, ledger_path = write_artifacts(tmp_path)
    original = ledger_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        run_cli(
            tmp_path,
            "--write",
            monkeypatch=monkeypatch,
            schedule=invalid_schedule(case),
        )

    assert ledger_path.read_bytes() == original


def test_swapped_feature_game_ids_are_rejected_without_writing(tmp_path, monkeypatch):
    feature_path, ledger_path = write_artifacts(tmp_path)
    features = pd.read_parquet(feature_path)
    swapped = features["game_id"].isin([GAME_ID, "2026_01_SF_LA"])
    features.loc[swapped, "game_id"] = features.loc[swapped, "game_id"].iloc[::-1].to_numpy()
    features.to_parquet(feature_path, index=False)
    original = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="feature identity does not match schedule"):
        run_cli(tmp_path, "--write", monkeypatch=monkeypatch)

    assert ledger_path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["extra", "reordered"])
def test_nonexact_ledger_schema_is_rejected_without_writing(tmp_path, monkeypatch, mutation):
    _, ledger_path = write_artifacts(tmp_path)
    ledger = pd.read_parquet(ledger_path)
    if mutation == "extra":
        ledger["unexpected"] = 1
    else:
        columns = ledger.columns.tolist()
        columns[0], columns[1] = columns[1], columns[0]
        ledger = ledger.loc[:, columns]
    ledger.to_parquet(ledger_path, index=False)
    original = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="ledger schema"):
        run_cli(tmp_path, "--write", monkeypatch=monkeypatch)

    assert ledger_path.read_bytes() == original


def test_first_publishable_week_is_the_minimum_week_for_the_season():
    features = pd.DataFrame(
        [
            {"game_id": "2025_18_AAA_BBB", "season": 2025, "week": 18},
            {"game_id": "2026_03_AAA_BBB", "season": 2026, "week": 3},
            {"game_id": "2026_04_AAA_BBB", "season": 2026, "week": 4},
        ]
    )

    assert update_live_tracker._first_publishable_week(features, 2026) == 3


def test_first_publishable_week_is_none_when_the_season_has_no_rows():
    features = pd.DataFrame([{"game_id": "2025_18_AAA_BBB", "season": 2025, "week": 18}])

    assert update_live_tracker._first_publishable_week(features, 2026) is None


def test_select_schedule_excludes_weeks_above_the_floor():
    now = pd.Timestamp("2026-09-20T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_03_AAA_BBB",
                "week": 3,
                "kickoff_at": now + pd.Timedelta(hours=12),
            },
            {
                "game_id": "2026_04_CCC_DDD",
                "week": 4,
                "kickoff_at": now + pd.Timedelta(hours=13),
            },
        ]
    )
    live = pd.DataFrame({"game_id": pd.Series(dtype=str)})

    selected = update_live_tracker._select_schedule(schedule, live, now, 3)

    assert selected["game_id"].tolist() == ["2026_03_AAA_BBB"]


def test_select_schedule_keeps_existing_records_above_the_floor():
    now = pd.Timestamp("2026-09-20T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_04_CCC_DDD",
                "week": 4,
                "kickoff_at": now + pd.Timedelta(hours=13),
            }
        ]
    )
    live = pd.DataFrame({"game_id": ["2026_04_CCC_DDD"]})

    selected = update_live_tracker._select_schedule(schedule, live, now, 3)

    assert selected["game_id"].tolist() == ["2026_04_CCC_DDD"]


def test_select_schedule_publishes_nothing_when_the_floor_is_none():
    now = pd.Timestamp("2026-09-20T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_03_AAA_BBB",
                "week": 3,
                "kickoff_at": now + pd.Timedelta(hours=12),
            }
        ]
    )
    live = pd.DataFrame({"game_id": pd.Series(dtype=str)})

    selected = update_live_tracker._select_schedule(schedule, live, now, None)

    assert selected.empty
