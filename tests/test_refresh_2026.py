import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from scripts import refresh_2026 as refresh_cli

from nfl_game.model.features import FEATURE_COLS, _trailing_ngs
from nfl_game.pipeline.refresh_2026 import (
    RefreshArtifacts,
    build_refresh_artifacts,
    write_artifacts_atomic,
)
from nfl_game.ratings.ngs import NGS_METRICS

NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_cli_supports_the_documented_direct_script_invocation():
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "refresh_2026.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def historical_feature_fixture():
    rows = []
    for game_id, week in (("2025_01_AAA_BBB", 1), ("2025_02_CCC_DDD", 2)):
        row = {
            "game_id": game_id,
            "season": 2025,
            "week": week,
            "home_team": game_id[-3:],
            "away_team": game_id[8:11],
            "spread_line": float(week) + 0.5,
            "total_line": 40.0 + week,
            "margin": float(week),
            "total_points": 42.0 + week,
        }
        row.update({column: float(index + week) for index, column in enumerate(FEATURE_COLS)})
        rows.append(row)
    return pd.DataFrame(rows)


def normalized_2026_schedule_fixture(weeks=(1, 2, 3), completed=()):
    completed = set(completed)
    rows = []
    for week in weeks:
        is_complete = week in completed
        rows.append(
            {
                "game_id": f"2026_{week:02d}_AAA_BBB",
                "season": 2026,
                "game_type": "REG",
                "week": week,
                "gameday": f"2026-09-{6 + week * 7:02d}",
                "gametime": "13:00",
                "kickoff_at": pd.Timestamp(f"2026-08-{week:02d} 17:00", tz="UTC")
                if is_complete
                else pd.Timestamp(f"2026-09-{6 + week * 7:02d} 17:00", tz="UTC"),
                "away_team": "AAA",
                "home_team": "BBB",
                "home_rest": 7,
                "away_rest": 7,
                "roof": "outdoors",
                "temp": 70.0,
                "wind": 5.0,
                "div_game": 0,
                "result": 3.0 if is_complete else float("nan"),
                "total": 41.0 if is_complete else float("nan"),
                "spread_line": -2.5,
                "total_line": 44.0,
            }
        )
    return pd.DataFrame(rows)


def rating_fixture(targets):
    rows = []
    for season, week in targets:
        for team in ("AAA", "BBB"):
            rows.append({"season": season, "week": week, "team": team})
    return pd.DataFrame(rows)


def feature_fixture_for_schedule(schedules):
    rows = []
    for game in schedules.itertuples():
        row = {
            "game_id": game.game_id,
            "season": game.season,
            "week": game.week,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "spread_line": game.spread_line,
            "total_line": game.total_line,
            "margin": game.result,
            "total_points": game.total,
        }
        row.update({column: float(index + game.week) for index, column in enumerate(FEATURE_COLS)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_refresh_preserves_historical_rows_byte_for_value_and_adds_only_active_2026_weeks(
    monkeypatch,
):
    historical = historical_feature_fixture()
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2, 3))

    monkeypatch.setattr(
        "nfl_game.pipeline.refresh_2026.ratings_for_targets",
        lambda team_games, targets: rating_fixture(targets),
    )
    monkeypatch.setattr(
        "nfl_game.pipeline.refresh_2026.build_game_features",
        lambda schedules, ratings, ngs: feature_fixture_for_schedule(schedules),
    )

    result = build_refresh_artifacts(
        historical_features=historical,
        schedules=schedule,
        team_games=pd.DataFrame(),
        ngs=pd.DataFrame(),
        now=NOW,
    )

    pd.testing.assert_frame_equal(
        result.features.query("season <= 2025").reset_index(drop=True),
        historical.reset_index(drop=True),
        check_exact=True,
    )
    assert sorted(result.features.query("season == 2026")["week"].unique()) == [1, 2]
    assert result.features["game_id"].is_unique
    assert result.features[FEATURE_COLS].notna().all().all()
    assert sorted(result.schedule["week"].unique()) == [1, 2, 3]


def test_no_active_weeks_returns_only_the_frozen_history_without_building_live_rows(
    monkeypatch,
):
    historical = historical_feature_fixture()
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2), completed=(1, 2))

    def unexpected_call(*args, **kwargs):
        raise AssertionError("live feature dependencies must not run without active weeks")

    monkeypatch.setattr("nfl_game.pipeline.refresh_2026.ratings_for_targets", unexpected_call)
    monkeypatch.setattr("nfl_game.pipeline.refresh_2026.build_game_features", unexpected_call)

    result = build_refresh_artifacts(historical, schedule, pd.DataFrame(), pd.DataFrame(), NOW)

    pd.testing.assert_frame_equal(
        result.features, historical.reset_index(drop=True), check_exact=True
    )
    pd.testing.assert_frame_equal(result.schedule, schedule, check_exact=True)


def test_empty_typed_ngs_input_produces_the_keyed_empty_trailing_schema():
    ngs = refresh_cli.empty_ngs_frame()

    trailing = _trailing_ngs(ngs, halflife=4.0)

    assert list(ngs.columns) == [
        "season",
        "week",
        "team",
        *NGS_METRICS,
        *[f"{metric}_imputed" for metric in NGS_METRICS],
    ]
    assert str(ngs["season"].dtype) == "int64"
    assert str(ngs["week"].dtype) == "int64"
    assert str(ngs["team"].dtype) == "string"
    assert list(trailing.columns) == [
        "season",
        "week",
        "team",
        *[f"trail_{metric}" for metric in NGS_METRICS],
        "trail_imputed_any",
    ]
    assert trailing.empty


def test_failed_second_parquet_write_leaves_both_original_artifacts_unchanged(
    tmp_path, monkeypatch
):
    feature_path = tmp_path / "game_features.parquet"
    schedule_path = tmp_path / "schedule_2026.parquet"
    historical_feature_fixture().to_parquet(feature_path, index=False)
    normalized_2026_schedule_fixture(weeks=(1,)).to_parquet(schedule_path, index=False)
    original_features = feature_path.read_bytes()
    original_schedule = schedule_path.read_bytes()
    artifacts = RefreshArtifacts(
        features=feature_fixture_for_schedule(normalized_2026_schedule_fixture(weeks=(1, 2))),
        schedule=normalized_2026_schedule_fixture(weeks=(1, 2)),
    )
    real_to_parquet = pd.DataFrame.to_parquet
    calls = 0

    def fail_second_write(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second parquet write failure")
        return real_to_parquet(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second_write)

    with pytest.raises(OSError, match="second parquet write failure"):
        write_artifacts_atomic(artifacts, feature_path, schedule_path)

    assert feature_path.read_bytes() == original_features
    assert schedule_path.read_bytes() == original_schedule


def test_failed_second_replacement_rolls_back_the_first_replacement(tmp_path, monkeypatch):
    feature_path = tmp_path / "game_features.parquet"
    schedule_path = tmp_path / "schedule_2026.parquet"
    historical_feature_fixture().to_parquet(feature_path, index=False)
    normalized_2026_schedule_fixture(weeks=(1,)).to_parquet(schedule_path, index=False)
    original_features = feature_path.read_bytes()
    original_schedule = schedule_path.read_bytes()
    artifacts = RefreshArtifacts(
        features=feature_fixture_for_schedule(normalized_2026_schedule_fixture(weeks=(1, 2))),
        schedule=normalized_2026_schedule_fixture(weeks=(1, 2)),
    )
    real_replace = Path.replace
    failed = False

    def fail_feature_publication(source, destination):
        nonlocal failed
        source = Path(source)
        if not failed and Path(destination) == feature_path and ".publish-" in source.name:
            failed = True
            raise OSError("simulated feature replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_feature_publication)

    with pytest.raises(OSError, match="feature replacement failure"):
        write_artifacts_atomic(artifacts, feature_path, schedule_path)

    assert feature_path.read_bytes() == original_features
    assert schedule_path.read_bytes() == original_schedule


def test_atomic_writer_is_digest_aware_and_replaces_only_changed_content(tmp_path, monkeypatch):
    feature_path = tmp_path / "game_features.parquet"
    schedule_path = tmp_path / "schedule_2026.parquet"
    features = historical_feature_fixture()
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2))
    features.to_parquet(feature_path, index=False)
    schedule.to_parquet(schedule_path, index=False)
    original_feature_bytes = feature_path.read_bytes()
    original_schedule_bytes = schedule_path.read_bytes()
    replacements = []
    real_replace = Path.replace

    def recording_replace(source, destination):
        replacements.append(Path(destination).name)
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", recording_replace)

    write_artifacts_atomic(RefreshArtifacts(features, schedule), feature_path, schedule_path)

    assert replacements == []
    assert feature_path.read_bytes() == original_feature_bytes
    assert schedule_path.read_bytes() == original_schedule_bytes

    changed_schedule = schedule.copy()
    changed_schedule.loc[0, "spread_line"] = -3.0
    write_artifacts_atomic(
        RefreshArtifacts(features, changed_schedule), feature_path, schedule_path
    )

    assert replacements == ["schedule_2026.parquet"]
    assert feature_path.read_bytes() == original_feature_bytes
    assert schedule_path.read_bytes() != original_schedule_bytes


def test_atomic_writer_replaces_from_destination_directory_for_acl_inheritance(
    tmp_path, monkeypatch
):
    feature_path = tmp_path / "game_features.parquet"
    schedule_path = tmp_path / "schedule_2026.parquet"
    artifacts = RefreshArtifacts(
        historical_feature_fixture(), normalized_2026_schedule_fixture(weeks=(1, 2))
    )
    replacement_sources = []
    real_replace = Path.replace

    def recording_replace(source, destination):
        replacement_sources.append(Path(source).parent.resolve())
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", recording_replace)

    write_artifacts_atomic(artifacts, feature_path, schedule_path)

    assert replacement_sources == [tmp_path.resolve(), tmp_path.resolve()]


def _cli_loaders(schedule):
    calls = {"pbp_seasons": [], "ngs": []}

    def load_pbp(seasons, save=False):
        calls["pbp_seasons"].append(list(seasons))
        return pd.DataFrame({"sentinel": [1]})

    def load_ngs(seasons, stat_type, save=False):
        calls["ngs"].append((list(seasons), stat_type))
        return pd.DataFrame({"sentinel": [1]})

    loaders = {
        "read_parquet": pd.read_parquet,
        "load_schedules": lambda seasons, save=False: schedule.copy(),
        "load_pbp": load_pbp,
        "load_ngs": load_ngs,
    }
    return loaders, calls


def test_cli_runs_the_exact_historical_acceptance_gate_before_any_artifact_write(
    tmp_path, monkeypatch
):
    feature_path = tmp_path / "game_features.parquet"
    schedule_path = tmp_path / "schedule_2026.parquet"
    historical_feature_fixture().to_parquet(feature_path, index=False)
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2))
    loaders, _ = _cli_loaders(schedule)
    artifacts = RefreshArtifacts(historical_feature_fixture(), schedule)
    events = []

    monkeypatch.setattr(refresh_cli, "normalize_schedule", lambda rows, season: rows.copy())
    monkeypatch.setattr(refresh_cli, "team_game_epa", lambda pbp: pd.DataFrame())
    monkeypatch.setattr(refresh_cli, "build_refresh_artifacts", lambda **kwargs: artifacts)
    monkeypatch.setattr(
        refresh_cli,
        "build_historical_ledger",
        lambda features: events.append("build historical ledger") or pd.DataFrame(),
    )
    monkeypatch.setattr(
        refresh_cli,
        "assert_acceptance_baseline",
        lambda ledger, expected: events.append("acceptance gate"),
    )
    monkeypatch.setattr(
        refresh_cli,
        "write_artifacts_atomic",
        lambda *args: events.append("artifact write"),
    )

    refresh_cli.main(
        ["--write", "--features", str(feature_path), "--schedule", str(schedule_path)],
        loaders=loaders,
        now=NOW,
    )

    assert events == ["build historical ledger", "acceptance gate", "artifact write"]


@pytest.mark.parametrize("mode", [[], ["--dry-run"]])
def test_cli_dry_run_and_default_mode_never_change_output_files(tmp_path, monkeypatch, mode):
    feature_path = tmp_path / "game_features.parquet"
    schedule_path = tmp_path / "schedule_2026.parquet"
    historical_feature_fixture().to_parquet(feature_path, index=False)
    normalized_2026_schedule_fixture(weeks=(1,)).to_parquet(schedule_path, index=False)
    old_features = feature_path.read_bytes()
    old_schedule = schedule_path.read_bytes()
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2))
    loaders, _ = _cli_loaders(schedule)

    monkeypatch.setattr(refresh_cli, "normalize_schedule", lambda rows, season: rows.copy())
    monkeypatch.setattr(refresh_cli, "team_game_epa", lambda pbp: pd.DataFrame())
    monkeypatch.setattr(
        refresh_cli,
        "build_refresh_artifacts",
        lambda **kwargs: RefreshArtifacts(historical_feature_fixture(), schedule),
    )
    monkeypatch.setattr(refresh_cli, "build_historical_ledger", lambda features: pd.DataFrame())
    monkeypatch.setattr(refresh_cli, "assert_acceptance_baseline", lambda ledger, expected: None)

    refresh_cli.main(
        [*mode, "--features", str(feature_path), "--schedule", str(schedule_path)],
        loaders=loaders,
        now=NOW,
    )

    assert feature_path.read_bytes() == old_features
    assert schedule_path.read_bytes() == old_schedule


def test_cli_requests_2026_pbp_and_ngs_only_after_a_regular_season_game_is_final(
    tmp_path, monkeypatch
):
    feature_path = tmp_path / "game_features.parquet"
    historical_feature_fixture().to_parquet(feature_path, index=False)
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2), completed=(1,))
    loaders, calls = _cli_loaders(schedule)
    captured = {}

    def load_empty_ngs(seasons, stat_type, save=False):
        calls["ngs"].append((list(seasons), stat_type))
        return pd.DataFrame(
            {
                "season_type": pd.Series(dtype="string"),
                "week": pd.Series(dtype="int64"),
            }
        )

    def capture_artifacts(**kwargs):
        captured.update(kwargs)
        return RefreshArtifacts(historical_feature_fixture(), schedule)

    loaders["load_ngs"] = load_empty_ngs

    monkeypatch.setattr(refresh_cli, "normalize_schedule", lambda rows, season: rows.copy())
    monkeypatch.setattr(refresh_cli, "team_game_epa", lambda pbp: pd.DataFrame())
    monkeypatch.setattr(refresh_cli, "build_refresh_artifacts", capture_artifacts)
    monkeypatch.setattr(refresh_cli, "build_historical_ledger", lambda features: pd.DataFrame())
    monkeypatch.setattr(refresh_cli, "assert_acceptance_baseline", lambda ledger, expected: None)

    refresh_cli.main(
        ["--dry-run", "--features", str(feature_path), "--schedule", str(tmp_path / "s.parquet")],
        loaders=loaders,
        now=NOW,
    )

    assert calls["pbp_seasons"][0][-1] == 2026
    assert calls["ngs"] == [
        ([2026], "passing"),
        ([2026], "rushing"),
        ([2026], "receiving"),
    ]
    pd.testing.assert_frame_equal(captured["ngs"], refresh_cli.empty_ngs_frame())
    assert "spread_line" not in FEATURE_COLS
    assert "total_line" not in FEATURE_COLS
