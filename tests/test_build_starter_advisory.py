from datetime import UTC, datetime

import pandas as pd
import pytest
from scripts.build_starter_advisory import ARTIFACT_COLUMNS, build_advisory_artifact, main

from nfl_game.data.schedule import normalize_schedule
from nfl_game.ratings.starters import ADVISORY_COLS


def _packaged_schedule_2026():
    """A normalize_schedule-shaped packaged 2026 schedule, weeks 1-2 unplayed."""
    raw = pd.DataFrame(
        {
            "game_id": ["2026_01_CIN_BAL", "2026_02_KC_DEN"],
            "season": [2026, 2026],
            "game_type": ["REG", "REG"],
            "week": [1, 2],
            "gameday": ["2026-09-10", "2026-09-17"],
            "gametime": ["13:00", "13:00"],
            "away_team": ["CIN", "KC"],
            "home_team": ["BAL", "DEN"],
            "result": [None, None],
            "total": [None, None],
            "spread_line": [-2.5, -3.0],
            "total_line": [44.5, 43.0],
        }
    )
    return normalize_schedule(raw, 2026)


def _live_schedules():
    return pd.DataFrame(
        {
            "game_id": ["2026_01_CIN_BAL", "2026_02_KC_DEN"],
            "season": [2026, 2026],
            "week": [1, 2],
            "home_team": ["BAL", "DEN"],
            "away_team": ["CIN", "KC"],
            "kickoff_at": [
                pd.Timestamp("2026-09-10 17:00", tz="UTC"),
                pd.Timestamp("2026-09-17 17:00", tz="UTC"),
            ],
        }
    )


def _depth():
    return pd.DataFrame(
        {
            "club_code": ["BAL", "CIN", "KC", "DEN"],
            "gsis_id": ["HUNT", "BURROW", "MAHOMES", "WILSON"],
            "pos_abb": ["QB", "QB", "QB", "QB"],
            "pos_rank": [1.0, 1.0, 1.0, 1.0],
            "dt": pd.to_datetime(
                ["2026-09-09 12:00", "2026-09-09 12:00", "2026-09-16 12:00", "2026-09-16 12:00"],
                utc=True,
            ),
        }
    )


def _stats():
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [17, 17],
            "team": ["BAL", "CIN"],
            "player_id": ["HUNT", "BURROW"],
            "position": ["QB", "QB"],
            "season_type": ["REG", "REG"],
            "attempts": [30.0, 30.0],
            "sacks_suffered": [2.0, 2.0],
            "passing_epa": [5.0, 8.0],
            "passing_cpoe": [1.0, 1.0],
            "passing_interceptions": [0.0, 1.0],
        }
    )


def _players():
    return pd.DataFrame(
        {
            "gsis_id": ["HUNT", "BURROW", "MAHOMES", "WILSON"],
            "display_name": ["Tyler Huntley", "Joe Burrow", "Patrick Mahomes", "Russell Wilson"],
        }
    )


def _loaders(depth=None, stats=None, players=None, schedules=None):
    return {
        "load_depth_charts": lambda seasons, save=False: depth if depth is not None else _depth(),
        "load_player_stats": lambda seasons, save=False: stats if stats is not None else _stats(),
        "load_players": lambda save=False: players if players is not None else _players(),
        "load_schedules": lambda seasons, save=False: (
            schedules if schedules is not None else _live_schedules()
        ),
    }


NOW = datetime(2026, 9, 8, 12, tzinfo=UTC)


def test_build_advisory_artifact_covers_active_weeks_with_expected_shape():
    schedule = _packaged_schedule_2026()
    frame = build_advisory_artifact(
        schedule,
        [1, 2],
        NOW,
        depth_loader=_loaders()["load_depth_charts"],
        stats_loader=_loaders()["load_player_stats"],
        players_loader=_loaders()["load_players"],
        schedule_loader=_loaders()["load_schedules"],
    )
    assert list(frame.columns) == ARTIFACT_COLUMNS
    assert set(frame["week"]) == {1, 2}
    assert (frame["season"] == 2026).all()
    assert (frame["computed_at"] == pd.Timestamp(NOW)).all()
    row = frame.loc[frame["game_id"] == "2026_01_CIN_BAL"].iloc[0]
    assert row["home_qb"] == "Tyler Huntley"
    assert row["away_qb"] == "Joe Burrow"


def test_build_advisory_artifact_returns_empty_frame_for_no_active_weeks():
    schedule = _packaged_schedule_2026()
    frame = build_advisory_artifact(schedule, [], NOW)
    assert list(frame.columns) == ARTIFACT_COLUMNS
    assert frame.empty


def test_build_advisory_artifact_tolerates_an_unpublished_current_season():
    """Mirrors NflverseStarterProvider's per-season stats tolerance: nflreadpy 404s
    stats_player_week_2026.parquet for the whole preseason and week 1, and the
    advisory must still build from the prior season's rows."""
    schedule = _packaged_schedule_2026()

    def per_season_stats(seasons, save=False):
        if 2026 in seasons:
            raise ConnectionError("404 Client Error: Not Found")
        return _stats()

    frame = build_advisory_artifact(
        schedule,
        [1, 2],
        NOW,
        depth_loader=_loaders()["load_depth_charts"],
        stats_loader=per_season_stats,
        players_loader=_loaders()["load_players"],
        schedule_loader=_loaders()["load_schedules"],
    )
    row = frame.loc[frame["game_id"] == "2026_01_CIN_BAL"].iloc[0]
    assert row["home_qb"] == "Tyler Huntley"


def test_build_advisory_artifact_raises_when_every_season_stats_load_fails():
    schedule = _packaged_schedule_2026()

    def always_fails(seasons, save=False):
        raise ConnectionError("404 Client Error: Not Found")

    with pytest.raises(ConnectionError):
        build_advisory_artifact(
            schedule,
            [1, 2],
            NOW,
            depth_loader=_loaders()["load_depth_charts"],
            stats_loader=always_fails,
            players_loader=_loaders()["load_players"],
            schedule_loader=_loaders()["load_schedules"],
        )


def test_advisory_rows_have_advisory_cols_only_before_extra_columns_are_added():
    schedule = _packaged_schedule_2026()
    frame = build_advisory_artifact(
        schedule,
        [1, 2],
        NOW,
        depth_loader=_loaders()["load_depth_charts"],
        stats_loader=_loaders()["load_player_stats"],
        players_loader=_loaders()["load_players"],
        schedule_loader=_loaders()["load_schedules"],
    )
    assert set(ADVISORY_COLS).issubset(set(frame.columns))


def test_main_dry_run_does_not_write(tmp_path, capsys):
    schedule_path = tmp_path / "schedule_2026.parquet"
    _packaged_schedule_2026().to_parquet(schedule_path, index=False)
    out_path = tmp_path / "starter_advisory.parquet"

    main(
        ["--dry-run", "--schedule", str(schedule_path), "--out", str(out_path)],
        loaders=_loaders(),
        now=NOW,
    )
    assert not out_path.exists()
    assert "dry-run" in capsys.readouterr().out


def test_main_write_creates_the_artifact_atomically(tmp_path, capsys):
    schedule_path = tmp_path / "schedule_2026.parquet"
    _packaged_schedule_2026().to_parquet(schedule_path, index=False)
    out_path = tmp_path / "starter_advisory.parquet"

    main(
        ["--write", "--schedule", str(schedule_path), "--out", str(out_path)],
        loaders=_loaders(),
        now=NOW,
    )
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert list(written.columns) == ARTIFACT_COLUMNS
    assert len(written) == 2
    # No leftover temp files from the atomic write.
    assert list(tmp_path.glob(".starter_advisory.parquet.update-*")) == []
    assert "write complete" in capsys.readouterr().out


def test_main_write_is_idempotent_and_skips_an_unchanged_write(tmp_path, capsys):
    schedule_path = tmp_path / "schedule_2026.parquet"
    _packaged_schedule_2026().to_parquet(schedule_path, index=False)
    out_path = tmp_path / "starter_advisory.parquet"
    argv = ["--write", "--schedule", str(schedule_path), "--out", str(out_path)]

    main(argv, loaders=_loaders(), now=NOW)
    capsys.readouterr()
    main(argv, loaders=_loaders(), now=NOW)
    assert "write skipped: artifact unchanged" in capsys.readouterr().out


def test_main_rejects_a_naive_clock(tmp_path):
    schedule_path = tmp_path / "schedule_2026.parquet"
    _packaged_schedule_2026().to_parquet(schedule_path, index=False)
    out_path = tmp_path / "starter_advisory.parquet"

    with pytest.raises(ValueError, match="timezone-aware"):
        main(
            ["--dry-run", "--schedule", str(schedule_path), "--out", str(out_path)],
            loaders=_loaders(),
            now=datetime(2026, 9, 8, 12),  # noqa: DTZ001 - deliberately naive, asserting rejection
        )
