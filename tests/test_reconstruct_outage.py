"""The 2026 week 2-3 outage reconstruction: its method, its artifact, and its isolation.

The reconstructed records must (a) be exactly what the tracker would have published from
information that existed then, and (b) never be readable as official live records.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts import reconstruct_outage_2026 as rc
from scripts import update_live_tracker

from nfl_game.tracking.ledger import validate_ledger
from nfl_game.web.tracker_service import TrackerService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed"
UTC = "UTC"


@pytest.fixture(scope="module")
def reconstructed():
    return pd.read_parquet(PROCESSED / "tracker_reconstructed.parquet")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PROCESSED / "tracker_reconstructed_manifest.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def official():
    return pd.read_parquet(PROCESSED / "tracker_ledger.parquet")


def ts(value):
    return pd.Timestamp(value, tz=UTC)


# --- method -------------------------------------------------------------------------------


def test_commit_lookup_takes_the_newest_commit_at_or_before_the_moment():
    commits = [(ts("2026-09-15 10:00"), "a"), (ts("2026-09-15 11:00"), "b")]
    assert rc.commit_at_or_before(commits, ts("2026-09-15 10:59")) == "a"
    assert rc.commit_at_or_before(commits, ts("2026-09-15 11:00")) == "b"
    assert rc.commit_at_or_before(commits, ts("2026-09-16 00:00")) == "b"
    with pytest.raises(ValueError, match="no nfldata commit"):
        rc.commit_at_or_before(commits, ts("2026-09-15 09:59"))


def schedule(rows):
    return pd.DataFrame(
        [
            {"week": week, "kickoff_at": ts(kickoff), "result": result, "total": total}
            for week, kickoff, result, total in rows
        ]
    )


def test_a_week_counts_as_known_only_once_every_game_is_final():
    frame = schedule(
        [
            (1, "2026-09-10 00:20", 3.0, 40.0),
            (1, "2026-09-15 00:15", 7.0, 45.0),
            (2, "2026-09-18 00:15", None, None),
        ]
    )
    # Monday night kicked off 00:15; FINALIZATION_DELAY is six hours.
    assert rc.final_weeks(frame, ts("2026-09-15 06:14")) == set()
    assert rc.final_weeks(frame, ts("2026-09-15 06:15")) == {1}


def test_truncation_drops_only_unknown_2026_weeks():
    frame = pd.DataFrame(
        {"season": [2025, 2025, 2026, 2026, 2026], "week": [17, 18, 1, 2, 3], "x": range(5)}
    )
    kept = rc.truncate_to_weeks(frame, {1})
    assert list(zip(kept["season"], kept["week"], strict=True)) == [
        (2025, 17),
        (2025, 18),
        (2026, 1),
    ]


def test_replay_uses_the_latest_refresh_committed_before_each_run(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def features_at(committed):
        seen.append(committed)
        raise Stop

    refreshes = [ts("2026-09-15 15:11"), ts("2026-09-16 15:08")]
    for run, expected in (
        (ts("2026-09-16 15:07"), refreshes[0]),
        (ts("2026-09-16 15:08"), refreshes[1]),
    ):
        seen.clear()
        with pytest.raises(Stop):
            rc.replay([run], refreshes, lambda moment: None, features_at)
        assert seen == [expected]


def test_replay_skips_runs_before_the_first_refresh():
    def never(_):
        raise AssertionError("no refresh had committed yet")

    live, audit = rc.replay([ts("2026-09-15 12:00")], [ts("2026-09-15 15:11")], never, never)
    assert live.empty
    assert audit == []


def test_records_outside_the_outage_weeks_are_dropped_and_relabelled(official):
    live = official.loc[official["record_type"].eq("live")].copy()
    assert not live.empty
    week_one = live.copy()
    shifted = live.copy()
    shifted["week"] = 2
    shifted["game_id"] = shifted["game_id"].str.replace("2026_01_", "2026_02_")
    kept = rc.reconstructed_records(pd.concat([week_one, shifted], ignore_index=True))
    assert set(kept["week"]) == {2}
    assert kept["record_type"].eq("reconstructed").all()


# --- the committed artifact ---------------------------------------------------------------


def test_artifact_covers_every_week_two_and_three_game_once(reconstructed):
    schedule_2026 = pd.read_parquet(PROCESSED / "schedule_2026.parquet")
    expected = set(schedule_2026.loc[schedule_2026["week"].isin([2, 3]), "game_id"].astype(str))
    assert set(reconstructed["game_id"]) == expected
    assert reconstructed["game_id"].is_unique
    assert reconstructed["record_type"].eq("reconstructed").all()
    validate_ledger(reconstructed)


def test_every_pick_was_published_before_kickoff_and_within_the_lock(reconstructed):
    lead = reconstructed["kickoff_at"] - reconstructed["published_at"]
    assert (lead > pd.Timedelta(0)).all()
    # PUBLISH_BEFORE is five days: nothing may publish earlier than that.
    assert (lead <= pd.Timedelta(days=5)).all()


def test_every_pick_is_graded_at_its_published_line_with_a_close(reconstructed):
    for kind in ("spread", "total"):
        assert reconstructed[f"{kind}_publication_status"].eq("published").all()
        assert reconstructed[f"published_{kind}_line"].notna().all()
        assert reconstructed[f"closing_{kind}_line"].notna().all()
        assert reconstructed[f"{kind}_grade"].isin({"win", "loss", "push"}).all()


def test_manifest_records_the_verification_and_every_publication(reconstructed, manifest):
    assert manifest["record_type"] == "reconstructed"
    assert manifest["records"] == len(reconstructed)
    assert manifest["verification"]["compared_with"] == rc.VERIFY_COMMIT
    assert manifest["verification"]["max_abs_feature_diff"] <= rc.VERIFY_TOLERANCE
    published = [game for entry in manifest["publications"] for game in entry["published"]]
    assert sorted(published) == sorted(reconstructed["game_id"])
    for entry in manifest["publications"]:
        # Each publication used a refresh that had committed before the tracker run.
        assert pd.Timestamp(entry["features_from_refresh"]) <= pd.Timestamp(entry["tracker_run"])


# --- isolation from the official record ---------------------------------------------------


def test_official_ledger_holds_no_reconstructed_records(official):
    assert set(official["record_type"]) == {"backtest", "live"}


def test_reconstructed_rows_need_a_lifecycle_kickoff_like_live_rows(reconstructed):
    broken = reconstructed.copy()
    broken.loc[0, "current_kickoff_at"] = pd.NaT
    with pytest.raises(ValueError, match="require a current kickoff"):
        validate_ledger(broken)


def test_the_live_updater_refuses_a_ledger_holding_reconstructed_rows(
    tmp_path, official, reconstructed
):
    ledger_path = tmp_path / "tracker_ledger.parquet"
    pd.concat([official, reconstructed], ignore_index=True).to_parquet(ledger_path, index=False)
    before = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="only backtest and live rows"):
        update_live_tracker.main(
            ["--ledger", str(ledger_path), "--season", "2026", "--write"],
            loader=lambda seasons, save=False: pytest.fail("must refuse before loading"),
        )
    assert ledger_path.read_bytes() == before


def test_service_keeps_reconstructed_records_out_of_the_live_selection(official, reconstructed):
    service = TrackerService(official, reconstructed)
    options = service.options()
    assert options["record_types"] == ["backtest", "live", "reconstructed"]
    assert options["reconstructed_available"] is True
    # Loading the artifact must not move the live or historical numbers at all.
    alone = TrackerService(official)
    for record_type in ("live", "backtest"):
        assert service.summary(record_type, "all") == alone.summary(record_type, "all")
    rebuilt = service.summary("reconstructed", "all")
    assert rebuilt["record_type"] == "reconstructed"
    assert rebuilt["closing_line"] is not None
    assert {row["game_id"] for row in service.records("reconstructed", 2026)} == set(
        reconstructed["game_id"]
    )
    assert not {row["game_id"] for row in service.records("live", 2026)} & set(
        reconstructed["game_id"]
    )


def test_service_without_the_artifact_offers_no_reconstructed_tab(official):
    options = TrackerService(official).options()
    assert options["record_types"] == ["backtest", "live"]
    assert "reconstructed_available" not in options


def test_service_rejects_reconstructed_rows_inside_the_official_ledger(official, reconstructed):
    with pytest.raises(ValueError, match="only backtest and live"):
        TrackerService(pd.concat([official, reconstructed], ignore_index=True))


def test_service_rejects_live_rows_in_the_reconstructed_artifact(official):
    live = official.loc[official["record_type"].eq("live")]
    with pytest.raises(ValueError, match="only reconstructed records"):
        TrackerService(official, live)
