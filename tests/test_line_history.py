from __future__ import annotations

import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from nfl_game.data.line_history import (
    _COMMITS_API,
    GAMES_CSV_PATH,
    NFLDATA_REPO,
    collect_game_line_history,
    commit_at,
    game_snapshot_timestamps,
    games_at,
    line_snapshot,
    line_snapshot_for_games,
    snapshot_timestamps,
)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2024_10_ATL_NO",
                "season": 2024,
                "week": 10,
                "kickoff_at": "2024-11-10T18:00:00Z",
            },
            {
                "game_id": "2024_10_DEN_KC",
                "season": 2024,
                "week": 10,
                "kickoff_at": "2024-11-10T21:25:00Z",
            },
            {
                "game_id": "2024_10_MIA_LA",
                "season": 2024,
                "week": 10,
                "kickoff_at": "2024-11-11T01:20:00Z",
            },
            {
                "game_id": "2024_11_BUF_IND",
                "season": 2024,
                "week": 11,
                "kickoff_at": "2024-11-17T18:00:00Z",
            },
        ]
    )


def _games_csv(rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False)


def _snapshot_rows() -> list[dict[str, object]]:
    return [
        # Already final at snapshot time: its "line" is the closing number.
        {
            "game_id": "2024_09_OLD_GAME",
            "season": 2024,
            "week": 9,
            "spread_line": -3.0,
            "total_line": 44.0,
            "result": -7.0,
        },
        {
            "game_id": "2024_10_ATL_NO",
            "season": 2024,
            "week": 10,
            "spread_line": -1.0,
            "total_line": 43.5,
            "result": "",
        },
        {
            "game_id": "2024_10_DEN_KC",
            "season": 2024,
            "week": 10,
            "spread_line": 9.5,
            "total_line": 41.0,
            "result": "",
        },
        # In the requested week but no line posted yet.
        {
            "game_id": "2024_10_MIA_LA",
            "season": 2024,
            "week": 10,
            "spread_line": "",
            "total_line": "",
            "result": "",
        },
    ]


def test_snapshot_timestamp_precedes_every_kickoff_in_its_week():
    stamps = snapshot_timestamps(_schedule(), days_before=5)

    assert set(stamps) == {(2024, 10), (2024, 11)}
    schedule = _schedule()
    for (season, week), stamp in stamps.items():
        kickoffs = pd.to_datetime(
            schedule.loc[schedule["season"].eq(season) & schedule["week"].eq(week), "kickoff_at"],
            utc=True,
        )
        assert stamp < kickoffs.min()


def test_a_larger_days_before_moves_the_snapshot_earlier():
    early = snapshot_timestamps(_schedule(), days_before=7)[(2024, 10)]
    late = snapshot_timestamps(_schedule(), days_before=2)[(2024, 10)]

    assert early < late
    assert (late - early) == pd.Timedelta(days=5)


def test_games_already_played_at_snapshot_time_are_refused():
    """A finished game's spread IS the closing line; taking it as an early line is a leak."""
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())

    out = line_snapshot(frame, season=2024, week=9)

    assert out.empty


def test_unplayed_games_yield_their_line_and_an_unposted_line_is_nan_not_zero():
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())

    out = line_snapshot(frame, season=2024, week=10)

    assert list(out["game_id"]) == ["2024_10_ATL_NO", "2024_10_DEN_KC", "2024_10_MIA_LA"]
    assert out.loc[out["game_id"].eq("2024_10_ATL_NO"), "early_spread_line"].iloc[0] == -1.0
    assert out.loc[out["game_id"].eq("2024_10_DEN_KC"), "early_total_line"].iloc[0] == 41.0
    # No line posted yet must stay missing: a zero would read as a pick'em.
    assert np.isnan(out.loc[out["game_id"].eq("2024_10_MIA_LA"), "early_spread_line"].iloc[0])


def test_games_at_requests_the_versioned_path_and_parses_numerics():
    seen: dict[str, str] = {}

    def fetch(url: str) -> bytes:
        seen["url"] = url
        return _games_csv(_snapshot_rows()).encode()

    frame = games_at("abc123", fetch=fetch)

    assert "abc123" in seen["url"]
    assert GAMES_CSV_PATH in seen["url"]
    assert frame["spread_line"].dtype.kind == "f"


def test_commit_at_asks_for_the_newest_commit_at_or_before_the_timestamp():
    seen: dict[str, str] = {}

    def fetch(url: str) -> bytes:
        seen["url"] = url
        return b'[{"sha": "feedface", "commit": {"committer": {"date": "2024-11-05T12:00:00Z"}}}]'

    sha = commit_at(pd.Timestamp("2024-11-05T12:00:00Z"), fetch=fetch)

    assert sha == "feedface"
    assert (
        "until=2024-11-05T12%3A00%3A00Z" in seen["url"]
        or "until=2024-11-05T12:00:00Z" in seen["url"]
    )
    assert "per_page=1" in seen["url"]


def test_commit_at_raises_when_no_commit_exists_before_the_timestamp():
    with pytest.raises(ValueError, match="no nflverse commit"):
        commit_at(pd.Timestamp("2015-01-01T00:00:00Z"), fetch=lambda url: b"[]")


def test_per_game_anchoring_gives_each_game_its_own_kickoff_minus_the_lead():
    stamps = game_snapshot_timestamps(_schedule(), days_before=5)

    assert stamps == {
        "2024_10_ATL_NO": pd.Timestamp("2024-11-05T18:00:00Z"),
        "2024_10_DEN_KC": pd.Timestamp("2024-11-05T21:25:00Z"),
        "2024_10_MIA_LA": pd.Timestamp("2024-11-06T01:20:00Z"),
        "2024_11_BUF_IND": pd.Timestamp("2024-11-12T18:00:00Z"),
    }


def test_per_game_anchoring_separates_games_the_week_anchor_collapses():
    """The defect this exists to fix: one stamp per week understates every later game's lead."""
    per_game = game_snapshot_timestamps(_schedule(), days_before=5)
    per_week = snapshot_timestamps(_schedule(), days_before=5)

    week_10 = ["2024_10_ATL_NO", "2024_10_DEN_KC", "2024_10_MIA_LA"]
    assert len({per_game[game_id] for game_id in week_10}) == 3

    schedule = _schedule().set_index("game_id")
    for game_id in week_10:
        kickoff = pd.to_datetime(schedule.loc[game_id, "kickoff_at"], utc=True)
        assert kickoff - per_game[game_id] == pd.Timedelta(days=5)

    # Only the week's earliest game agrees with the week anchor. Under the week anchor the last
    # game of the week is 5d + 7h20m out, which is the understatement this change removes.
    week_stamp = per_week[(2024, 10)]
    assert per_game["2024_10_ATL_NO"] == week_stamp
    last_kickoff = pd.to_datetime(schedule.loc["2024_10_MIA_LA", "kickoff_at"], utc=True)
    assert last_kickoff - week_stamp == pd.Timedelta(days=5, hours=7, minutes=20)


def test_per_game_anchoring_scales_with_the_lead():
    early = game_snapshot_timestamps(_schedule(), days_before=7)["2024_10_DEN_KC"]
    late = game_snapshot_timestamps(_schedule(), days_before=2)["2024_10_DEN_KC"]

    assert (late - early) == pd.Timedelta(days=5)


def test_per_game_anchoring_refuses_duplicate_game_ids():
    doubled = pd.concat([_schedule(), _schedule().head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate game_id"):
        game_snapshot_timestamps(doubled, days_before=5)


def test_per_game_anchoring_refuses_a_missing_kickoff_column():
    with pytest.raises(ValueError, match="kickoff_at"):
        game_snapshot_timestamps(_schedule().drop(columns=["kickoff_at"]), days_before=5)


def test_line_snapshot_for_games_returns_only_the_requested_games():
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())

    snapshot = line_snapshot_for_games(frame, game_ids=["2024_10_DEN_KC"])

    assert list(snapshot["game_id"]) == ["2024_10_DEN_KC"]


def test_line_snapshot_for_games_still_refuses_a_game_already_played():
    """The honesty rule must survive the change of anchoring, not just the change of filter."""
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())
    played = frame.loc[frame["result"].notna(), "game_id"].tolist()
    assert played, "fixture must contain a played game or this test cannot fail"

    snapshot = line_snapshot_for_games(frame, game_ids=played)

    assert snapshot.empty


def test_line_snapshot_for_games_reports_an_unposted_line_as_nan_not_zero():
    frame = games_at("deadbeef", fetch=lambda url: _games_csv(_snapshot_rows()).encode())

    snapshot = line_snapshot_for_games(frame, game_ids=list(frame["game_id"]))
    unposted = snapshot.loc[snapshot["early_spread_line"].isna()]

    assert not unposted.empty, "fixture must contain an unpriced game or this test cannot fail"
    assert not (snapshot["early_spread_line"] == 0).any()


def _fetch_for(rows_by_sha: dict[str, list[dict[str, object]]], commits: dict[str, str]):
    """Serve the commits API and raw file from fixed fixtures, recording each snapshot fetched."""

    def fetch(url: str) -> bytes:
        if url.startswith(_COMMITS_API.format(repo=NFLDATA_REPO)):
            until = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["until"][0]
            return json.dumps([{"sha": commits[until]}]).encode()
        sha = url.split("/")[5]
        return _games_csv(rows_by_sha[sha]).encode()

    return fetch


def test_collect_game_line_history_fetches_one_snapshot_per_distinct_kickoff():
    schedule = _schedule().loc[lambda f: f["season"].eq(2024) & f["week"].eq(10)]
    stamps = game_snapshot_timestamps(schedule, days_before=5)
    commits = {stamp.strftime("%Y-%m-%dT%H:%M:%SZ"): f"sha{i}" for i, stamp in enumerate(sorted(stamps.values()))}
    rows_by_sha = {sha: _snapshot_rows() for sha in commits.values()}
    seen: list[dict[str, object]] = []

    frame = collect_game_line_history(
        schedule,
        days_before=5,
        fetch=_fetch_for(rows_by_sha, commits),
        observer=seen.append,
    )

    # Three distinct kickoffs in the week means three snapshots, not one.
    assert len(seen) == 3
    assert sorted(frame["game_id"]) == ["2024_10_ATL_NO", "2024_10_DEN_KC", "2024_10_MIA_LA"]
    assert frame["snapshot_at"].nunique() == 3


def test_collect_game_line_history_gives_each_game_its_own_snapshot_time():
    schedule = _schedule().loc[lambda f: f["season"].eq(2024) & f["week"].eq(10)]
    stamps = game_snapshot_timestamps(schedule, days_before=5)
    commits = {stamp.strftime("%Y-%m-%dT%H:%M:%SZ"): f"sha{i}" for i, stamp in enumerate(sorted(stamps.values()))}
    rows_by_sha = {sha: _snapshot_rows() for sha in commits.values()}

    frame = collect_game_line_history(
        schedule, days_before=5, fetch=_fetch_for(rows_by_sha, commits)
    ).set_index("game_id")

    for game_id, stamp in stamps.items():
        assert frame.loc[game_id, "snapshot_at"] == stamp
    assert frame.loc["2024_10_ATL_NO", "season"] == 2024
    assert frame.loc["2024_10_ATL_NO", "week"] == 10


def test_collect_game_line_history_returns_the_empty_schema_for_no_games():
    frame = collect_game_line_history(_schedule().head(0), days_before=5, fetch=lambda url: b"")

    assert list(frame.columns) == [
        "game_id",
        "early_spread_line",
        "early_total_line",
        "season",
        "week",
        "snapshot_at",
        "snapshot_sha",
    ]
    assert frame.empty
