import pandas as pd
import pytest

from nfl_game.data import nfl


def test_seasons_label_single():
    assert nfl._seasons_label([2024]) == "2024"


def test_seasons_label_range():
    assert nfl._seasons_label([2024, 2016, 2020]) == "2016-2024"


def test_load_pbp_converts_and_saves(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"game_id": ["2024_01_ARI_BUF"], "epa": [0.5]})

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    out = nfl.load_pbp([2024])

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["game_id", "epa"]
    assert (tmp_path / "pbp_2024.parquet").exists()


def test_load_pbp_can_skip_save(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"game_id": ["x"], "epa": [0.1]})

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    nfl.load_pbp([2024], save=False)

    assert list(tmp_path.iterdir()) == []


def test_load_ngs_rejects_bad_stat_type():
    with pytest.raises(ValueError, match="stat_type"):
        nfl.load_ngs([2024], stat_type="kicking")


# The three tests below pin that each loader actually APPLIES normalize_team_codes, and
# applies it to the right columns. teams.py has its own suite proving the mapping works;
# these prove it is wired in. Without them, deleting the normalize_team_codes call from a
# loader -- or dropping posteam/defteam from PBP_TEAM_COLS, which breaks the ratings join
# while leaving the schedule join intact -- leaves the whole suite green, which is exactly
# how the LA/LAR mismatch survived undetected for the life of the project.


def test_load_schedules_normalizes_team_codes(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "season": [2016, 2016],
                    "home_team": ["OAK", "LA"],
                    "away_team": ["SD", "STL"],
                }
            )

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_schedules", lambda seasons: FakePolars())

    out = nfl.load_schedules(save=False)

    assert list(out["home_team"]) == ["LV", "LAR"]
    assert list(out["away_team"]) == ["LAC", "LAR"]


def test_load_schedules_forwards_requested_seasons(monkeypatch):
    calls = []

    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"season": [2026], "home_team": ["KC"], "away_team": ["BUF"]})

    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_schedules",
        lambda seasons: calls.append(seasons) or FakePolars(),
    )
    nfl.load_schedules([2026], save=False)
    assert calls == [[2026]]


def test_load_pbp_normalizes_team_codes(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "posteam": ["OAK", "LA"],
                    "defteam": ["LA", "SD"],
                    "home_team": ["SD", "STL"],
                    "away_team": ["STL", "OAK"],
                    "epa": [0.5, -0.2],
                }
            )

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    out = nfl.load_pbp([2016], save=False)

    # posteam/defteam carry the ratings join; home_team/away_team carry the schedule join.
    # Asserting all four is what kills a mutation that narrows PBP_TEAM_COLS to either pair.
    assert list(out["posteam"]) == ["LV", "LAR"]
    assert list(out["defteam"]) == ["LAR", "LAC"]
    assert list(out["home_team"]) == ["LAC", "LAR"]
    assert list(out["away_team"]) == ["LAR", "LV"]


def test_load_ngs_normalizes_team_codes(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "season": [2016, 2016],
                    "week": [1, 1],
                    "team_abbr": ["OAK", "LA"],
                }
            )

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_nextgen_stats",
        lambda seasons, stat_type: FakePolars(),
    )

    out = nfl.load_ngs([2016], stat_type="passing", save=False)

    assert list(out["team_abbr"]) == ["LV", "LAR"]
