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


class FakePolars:
    """Small stand-in that proves loaders cross the Polars-to-pandas boundary."""

    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame


@pytest.mark.parametrize(
    ("loader_name", "source_name", "team_columns", "filename"),
    [
        ("load_player_stats", "load_player_stats", ["team", "opponent_team"], "player_stats_2024.parquet"),
        ("load_rosters_weekly", "load_rosters_weekly", ["team"], "rosters_weekly_2024.parquet"),
        ("load_depth_charts", "load_depth_charts", ["team"], "depth_charts_2024.parquet"),
        ("load_snap_counts", "load_snap_counts", ["team", "opponent"], "snap_counts_2024.parquet"),
        ("load_ftn_charting", "load_ftn_charting", [], "ftn_charting_2024.parquet"),
    ],
)
def test_new_season_loaders_convert_normalize_and_cache(
    monkeypatch, tmp_path, loader_name, source_name, team_columns, filename
):
    calls = []
    frame = pd.DataFrame({column: ["OAK"] for column in team_columns})
    if source_name == "load_depth_charts":
        frame["dt"] = ["2024-09-01T12:00:00-04:00"]
    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        source_name,
        lambda *args, **kwargs: calls.append((args, kwargs)) or FakePolars(frame),
    )

    out = getattr(nfl, loader_name)([2024])

    assert isinstance(out, pd.DataFrame)
    expected = [(([2024],), {"summary_level": "week"})]
    if source_name != "load_player_stats":
        expected = [(([2024],), {})]
    assert calls == expected
    assert all(out.loc[0, column] == "LV" for column in team_columns)
    if source_name == "load_depth_charts":
        assert str(out.loc[0, "dt"].tz) == "UTC"
    assert (tmp_path / filename).exists()


def test_load_players_converts_without_a_season_and_caches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_players",
        lambda: calls.append(()) or FakePolars(pd.DataFrame({"gsis_id": ["00-003"]})),
    )

    out = nfl.load_players()

    assert isinstance(out, pd.DataFrame)
    assert calls == [()]
    assert (tmp_path / "players.parquet").exists()


@pytest.mark.parametrize("stat_type", ["pass", "rush", "rec", "def"])
def test_load_pfr_advstats_forwards_week_level(monkeypatch, tmp_path, stat_type):
    calls = []
    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_pfr_advstats",
        lambda seasons, stat_type, summary_level: calls.append(
            (seasons, stat_type, summary_level)
        )
        or FakePolars(pd.DataFrame({"team": ["OAK"], "opponent": ["SD"]})),
    )

    out = nfl.load_pfr_advstats([2024], stat_type, save=True)

    assert isinstance(out, pd.DataFrame)
    assert calls == [([2024], stat_type, "week")]
    assert out.loc[0, "team"] == "LV"
    assert out.loc[0, "opponent"] == "LAC"
    assert (tmp_path / f"pfr_{stat_type}_2024.parquet").exists()


def test_load_pfr_advstats_rejects_bad_stat_type():
    with pytest.raises(ValueError, match="stat_type"):
        nfl.load_pfr_advstats([2024], "kicking", save=False)


@pytest.mark.parametrize(
    ("loader_name", "source_name", "arguments"),
    [
        ("load_player_stats", "load_player_stats", ([2024],)),
        ("load_players", "load_players", ()),
        ("load_rosters_weekly", "load_rosters_weekly", ([2024],)),
        ("load_depth_charts", "load_depth_charts", ([2024],)),
        ("load_snap_counts", "load_snap_counts", ([2024],)),
        ("load_pfr_advstats", "load_pfr_advstats", ([2024], "pass")),
        ("load_ftn_charting", "load_ftn_charting", ([2024],)),
    ],
)
def test_new_loaders_can_skip_cache_writes(monkeypatch, tmp_path, loader_name, source_name, arguments):
    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        source_name,
        lambda *args, **kwargs: FakePolars(pd.DataFrame()),
    )

    getattr(nfl, loader_name)(*arguments, save=False)

    assert list(tmp_path.iterdir()) == []


def test_load_players_without_season_can_skip_cache_writes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_players",
        lambda: calls.append(()) or FakePolars(pd.DataFrame({"gsis_id": ["00-003"]})),
    )

    out = nfl.load_players(save=False)

    assert isinstance(out, pd.DataFrame)
    assert calls == [()]
    assert list(tmp_path.iterdir()) == []
