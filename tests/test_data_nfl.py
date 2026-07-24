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
