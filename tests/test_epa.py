import pytest
import pandas as pd

from nfl_game.ratings import epa


def _pbp_fixture():
    """Two teams, one game. BUF offense: 2 pass (+1.0, +0.0), 1 rush (-0.6).
    ARI offense: 1 pass (+0.4), 1 rush (+0.2). Plus rows that must be excluded."""
    return pd.DataFrame(
        {
            "game_id": ["2024_01_ARI_BUF"] * 8,
            "season": [2024] * 8,
            "week": [1] * 8,
            "season_type": ["REG"] * 8,
            "home_team": ["BUF"] * 8,
            "away_team": ["ARI"] * 8,
            "posteam": ["BUF", "BUF", "BUF", "ARI", "ARI", "BUF", "BUF", None],
            "defteam": ["ARI", "ARI", "ARI", "BUF", "BUF", "ARI", "ARI", None],
            "play_type": ["pass", "pass", "run", "pass", "run", "punt", "pass", None],
            "pass": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "rush": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "epa": [1.0, 0.0, -0.6, 0.4, 0.2, 3.0, None, 0.0],
            "success": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, None, 0.0],
        }
    )


def test_one_row_per_offense_per_game():
    out = epa.team_game_epa(_pbp_fixture())
    assert len(out) == 2
    assert set(out["team"]) == {"BUF", "ARI"}


def test_excludes_special_teams_and_null_epa():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    # the punt (epa 3.0) and the null-epa pass are both dropped
    assert buf["n_pass"] == 2
    assert buf["n_rush"] == 1


def test_pass_rush_split_uses_indicator_columns():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    assert buf["epa_pass"] == 0.5          # (1.0 + 0.0) / 2
    assert buf["epa_rush"] == -0.6
    assert buf["epa_play"] == pytest.approx(0.4 / 3)   # (1.0 + 0.0 - 0.6) / 3


def test_opponent_and_home_flag():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    ari = out[out["team"] == "ARI"].iloc[0]
    assert buf["opponent"] == "ARI"
    assert buf["is_home"] == 1
    assert ari["opponent"] == "BUF"
    assert ari["is_home"] == 0


def test_success_rate():
    out = epa.team_game_epa(_pbp_fixture())
    ari = out[out["team"] == "ARI"].iloc[0]
    assert ari["success_rate"] == 1.0


def test_filters_to_regular_season():
    df = _pbp_fixture()
    df["season_type"] = "POST"
    out = epa.team_game_epa(df)
    assert out.empty
