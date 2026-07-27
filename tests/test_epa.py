import pytest
import pandas as pd

from nfl_game.ratings import epa


def _pbp_fixture():
    """Two teams, one game. BUF offense: 3 dropbacks (+1.0, +0.0, and a +0.8 scramble),
    1 rush (-0.6). ARI offense: 1 pass (+0.4), 1 rush (+0.2). Plus rows that must be
    excluded.

    Row 3 is the scramble, and it is the only row where the `pass`/`rush` indicators
    disagree with `play_type`: nflverse labels a scramble play_type "run" while setting
    pass=1, because it is a called dropback. Without such a row every pass/rush
    assertion here is satisfied identically by `play_type` and by the indicators, so the
    split could be rewritten to use the wrong column and no test would notice -- which
    is exactly what happened before this row existed.
    """
    return pd.DataFrame(
        {
            "game_id": ["2024_01_ARI_BUF"] * 9,
            "season": [2024] * 9,
            "week": [1] * 9,
            "season_type": ["REG"] * 9,
            "home_team": ["BUF"] * 9,
            "away_team": ["ARI"] * 9,
            "posteam": ["BUF", "BUF", "BUF", "BUF", "ARI", "ARI", "BUF", "BUF", None],
            "defteam": ["ARI", "ARI", "ARI", "ARI", "BUF", "BUF", "ARI", "ARI", None],
            "play_type": ["pass", "pass", "run", "run", "pass", "run", "punt", "pass", None],
            "pass": [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "rush": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "epa": [1.0, 0.0, 0.8, -0.6, 0.4, 0.2, 3.0, None, 0.0],
            "success": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, None, 0.0],
        }
    )


def test_one_row_per_offense_per_game():
    out = epa.team_game_epa(_pbp_fixture())
    assert len(out) == 2
    assert set(out["team"]) == {"BUF", "ARI"}


def test_excludes_special_teams_and_null_epa():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    # the punt (epa 3.0) and the null-epa pass are both dropped, leaving BUF's two
    # thrown passes plus the scramble
    assert buf["n_pass"] == 3
    assert buf["n_rush"] == 1


def test_pass_rush_split_uses_indicator_columns():
    """The split must key on nflverse's `pass`/`rush` indicators, not on `play_type`,
    so that scrambles and sacks are counted as the dropbacks they are.

    BUF's scramble is play_type "run" with pass=1, so each assertion below separates
    the two possible readings rather than merely being consistent with both:
      * counting it as a rush (keying off play_type for the rush side) makes n_rush 2
        and epa_rush 0.1 rather than -0.6;
      * dropping it from the pass side (keying off play_type there) makes n_pass 2 and
        epa_pass 0.5 rather than 0.6;
      * dropping it from the scrimmage-play filter entirely makes epa_play 0.4/3.
    """
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    assert buf["n_pass"] == 3
    assert buf["n_rush"] == 1
    assert buf["epa_pass"] == pytest.approx(0.6)       # (1.0 + 0.0 + 0.8) / 3
    assert buf["epa_rush"] == -0.6
    assert buf["epa_play"] == pytest.approx(0.3)       # (1.0 + 0.0 + 0.8 - 0.6) / 4


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
