import pandas as pd
import pytest

from nfl_game.ratings.style import team_game_style
from nfl_game.ratings.v2_team import team_game_v2


def test_live_score_differential_column_drives_neutral_pbp_features():
    """Renaming the live score field back to a retired alias must break this test."""
    pbp = pd.DataFrame(
        [
            {
                "game_id": "2025_01_KC_BUF",
                "season": 2025,
                "week": 1,
                "season_type": "REG",
                "posteam": "BUF",
                "defteam": "KC",
                "home_team": "BUF",
                "pass": 1,
                "rush": 0,
                "qb_dropback": 1,
                "sack": 0,
                "down": 1,
                "qtr": 1,
                "score_differential": 0,
                "epa": 0.5,
                "success": 1,
                "yards_gained": 20,
                "game_seconds_remaining": 900,
                "yardline_100": 75,
                "drive": 1,
                "play_id": 1,
                "interception": 0,
                "fumble_lost": 0,
                "special_teams_play": 0,
            }
        ]
    )

    rating = team_game_v2(pbp).iloc[0]
    style = team_game_style(pbp).iloc[0]

    assert rating["neutral_epa"] == pytest.approx(0.5)
    assert style["neutral_pass_rate"] == pytest.approx(1.0)
