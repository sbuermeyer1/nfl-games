import pandas as pd
import pytest

from nfl_game.ratings.v2_team import V2_RATING_TARGETS, team_game_v2, v2_team_ratings


def pbp_fixture():
    """One BUF-ARI game with deliberately distinct situational play classes."""
    rows = [
        # BUF's three pass attempts include one 20-yard explosive; its sack is a fourth dropback.
        ("BUF", "ARI", 1, 1, 0, 0, 1, 0, 0.20, 1, 20, 0, 0),
        ("BUF", "ARI", 2, 3, 0, 0, 1, 0, 0.10, 0, 5, 0, 0),
        ("BUF", "ARI", 3, 2, 8, 0, 0, 1, 0.20, 1, 0, 10, 0),
        ("BUF", "ARI", 3, 3, 0, 0, 1, 0, 0.10, 0, 0, 0, 1),
        ("BUF", "ARI", 4, 1, 0, 0, 1, 0, 0.20, 1, 5, 0, 0),
        # A nine-point late-down rush must not enter neutral EPA.
        ("BUF", "ARI", 2, 3, 9, 0, 0, 1, 0.90, 1, 25, 25, 0),
        # ARI's offense supplies the opponent's offensive row for the game.
        ("ARI", "BUF", 1, 1, 0, 0, 1, 0, -0.10, 0, 4, 0, 0),
        ("ARI", "BUF", 2, 2, 0, 0, 0, 1, -0.20, 0, 0, 4, 0),
    ]
    columns = [
        "posteam", "defteam", "qtr", "down", "posteam_score_differential", "unused",
        "pass", "rush", "epa", "success", "yards_gained", "rush_yards", "sack",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out["game_id"] = "2024_01_ARI_BUF"
    out["season"] = 2024
    out["week"] = 1
    out["season_type"] = "REG"
    out["home_team"] = "BUF"
    return out


def test_neutral_and_early_down_filters_are_fixed():
    out = team_game_v2(pbp_fixture()).set_index("team")

    assert out.loc["BUF", "early_down_epa"] == pytest.approx(0.20)
    assert out.loc["BUF", "neutral_epa"] == pytest.approx(0.15)


def test_explosive_and_sack_rates_use_declared_denominators():
    out = team_game_v2(pbp_fixture()).set_index("team")

    assert out.loc["BUF", "explosive_pass_rate"] == pytest.approx(1 / 3)
    assert out.loc["BUF", "sack_rate"] == pytest.approx(1 / 4)


def team_games():
    rows = []
    for week, a_value in ((1, -1.0), (2, -1.0), (3, 1.0)):
        for team, opponent, value in (("A", "B", a_value), ("B", "A", -a_value)):
            row = {
                "game_id": f"2024_{week}_{team}_{opponent}",
                "season": 2024,
                "week": week,
                "team": team,
                "opponent": opponent,
                "is_home": int(team == "A"),
            }
            row.update({target: value for target in V2_RATING_TARGETS})
            rows.append(row)
    return pd.DataFrame(rows)


def future_game_with_extreme_epa():
    row = team_games().iloc[[0]].copy()
    row["game_id"] = "2024_04_C_D"
    row["week"] = 4
    row["team"] = "C"
    row["opponent"] = "D"
    row[list(V2_RATING_TARGETS)] = 1_000_000.0
    return row


def test_short_ratings_react_more_strongly_to_recent_game():
    out = v2_team_ratings(team_games(), [(2024, 4)], 4, 16, 0.6).set_index("team")

    assert out.loc["A", "short_off_epa_play"] > out.loc["A", "long_off_epa_play"]


def test_all_rating_columns_use_window_prefixes():
    out = v2_team_ratings(team_games(), [(2024, 4)], 4, 16, 0.6)

    assert list(out.columns[:3]) == ["season", "week", "team"]
    assert all(column.startswith(("short_", "long_")) for column in out.columns[3:])
    assert len(out.columns) == 3 + len(V2_RATING_TARGETS) * 4


def test_future_rows_cannot_change_prior_rating():
    base = v2_team_ratings(team_games(), [(2024, 4)], 4, 16, 0.6)
    poisoned = pd.concat([team_games(), future_game_with_extreme_epa()], ignore_index=True)

    actual = v2_team_ratings(poisoned, [(2024, 4)], 4, 16, 0.6)

    pd.testing.assert_frame_equal(base, actual)
