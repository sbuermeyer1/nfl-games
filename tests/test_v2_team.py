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


def multi_game_pbp_fixture():
    first = pbp_fixture()
    second = first.copy()
    second["game_id"] = "2024_02_ARI_BUF"
    second["week"] = 2
    second["home_team"] = "ARI"
    return pd.concat([first, second], ignore_index=True)


def test_team_game_v2_returns_one_offensive_row_per_game_team_with_key_schema():
    out = team_game_v2(multi_game_pbp_fixture())
    keys = out[["season", "week", "team"]]

    assert len(out) == 4
    assert set(map(tuple, keys.to_numpy())) == {
        (2024, 1, "ARI"),
        (2024, 1, "BUF"),
        (2024, 2, "ARI"),
        (2024, 2, "BUF"),
    }
    assert not keys.duplicated().any()


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


def defensive_team_games():
    allowed = {"A": -1.0, "B": 1.0, "C": -0.5, "D": 0.5}
    matchups = (
        (("A", "B"), ("C", "D")),
        (("A", "C"), ("B", "D")),
        (("A", "D"), ("B", "C")),
    )
    rows = []
    for week, games in enumerate(matchups, start=1):
        for home, away in games:
            for team, opponent, is_home in ((home, away, 1), (away, home, 0)):
                row = {
                    "game_id": f"2024_{week}_{away}_{home}",
                    "season": 2024,
                    "week": week,
                    "team": team,
                    "opponent": opponent,
                    "is_home": is_home,
                }
                row.update({target: allowed[opponent] for target in V2_RATING_TARGETS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_defensive_ratings_remain_higher_is_better_in_both_windows():
    out = v2_team_ratings(defensive_team_games(), [(2024, 4)], 4, 16, 0.6).set_index("team")

    assert out.loc["A", "short_def_epa_play"] > out.loc["B", "short_def_epa_play"]
    assert out.loc["A", "long_def_epa_play"] > out.loc["B", "long_def_epa_play"]


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


def test_team_game_v2_fails_closed_when_no_score_differential_column_exists():
    """Neither alias present means a broken feed, not a game with no score context.

    Failing open silently produced an all-NaN neutral mask, so `neutral_epa` became NaN
    for every team-game while nothing raised.
    """
    pbp = pbp_fixture().drop(columns=["posteam_score_differential"])

    with pytest.raises(ValueError, match="score differential"):
        team_game_v2(pbp)
