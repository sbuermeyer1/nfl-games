import pandas as pd
import pytest

from nfl_game.ratings.style import (
    STYLE_FEATURE_COLS,
    TURNOVER_PRIOR_PLAYS,
    style_features_for_targets,
    team_game_style,
)


def style_pbp():
    """Two offenses with rows that distinguish every declared style denominator."""
    rows = [
        # BUF drive 1: the only valid same-drive snap delta is 27 seconds.
        ("BUF", "ARI", 1, 1, 900, 0, 1, 0, 1, 0, 5, 80, 0, 0, 0, 0, 0, 0, None),
        ("BUF", "ARI", 1, 1, 873, 0, 0, 1, 0, 1, 12, 75, 0, 0, 0, 0, 0, 0, None),
        # The 173-second gap is a timeout/quarter-break-style gap and must be excluded.
        ("BUF", "ARI", 1, 1, 700, 0, 1, 0, 1, 0, 25, 60, 0, 0, 0, 0, 0, 0, None),
        # A nine-point score differential and Q4 do not belong in neutral pass rate.
        ("BUF", "ARI", 2, 2, 650, 9, 1, 0, 1, 0, 6, 55, 0, 0, 0, 0, 0, 0, None),
        ("BUF", "ARI", 4, 2, 600, 0, 1, 0, 1, 0, 6, 50, 0, 0, 0, 0, 0, 0, None),
        # BUF drive 2 starts at its own 50; this lost fumble is still a scrimmage play.
        ("BUF", "ARI", 3, 2, 580, 9, 0, 1, 0, 1, 9, 50, 1, 1, 0, 0, 0, 0, None),
        # Special teams is assigned to posteam, rather than a separate return-team field.
        ("BUF", "ARI", 1, 3, 540, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0, 1, 1.2, None),
        ("BUF", "ARI", 2, 3, 510, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0, 1, -0.2, None),
        # ARI supplies a distinct offense and an interception.
        ("ARI", "BUF", 1, 4, 900, 0, 1, 0, 1, 0, 4, 70, 0, 0, 1, 0, 0, 0, None),
    ]
    columns = [
        "posteam",
        "defteam",
        "qtr",
        "drive",
        "game_seconds_remaining",
        "posteam_score_differential",
        "pass",
        "rush",
        "qb_dropback",
        "rush_attempt",
        "yards_gained",
        "yardline_100",
        "fumble_lost",
        "fumble",
        "interception",
        "sack",
        "special_teams_play",
        "epa",
        "play_id",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out["game_id"] = "2024_01_ARI_BUF"
    out["season"] = 2024
    out["week"] = 1
    out["season_type"] = "REG"
    out["home_team"] = "BUF"
    return out


def test_team_game_style_uses_declared_formulas_and_denominators():
    """Changing a denominator to pass attempts, all plays, or all drive rows is a bug."""
    out = team_game_style(style_pbp()).set_index("team")
    buf = out.loc["BUF"]

    assert buf["neutral_pass_rate"] == pytest.approx(2 / 3)
    assert buf["turnover_rate"] == pytest.approx(1 / 6)
    assert buf["explosive_play_rate"] == pytest.approx(2 / 6)
    assert buf["starting_field_position"] == pytest.approx(32.5)
    assert buf["special_teams_epa"] == pytest.approx(0.5)
    assert out.loc["ARI", "turnover_rate"] == 1.0


def test_pace_excludes_quarter_break_and_timeout_gaps():
    out = team_game_style(style_pbp()).set_index("team")

    assert out.loc["BUF", "pace_seconds"] == pytest.approx(27.0)


def style_games():
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "season": 2024,
                    "week": week,
                    "team": "A",
                    "game_id": f"A-{week}",
                    "neutral_pass_rate": 0.6,
                    "pace_seconds": 28.0,
                    "turnover_rate": 0.1,
                    "explosive_play_rate": 0.2,
                    "starting_field_position": 30.0,
                    "special_teams_epa": 0.1,
                    "n_turnovers": 10,
                    "n_scrimmage_plays": 100,
                },
                {
                    "season": 2024,
                    "week": week,
                    "team": "B",
                    "game_id": f"B-{week}",
                    "neutral_pass_rate": 0.4,
                    "pace_seconds": 32.0,
                    "turnover_rate": 0.0,
                    "explosive_play_rate": 0.1,
                    "starting_field_position": 25.0,
                    "special_teams_epa": -0.1,
                    "n_turnovers": 0,
                    "n_scrimmage_plays": 100,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_turnover_rate_is_shrunk_over_trailing_history():
    out = style_features_for_targets(style_games(), [(2024, 5)]).set_index("team")
    a = out.loc["A"]

    assert a["raw_turnover_rate"] == pytest.approx(0.1)
    history_weight = sum(0.5 ** (age / 8) for age in (4, 3, 2, 1))
    assert a["turnover_rate"] == pytest.approx(
        (10 * history_weight + 200 * 0.05) / (100 * history_weight + 200)
    )
    assert 0 < a["turnover_rate"] < a["raw_turnover_rate"]
    assert TURNOVER_PRIOR_PLAYS == 200


def test_style_history_is_exponentially_weighted_and_strictly_as_of():
    games = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "team": "A",
                "neutral_pass_rate": 0.0,
                "pace_seconds": 20.0,
                "turnover_rate": 0.0,
                "explosive_play_rate": 0.0,
                "starting_field_position": 20.0,
                "special_teams_epa": 0.0,
                "n_turnovers": 0,
                "n_scrimmage_plays": 100,
            },
            {
                "season": 2024,
                "week": 4,
                "team": "A",
                "neutral_pass_rate": 1.0,
                "pace_seconds": 40.0,
                "turnover_rate": 0.0,
                "explosive_play_rate": 1.0,
                "starting_field_position": 40.0,
                "special_teams_epa": 1.0,
                "n_turnovers": 0,
                "n_scrimmage_plays": 100,
            },
        ]
    )
    base = style_features_for_targets(games, [(2024, 5)]).set_index("team")
    future = games.copy()
    future.loc[len(future)] = {**future.iloc[1].to_dict(), "week": 5, "neutral_pass_rate": 99.0}
    poisoned = style_features_for_targets(future, [(2024, 5)]).set_index("team")
    old_weight = 0.5 ** (2 / 8)
    recent_weight = 0.5 ** (1 / 8)

    assert base.loc["A", "neutral_pass_rate"] == pytest.approx(
        recent_weight / (old_weight + recent_weight)
    )
    pd.testing.assert_series_equal(base.loc["A"], poisoned.loc["A"])


def test_no_eligible_history_is_imputed_with_documented_schema():
    target_only = style_games().query("week == 1").copy()
    out = style_features_for_targets(target_only, [(2024, 1)]).set_index("team")

    assert out.index.name == "team"
    assert set(STYLE_FEATURE_COLS).issubset(out.columns)
    assert out["style_imputed"].eq(1).all()
    assert out[list(STYLE_FEATURE_COLS[:-1])].notna().all().all()


def _style_game(season, week, team, neutral_pass_rate):
    return {
        "season": season,
        "week": week,
        "team": team,
        "neutral_pass_rate": neutral_pass_rate,
        "pace_seconds": 30.0,
        "turnover_rate": 0.0,
        "explosive_play_rate": 0.0,
        "starting_field_position": 30.0,
        "special_teams_epa": 0.0,
        "n_turnovers": 0,
        "n_scrimmage_plays": 100,
    }


def test_style_history_uses_game_ordinal_recency_across_byes_and_seasons():
    bye_games = pd.DataFrame([_style_game(2024, 1, "A", 0.0), _style_game(2024, 3, "A", 1.0)])
    bye = style_features_for_targets(bye_games, [(2024, 5)]).set_index("team")
    newest = 0.5 ** (1 / 8)
    previous = 0.5 ** (2 / 8)
    assert bye.loc["A", "neutral_pass_rate"] == pytest.approx(newest / (newest + previous))

    cross_season_games = pd.DataFrame(
        [_style_game(2022, 18, "A", 0.0), _style_game(2023, 18, "A", 1.0)]
    )
    cross_season = style_features_for_targets(cross_season_games, [(2024, 1)]).set_index("team")
    assert cross_season.loc["A", "neutral_pass_rate"] == pytest.approx(newest / (newest + previous))


def test_style_history_excludes_the_ninth_most_recent_game():
    games = pd.DataFrame(
        [_style_game(2024, week, "A", 0.0 if week == 1 else 1.0) for week in range(1, 10)]
    )
    out = style_features_for_targets(games, [(2024, 10)]).set_index("team")

    assert out.loc["A", "neutral_pass_rate"] == 1.0


def test_shuffled_plays_use_play_id_order_for_pace_and_drive_starts():
    ordered = style_pbp().copy()
    ordered["play_id"] = range(1, len(ordered) + 1)
    shuffled = ordered.sample(frac=1, random_state=7).reset_index(drop=True)
    expected = team_game_style(ordered).set_index("team").loc["BUF"]
    actual = team_game_style(shuffled).set_index("team").loc["BUF"]

    assert actual["pace_seconds"] == expected["pace_seconds"]
    assert actual["starting_field_position"] == expected["starting_field_position"]


def test_overlapping_interception_and_lost_fumble_is_one_turnover_play():
    pbp = style_pbp().copy()
    pbp.loc[pbp.index[0], ["interception", "fumble_lost"]] = 1
    buf = team_game_style(pbp).set_index("team").loc["BUF"]

    assert buf["n_turnovers"] == 2
    assert buf["turnover_rate"] == pytest.approx(2 / 6)
