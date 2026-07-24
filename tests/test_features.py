import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS, TARGET_COLS, build_game_features


def _schedules():
    return pd.DataFrame(
        {
            "game_id": ["2024_02_KC_BUF", "2024_02_MIA_NYJ"],
            "season": [2024, 2024],
            "week": [2, 2],
            "game_type": ["REG", "REG"],
            "home_team": ["BUF", "NYJ"],
            "away_team": ["KC", "MIA"],
            "home_score": [30, 17],
            "away_score": [20, 24],
            "result": [10, -7],
            "total": [50, 41],
            "home_rest": [7, 10],
            "away_rest": [7, 7],
            "div_game": [0, 1],
            "roof": ["outdoors", "dome"],
            "temp": [45.0, None],
            "wind": [12.0, None],
            "spread_line": [2.5, -1.0],
            "total_line": [48.5, 43.0],
        }
    )


def _ratings():
    rows = []
    for team, off, dfn in (("BUF", 0.2, 0.1), ("KC", 0.1, 0.15), ("NYJ", -0.1, 0.0), ("MIA", 0.0, -0.05)):
        rows.append(
            {
                "season": 2024, "week": 2, "team": team,
                "off_rating": off, "def_rating": dfn,
                "off_rating_pass": off + 0.05, "def_rating_pass": dfn,
                "off_rating_rush": off - 0.05, "def_rating_rush": dfn - 0.02,
            }
        )
    return pd.DataFrame(rows)


def _ngs():
    rows = []
    for week in (1, 2):
        for team, cpoe in (("BUF", 4.0), ("KC", 2.0), ("NYJ", -1.0), ("MIA", 0.5)):
            rows.append(
                {
                    "season": 2024, "week": week, "team": team,
                    "cpoe": cpoe, "time_to_throw": 2.7, "air_yards_to_sticks": 0.0,
                    "aggressiveness": 15.0, "ryoe_per_att": 0.1,
                    "pct_eight_defenders": 20.0, "separation": 2.8, "yac_oe": 0.0,
                    "cpoe_imputed": 0, "time_to_throw_imputed": 0,
                    "air_yards_to_sticks_imputed": 0, "aggressiveness_imputed": 0,
                    "ryoe_per_att_imputed": 0, "pct_eight_defenders_imputed": 0,
                    "separation_imputed": 0, "yac_oe_imputed": 0,
                }
            )
    return pd.DataFrame(rows)


def test_target_cols_fixed():
    assert TARGET_COLS == ["margin", "total_points"]


def test_produces_one_row_per_game():
    out = build_game_features(_schedules(), _ratings(), _ngs())
    assert len(out) == 2
    assert set(out["game_id"]) == {"2024_02_KC_BUF", "2024_02_MIA_NYJ"}


def test_all_feature_columns_present_and_numeric():
    out = build_game_features(_schedules(), _ratings(), _ngs())
    for col in FEATURE_COLS:
        assert col in out.columns, col
    assert out[FEATURE_COLS].notna().all().all()


def test_targets_computed_from_scores():
    out = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")
    assert out.loc["2024_02_KC_BUF", "margin"] == 10
    assert out.loc["2024_02_KC_BUF", "total_points"] == 50
    assert out.loc["2024_02_MIA_NYJ", "margin"] == -7


def test_rating_edges_use_opposing_defense():
    out = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")
    # BUF off_rating_pass 0.25 - KC def_rating_pass 0.15 = 0.10
    assert out.loc["2024_02_KC_BUF", "off_pass_edge_home"] == pytest.approx(0.10)


def test_rest_diff_is_home_minus_away():
    out = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")
    assert out.loc["2024_02_MIA_NYJ", "rest_diff"] == 3


def test_dome_zeroes_weather_and_sets_flag():
    out = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")
    dome = out.loc["2024_02_MIA_NYJ"]
    assert dome["is_dome"] == 1
    assert dome["temp_outdoor"] == 0
    assert dome["wind_outdoor"] == 0
    outdoor = out.loc["2024_02_KC_BUF"]
    assert outdoor["is_dome"] == 0
    assert outdoor["temp_outdoor"] == 45.0
    assert outdoor["wind_outdoor"] == 12.0


def test_ngs_features_exclude_current_week():
    """Leak guard: week-2 features must not see week-2 NGS."""
    ngs = _ngs()
    # Blow up week 2 CPOE. If it leaks into the features, the diff will move.
    ngs.loc[ngs["week"] == 2, "cpoe"] = 99.0
    baseline = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")
    poisoned = build_game_features(_schedules(), ratings=_ratings(), ngs=ngs).set_index("game_id")
    assert baseline.loc["2024_02_KC_BUF", "cpoe_diff"] == poisoned.loc["2024_02_KC_BUF", "cpoe_diff"]


def test_future_games_kept_with_null_targets():
    sched = _schedules()
    sched.loc[0, ["home_score", "away_score", "result", "total"]] = None
    out = build_game_features(sched, _ratings(), _ngs()).set_index("game_id")
    assert pd.isna(out.loc["2024_02_KC_BUF", "margin"])
    assert out.loc["2024_02_KC_BUF", FEATURE_COLS].notna().all()
