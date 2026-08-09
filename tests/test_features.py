import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import (
    FEATURE_COLS,
    TARGET_COLS,
    MissingRatingJoinError,
    build_game_features,
)


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
    for team, off, dfn in (
        ("BUF", 0.2, 0.1),
        ("KC", 0.1, 0.15),
        ("NYJ", -0.1, 0.0),
        # MIA's defence is -0.04 rather than -0.05 so that NYJ's home pass edge does not
        # land on exactly 0.0, where an assertion would be satisfied by too many mutations.
        ("MIA", 0.0, -0.04),
    ):
        rows.append(
            {
                "season": 2024,
                "week": 2,
                "team": team,
                "off_rating": off,
                "def_rating": dfn,
                "off_rating_pass": off + 0.05,
                "def_rating_pass": dfn,
                "off_rating_rush": off - 0.05,
                "def_rating_rush": dfn - 0.02,
            }
        )
    return pd.DataFrame(rows)


def _ngs():
    rows = []
    for week in (1, 2):
        # ryoe_per_att and separation vary per team as well as cpoe. They used to be
        # constant across teams, which made every ryoe_diff/separation_diff exactly 0.0
        # and left a sign flip on either one undetectable by construction.
        for team, cpoe, ryoe, sep in (
            ("BUF", 4.0, 0.30, 3.10),
            ("KC", 2.0, 0.10, 2.80),
            ("NYJ", -1.0, -0.20, 2.50),
            ("MIA", 0.5, 0.05, 2.90),
        ):
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "team": team,
                    "cpoe": cpoe,
                    "time_to_throw": 2.7,
                    "air_yards_to_sticks": 0.0,
                    "aggressiveness": 15.0,
                    "ryoe_per_att": ryoe,
                    "pct_eight_defenders": 20.0,
                    "separation": sep,
                    "yac_oe": 0.0,
                    "cpoe_imputed": 0,
                    "time_to_throw_imputed": 0,
                    "air_yards_to_sticks_imputed": 0,
                    "aggressiveness_imputed": 0,
                    "ryoe_per_att_imputed": 0,
                    "pct_eight_defenders_imputed": 0,
                    "separation_imputed": 0,
                    "yac_oe_imputed": 0,
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


def test_every_rating_feature_has_its_exact_value_pinned():
    """All four edges plus net_rating_diff, on both games.

    test_rating_edges_use_opposing_defense pins one cell of one game, which leaves the
    other three edges and net_rating_diff free: a sign flip, an own-defence substitution,
    or net_rating_diff silently dropping its defensive term all survived it.
    Both games are asserted because the two carry opposite-signed values, so a uniform
    sign flip cannot satisfy both.
    """
    out = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")

    # home BUF (off .2/.25/.15, def .1/.08) vs away KC (off .1/.15/.05, def .15/.13)
    buf = out.loc["2024_02_KC_BUF"]
    assert buf["off_pass_edge_home"] == pytest.approx(0.10)  # .25 - .15
    assert buf["off_rush_edge_home"] == pytest.approx(0.02)  # .15 - .13
    assert buf["off_pass_edge_away"] == pytest.approx(0.05)  # .15 - .10
    assert buf["off_rush_edge_away"] == pytest.approx(-0.03)  # .05 - .08
    assert buf["net_rating_diff"] == pytest.approx(0.05)  # (.2+.1) - (.1+.15)

    # home NYJ (off -.1/-.05/-.15, def 0/-.02) vs away MIA (off 0/.05/-.05, def -.04/-.06)
    nyj = out.loc["2024_02_MIA_NYJ"]
    assert nyj["off_pass_edge_home"] == pytest.approx(-0.01)  # -.05 - -.04
    assert nyj["off_rush_edge_home"] == pytest.approx(-0.09)  # -.15 - -.06
    assert nyj["off_pass_edge_away"] == pytest.approx(0.05)  # .05 - 0
    assert nyj["off_rush_edge_away"] == pytest.approx(-0.03)  # -.05 - -.02
    assert nyj["net_rating_diff"] == pytest.approx(-0.06)  # (-.1+0) - (0+-.04)


def test_ngs_diffs_are_home_minus_away():
    """The three NGS diffs, on both games.

    All three were unpinned: flipping cpoe_diff to away - home survived the whole file.
    Week-2 features come from week-1 NGS only (the leak guard), and the fixture repeats
    each team's values across both weeks, so the trailing mean equals the team's value.
    """
    out = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")

    buf = out.loc["2024_02_KC_BUF"]  # home BUF, away KC
    assert buf["cpoe_diff"] == pytest.approx(2.0)  # 4.0 - 2.0
    assert buf["ryoe_diff"] == pytest.approx(0.20)  # .30 - .10
    assert buf["separation_diff"] == pytest.approx(0.30)  # 3.10 - 2.80

    nyj = out.loc["2024_02_MIA_NYJ"]  # home NYJ, away MIA
    assert nyj["cpoe_diff"] == pytest.approx(-1.5)  # -1.0 - 0.5
    assert nyj["ryoe_diff"] == pytest.approx(-0.25)  # -.20 - .05
    assert nyj["separation_diff"] == pytest.approx(-0.40)  # 2.50 - 2.90


def test_ngs_imputed_flag_propagates_from_prior_weeks():
    """ngs_imputed_any is 0 when no prior week was imputed and 1 when one was.

    Asserting only the 0 case would let a hardcoded 0 through.
    """
    clean = build_game_features(_schedules(), _ratings(), _ngs()).set_index("game_id")
    assert clean.loc["2024_02_KC_BUF", "ngs_imputed_any"] == 0

    ngs = _ngs()
    # Flag BUF's week-1 CPOE as imputed; BUF is the home team of the KC_BUF game.
    ngs.loc[(ngs["team"] == "BUF") & (ngs["week"] == 1), "cpoe_imputed"] = 1
    flagged = build_game_features(_schedules(), _ratings(), ngs).set_index("game_id")
    assert flagged.loc["2024_02_KC_BUF", "ngs_imputed_any"] == 1
    # The other game's teams were untouched, so it stays clean.
    assert flagged.loc["2024_02_MIA_NYJ", "ngs_imputed_any"] == 0


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
    assert (
        baseline.loc["2024_02_KC_BUF", "cpoe_diff"] == poisoned.loc["2024_02_KC_BUF", "cpoe_diff"]
    )


def test_failed_rating_join_raises_instead_of_zero_filling():
    """A team whose code does not match the ratings frame must fail loudly.

    This is the detector for the bug class that cost this project ten seasons of Rams
    NGS features: every join onto a game is a left merge, so a code mismatch produced a
    silently zero-filled feature rather than an error. The rating columns carry no
    imputation flag, so without this there is no signal at all that the join missed.
    """
    ratings = _ratings()
    ratings = ratings[ratings["team"] != "BUF"]  # BUF plays in week 2 and is now unrated

    with pytest.raises(MissingRatingJoinError, match="2024_02_KC_BUF"):
        build_game_features(_schedules(), ratings, _ngs())


def test_week_with_no_ratings_at_all_is_zero_filled_without_error():
    """The guard must not fire where a missing rating is legitimate.

    Week 1 of a season has no strictly-prior games to build ratings from, so every team
    is unrated. That is by design, not a failed join, and must stay a quiet zero-fill.
    """
    ratings = _ratings().assign(week=1)  # ratings exist, but only for a week nobody plays

    out = build_game_features(_schedules(), ratings, _ngs()).set_index("game_id")

    assert out.loc["2024_02_KC_BUF", "net_rating_diff"] == 0.0
    assert out.loc["2024_02_KC_BUF", "off_pass_edge_home"] == 0.0


def test_future_games_kept_with_null_targets():
    sched = _schedules()
    sched.loc[0, ["home_score", "away_score", "result", "total"]] = None
    out = build_game_features(sched, _ratings(), _ngs()).set_index("game_id")
    assert pd.isna(out.loc["2024_02_KC_BUF", "margin"])
    assert out.loc["2024_02_KC_BUF", FEATURE_COLS].notna().all()


def test_future_2026_games_with_lines_keep_finite_features_and_null_targets():
    """Scheduled games retain model features while their game results are unknown."""
    sched = _schedules().iloc[[0]].copy()
    sched["game_id"] = "2026_01_KC_BUF"
    sched["season"] = 2026
    sched["week"] = 1
    sched[["home_score", "away_score", "result", "total"]] = None

    ratings = _ratings().copy()
    ratings["season"] = 2026
    ratings["week"] = 1
    ngs = _ngs().copy()
    ngs["season"] = 2026
    ngs["week"] = ngs["week"].replace({1: 0, 2: 1})

    out = build_game_features(sched, ratings, ngs).set_index("game_id")

    assert pd.isna(out.loc["2026_01_KC_BUF", "margin"])
    assert pd.isna(out.loc["2026_01_KC_BUF", "total_points"])
    assert np.isfinite(out.loc["2026_01_KC_BUF", FEATURE_COLS]).all()
