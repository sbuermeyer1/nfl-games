import pandas as pd

from nfl_game.ratings.ngs import NGS_METRICS, team_week_ngs


def _passing():
    return pd.DataFrame(
        {
            "season": [2024] * 4,
            "week": [0, 1, 1, 1],  # week 0 is a season aggregate and must be dropped
            "season_type": ["REG"] * 4,
            "team_abbr": ["BUF", "BUF", "BUF", "KC"],
            "attempts": [500, 30, 10, 40],
            "completion_percentage_above_expectation": [9.9, 5.0, 1.0, 2.0],
            "avg_time_to_throw": [9.9, 2.8, 2.4, 2.6],
            "avg_air_yards_to_sticks": [9.9, 1.0, -1.0, 0.5],
            "aggressiveness": [99.0, 20.0, 12.0, 15.0],
        }
    )


def _rushing():
    return pd.DataFrame(
        {
            "season": [2024], "week": [1], "season_type": ["REG"], "team_abbr": ["BUF"],
            "rush_attempts": [25],
            "rush_yards_over_expected_per_att": [0.8],
            "percent_attempts_gte_eight_defenders": [22.0],
        }
    )


def _receiving():
    return pd.DataFrame(
        {
            "season": [2024, 2024], "week": [1, 1], "season_type": ["REG"] * 2,
            "team_abbr": ["BUF", "KC"], "targets": [20, 30],
            "avg_separation": [3.0, 2.5],
            "avg_yac_above_expectation": [0.5, -0.2],
        }
    )


def test_drops_week_zero_aggregates():
    out = team_week_ngs(_passing(), _rushing(), _receiving())
    assert (out["week"] != 0).all()


def test_attempt_weighted_aggregation():
    out = team_week_ngs(_passing(), _rushing(), _receiving())
    buf = out[out["team"] == "BUF"].iloc[0]
    # (5.0*30 + 1.0*10) / 40 = 4.0
    assert buf["cpoe"] == 4.0


def test_one_row_per_team_week():
    out = team_week_ngs(_passing(), _rushing(), _receiving())
    assert len(out) == 2
    assert set(out["team"]) == {"BUF", "KC"}


def test_missing_rushing_is_imputed_and_flagged():
    out = team_week_ngs(_passing(), _rushing(), _receiving()).set_index("team")
    # KC has no rushing row at all
    assert out.loc["KC", "ryoe_per_att_imputed"] == 1
    assert out.loc["BUF", "ryoe_per_att_imputed"] == 0
    assert out.loc["KC", "ryoe_per_att"] == out.loc["BUF", "ryoe_per_att"]  # league mean of 1


def test_all_metrics_and_flags_present():
    out = team_week_ngs(_passing(), _rushing(), _receiving())
    for m in NGS_METRICS:
        assert m in out.columns
        assert f"{m}_imputed" in out.columns
    assert out[NGS_METRICS].notna().all().all()


def test_postseason_passing_rows_are_excluded():
    """Falsifiable both ways: real cpoe when the row is REG, imputed when it is POST."""
    p_reg = _passing()
    p_post = _passing()
    p_post["season_type"] = "POST"

    reg = team_week_ngs(p_reg, _rushing(), _receiving()).set_index("team")
    post = team_week_ngs(p_post, _rushing(), _receiving()).set_index("team")

    assert reg.loc["BUF", "cpoe_imputed"] == 0
    assert reg.loc["BUF", "cpoe"] == 4.0
    assert post.loc["BUF", "cpoe_imputed"] == 1
