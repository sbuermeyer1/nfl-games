from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_game.ratings.ftn import (
    FTN_FEATURE_COLS,
    ftn_features_for_targets,
    ftn_game_features,
    team_game_ftn,
)


def _charted(rows: list[dict[str, object]]) -> pd.DataFrame:
    """FTN rows in the live schema: no team column, and is_trick_play as object dtype."""
    frame = pd.DataFrame(rows)
    frame["is_trick_play"] = pd.Series([False] * len(frame), dtype=object)
    frame["date_pulled"] = pd.Timestamp("2024-09-06", tz="UTC")
    return frame


def _play(
    game_id: str,
    play_id: int,
    season: int = 2022,
    week: int = 1,
    *,
    motion: bool = False,
    play_action: bool = False,
    rpo: bool = False,
    screen: bool = False,
    out_of_pocket: bool = False,
    int_worthy: bool = False,
    catchable: bool = True,
    drop: bool = False,
    qb_fault_sack: bool = False,
    blitzers: int = 1,
    rushers: int = 4,
) -> dict[str, object]:
    return {
        "nflverse_game_id": game_id,
        "nflverse_play_id": play_id,
        "season": season,
        "week": week,
        "is_motion": motion,
        "is_play_action": play_action,
        "is_rpo": rpo,
        "is_screen_pass": screen,
        "is_qb_out_of_pocket": out_of_pocket,
        "is_interception_worthy": int_worthy,
        "is_catchable_ball": catchable,
        "is_drop": drop,
        "is_qb_fault_sack": qb_fault_sack,
        "n_blitzers": blitzers,
        "n_pass_rushers": rushers,
    }


def _pbp(rows: list[tuple[str, int, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "play_id": play_id,
                "posteam": posteam,
                "season": season,
                "week": week,
                "season_type": "REG",
            }
            for game_id, play_id, posteam, season, week in rows
        ]
    )


def test_offense_comes_from_the_play_by_play_join_because_ftn_has_no_team():
    ftn = _charted(
        [
            _play("2022_01_BUF_LA", 1, motion=True),
            _play("2022_01_BUF_LA", 2, motion=True),
            _play("2022_01_BUF_LA", 3, motion=False),
            _play("2022_01_BUF_LA", 4, motion=False),
        ]
    )
    pbp = _pbp(
        [
            ("2022_01_BUF_LA", 1, "BUF", 2022, 1),
            ("2022_01_BUF_LA", 2, "BUF", 2022, 1),
            ("2022_01_BUF_LA", 3, "LA", 2022, 1),
            ("2022_01_BUF_LA", 4, "LA", 2022, 1),
        ]
    )

    out = team_game_ftn(ftn, pbp)

    assert list(out["team"]) == ["BUF", "LA"]
    # Both teams charted two plays in the same game; only the join separates them.
    assert list(out["n_charted_plays"]) == [2, 2]
    assert out.loc[out["team"].eq("BUF"), "ftn_motion_rate"].iloc[0] == 1.0
    assert out.loc[out["team"].eq("LA"), "ftn_motion_rate"].iloc[0] == 0.0


def test_rates_and_means_are_computed_from_the_charted_plays():
    ftn = _charted(
        [
            _play("2022_01_BUF_LA", 1, play_action=True, drop=True, blitzers=0, rushers=4),
            _play("2022_01_BUF_LA", 2, play_action=True, rpo=True, blitzers=2, rushers=6),
            _play("2022_01_BUF_LA", 3, screen=True, catchable=False, blitzers=4, rushers=6),
            _play("2022_01_BUF_LA", 4, out_of_pocket=True, int_worthy=True, blitzers=2, rushers=4),
        ]
    )
    pbp = _pbp([("2022_01_BUF_LA", play, "BUF", 2022, 1) for play in (1, 2, 3, 4)])

    row = team_game_ftn(ftn, pbp).iloc[0]

    assert row["ftn_play_action_rate"] == 0.5
    assert row["ftn_rpo_rate"] == 0.25
    assert row["ftn_screen_rate"] == 0.25
    assert row["ftn_out_of_pocket_rate"] == 0.25
    assert row["ftn_int_worthy_rate"] == 0.25
    assert row["ftn_catchable_rate"] == 0.75
    assert row["ftn_drop_rate"] == 0.25
    assert row["ftn_blitzers_mean"] == 2.0
    assert row["ftn_pass_rushers_mean"] == 5.0


def test_a_charted_play_with_no_matching_pbp_row_is_dropped_not_guessed():
    ftn = _charted([_play("2022_01_BUF_LA", 1), _play("2022_01_XXX_YYY", 99)])
    pbp = _pbp([("2022_01_BUF_LA", 1, "BUF", 2022, 1)])

    out = team_game_ftn(ftn, pbp)

    assert list(out["team"]) == ["BUF"]
    assert list(out["n_charted_plays"]) == [1]


@pytest.mark.parametrize("missing", ["nflverse_play_id", "is_motion", "n_blitzers"])
def test_an_absent_charted_column_raises_rather_than_yielding_a_constant(missing):
    ftn = _charted([_play("2022_01_BUF_LA", 1)]).drop(columns=[missing])
    pbp = _pbp([("2022_01_BUF_LA", 1, "BUF", 2022, 1)])

    with pytest.raises(ValueError, match=missing):
        team_game_ftn(ftn, pbp)


def test_an_absent_pbp_column_raises():
    ftn = _charted([_play("2022_01_BUF_LA", 1)])
    pbp = _pbp([("2022_01_BUF_LA", 1, "BUF", 2022, 1)]).drop(columns=["posteam"])

    with pytest.raises(ValueError, match="posteam"):
        team_game_ftn(ftn, pbp)


def _team_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": f"2022_{week:02d}_BUF_LA",
                "season": 2022,
                "week": week,
                "team": team,
                "n_charted_plays": 40,
                "ftn_motion_rate": rate,
                "ftn_play_action_rate": 0.2,
                "ftn_rpo_rate": 0.1,
                "ftn_screen_rate": 0.1,
                "ftn_out_of_pocket_rate": 0.1,
                "ftn_int_worthy_rate": 0.02,
                "ftn_catchable_rate": 0.8,
                "ftn_drop_rate": 0.05,
                "ftn_blitzers_mean": 1.5,
                "ftn_pass_rushers_mean": 4.2,
                "ftn_qb_fault_sack_rate": 0.03,
            }
            for week, team, rate in (
                (1, "BUF", 0.4),
                (1, "LA", 0.2),
                (2, "BUF", 0.6),
                (2, "LA", 0.2),
            )
        ]
    )


def test_a_later_game_cannot_change_an_earlier_target_feature():
    games = _team_games()
    targets = [(2022, 2)]

    before = ftn_features_for_targets(games.loc[games["week"].eq(1)], targets)
    after = ftn_features_for_targets(games, targets)

    buffalo_before = before.loc[before["team"].eq("BUF")].reset_index(drop=True)
    buffalo_after = after.loc[after["team"].eq("BUF")].reset_index(drop=True)
    # Week 2's own result is in `after` but must not touch the week-2 target's features.
    pd.testing.assert_frame_equal(buffalo_before, buffalo_after)
    assert buffalo_after["ftn_motion_rate"].iloc[0] == pytest.approx(0.4)


def test_a_team_with_no_prior_history_is_flagged_imputed_and_one_with_history_is_not():
    games = _team_games()

    week_one = ftn_features_for_targets(games, [(2022, 1)])
    week_two = ftn_features_for_targets(games, [(2022, 2)])

    assert set(week_one["ftn_imputed"]) == {1.0}
    assert set(week_two["ftn_imputed"]) == {0.0}
    assert list(week_two.columns) == ["season", "week", "team", *FTN_FEATURE_COLS]


def test_game_features_are_diffs_and_sums_with_a_missing_side_flagged():
    features = pd.DataFrame(
        [
            {
                "season": 2022,
                "week": 2,
                "team": team,
                **{name: value for name in FTN_FEATURE_COLS if name != "ftn_imputed"},
                "ftn_imputed": 0.0,
            }
            for team, value in (("BUF", 0.5), ("LA", 0.2))
        ]
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2022,
                "week": 2,
                "home_team": "BUF",
                "away_team": "LA",
            },
            {
                "game_id": "g2",
                "season": 2022,
                "week": 2,
                "home_team": "BUF",
                "away_team": "ZZZ",
            },
        ]
    )

    out = ftn_game_features(games, features)

    assert out.loc[out["game_id"].eq("g1"), "ftn_motion_rate_diff"].iloc[0] == pytest.approx(0.3)
    assert out.loc[out["game_id"].eq("g1"), "ftn_motion_rate_sum"].iloc[0] == pytest.approx(0.7)
    assert out.loc[out["game_id"].eq("g1"), "ftn_imputed_any"].iloc[0] == 0.0
    # An unjoined side is an imputation, not a zero.
    assert out.loc[out["game_id"].eq("g2"), "ftn_imputed_any"].iloc[0] == 1.0
    assert np.isnan(out.loc[out["game_id"].eq("g2"), "ftn_motion_rate_diff"].iloc[0])
