import numpy as np
import pandas as pd

from nfl_game.ratings.epa import fit_ratings


def _round_robin():
    """A: great offense, terrible defense. D: terrible offense, great defense.
    Every team plays every other, so schedule strength is balanced by construction."""
    off_skill = {"A": 0.30, "B": 0.10, "C": -0.10, "D": -0.30}
    def_skill = {"A": 0.20, "B": 0.05, "C": -0.05, "D": -0.20}  # positive = allows more
    rows = []
    gid = 0
    for home in off_skill:
        for away in off_skill:
            if home == away:
                continue
            gid += 1
            for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                rows.append(
                    {
                        "game_id": f"g{gid}",
                        "season": 2024,
                        "week": gid,
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "epa_play": off_skill[team] + def_skill[opp],
                        "epa_pass": off_skill[team] + def_skill[opp],
                        "epa_rush": off_skill[team] + def_skill[opp],
                        "success_rate": 0.45,
                        "n_pass": 30,
                        "n_rush": 25,
                    }
                )
    return pd.DataFrame(rows)


def test_recovers_offensive_ordering():
    out = fit_ratings(_round_robin(), alpha=0.01).set_index("team")
    assert out.loc["A", "off_rating"] > out.loc["B", "off_rating"]
    assert out.loc["B", "off_rating"] > out.loc["C", "off_rating"]
    assert out.loc["C", "off_rating"] > out.loc["D", "off_rating"]


def test_higher_def_rating_means_better_defense():
    # D allows the least EPA, so D must have the HIGHEST def_rating.
    out = fit_ratings(_round_robin(), alpha=0.01).set_index("team")
    assert out.loc["D", "def_rating"] > out.loc["A", "def_rating"]


def test_returns_one_row_per_team():
    out = fit_ratings(_round_robin(), alpha=0.01)
    assert sorted(out["team"]) == ["A", "B", "C", "D"]


def test_league_mean_available():
    out = fit_ratings(_round_robin(), alpha=0.01)
    assert "league_mean" in out.attrs
    assert abs(out.attrs["league_mean"]) < 0.5


def test_opponent_adjustment_beats_raw_average():
    """A team fed only elite defenses should not be judged as badly as its raw EPA."""
    df = _round_robin()
    # Give team C a brutal extra slate: three more games, all against A's offense-crushing D.
    extra = []
    for i in range(3):
        extra.append(
            {
                "game_id": f"x{i}",
                "season": 2024,
                "week": 20 + i,
                "team": "C",
                "opponent": "A",
                "is_home": 0,
                "epa_play": -0.10 + 0.20,
                "epa_pass": 0.10,
                "epa_rush": 0.10,
                "success_rate": 0.45,
                "n_pass": 30,
                "n_rush": 25,
            }
        )
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    out = fit_ratings(df, alpha=0.01).set_index("team")
    # C's adjusted offense still sits between B and D despite the schedule distortion
    assert out.loc["B", "off_rating"] > out.loc["C", "off_rating"] > out.loc["D", "off_rating"]


def test_sample_weights_shift_ratings():
    df = _round_robin()
    flat = fit_ratings(df, alpha=0.01).set_index("team")["off_rating"]
    w = np.where(df["team"] == "A", 10.0, 1.0)
    weighted = fit_ratings(df, alpha=0.01, weights=w).set_index("team")["off_rating"]
    assert not np.allclose(flat.values, weighted.values)


def test_weights_aligned_to_prefilter_rows_with_null_target():
    """weights must be filtered by the same mask as X/y, not passed through raw.

    A caller supplies weights aligned to the ORIGINAL team_games frame (pre-filter). If
    that frame has a null-target row, fit_ratings must drop that row's weight along with
    the row itself -- not misalign weights against the filtered design matrix.
    """
    df_null = _round_robin()
    df_null.loc[0, "epa_play"] = np.nan
    w_null = np.where(df_null["team"] == "A", 10.0, 1.0)

    with_null = fit_ratings(df_null, alpha=0.01, weights=w_null).set_index("team")["off_rating"]

    mask = df_null["epa_play"].notna()
    df_dropped = df_null[mask].reset_index(drop=True)
    w_dropped = w_null[mask.to_numpy()]
    dropped = fit_ratings(df_dropped, alpha=0.01, weights=w_dropped).set_index("team")["off_rating"]

    pd.testing.assert_series_equal(with_null.sort_index(), dropped.sort_index())
