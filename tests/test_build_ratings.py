import pandas as pd
import pytest

from nfl_game.ratings.build import (
    build_ratings,
    decay_weights,
    ratings_by_week,
    ratings_for_targets,
)


def _games():
    rows = []
    gid = 0
    for season in (2023, 2024):
        for week in range(1, 5):
            for team, opp in (("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")):
                gid += 1
                rows.append(
                    {
                        "game_id": f"g{gid}",
                        "season": season,
                        "week": week,
                        "team": team,
                        "opponent": opp,
                        "is_home": 1,
                        "epa_play": 0.1 if team in ("A", "C") else -0.1,
                        "epa_pass": 0.1 if team in ("A", "C") else -0.1,
                        "epa_rush": 0.05 if team in ("A", "C") else -0.05,
                        "success_rate": 0.45,
                        "n_pass": 30,
                        "n_rush": 25,
                    }
                )
    return pd.DataFrame(rows)


def test_excludes_current_and_future_weeks():
    """The central correctness property: no data at or after the as-of point."""
    df = _games()
    w = decay_weights(df, asof_season=2024, asof_week=3)
    future = (df["season"] > 2024) | ((df["season"] == 2024) & (df["week"] >= 3))
    assert (w[future.to_numpy()] == 0).all()
    assert (w[~future.to_numpy()] > 0).all()


def test_recent_games_weigh_more():
    df = _games()
    w = decay_weights(df, asof_season=2024, asof_week=5, halflife_games=2.0)
    latest = (df["season"] == 2024) & (df["week"] == 4)
    oldest = (df["season"] == 2023) & (df["week"] == 1)
    assert w[latest.to_numpy()].mean() > w[oldest.to_numpy()].mean()


def test_prior_season_downweighted_by_penalty():
    df = _games()
    w = decay_weights(df, asof_season=2024, asof_week=1, halflife_games=1e9, season_penalty=0.5)
    prior = (df["season"] == 2023).to_numpy()
    # halflife is effectively infinite, so any gap must come from the season penalty
    assert w[prior].max() == pytest.approx(0.5, rel=1e-6)


def test_build_ratings_returns_all_rating_columns():
    out = build_ratings(_games(), asof_season=2024, asof_week=5)
    expected = {
        "team",
        "off_rating",
        "def_rating",
        "off_rating_pass",
        "def_rating_pass",
        "off_rating_rush",
        "def_rating_rush",
    }
    assert set(out.columns) == expected
    assert sorted(out["team"]) == ["A", "B", "C", "D"]


def test_build_ratings_orders_teams_correctly():
    out = build_ratings(_games(), asof_season=2024, asof_week=5).set_index("team")
    assert out.loc["A", "off_rating"] > out.loc["B", "off_rating"]


def test_build_ratings_drops_future_rows_rather_than_zero_weighting_them():
    """The leak guard is doubled -- decay_weights zeroes future games AND build_ratings
    drops them -- and only the weighting half was pinned: keeping every row with its
    zero weight survived the whole suite, because the ridge fit ignores zero-weight rows
    anyway. It is not equivalent, though. fit_ratings derives its team list from the
    rows it is handed, so a zero-weighted future row still puts its teams in the output
    with a rating fitted from no data at all -- a team that has not played yet quietly
    acquiring a number. Every team in the shared fixture plays every week, which is why
    the difference was invisible; teams E and F exist only at and after the cutoff."""
    df = _games()
    future_only = pd.DataFrame(
        [
            {
                "game_id": "gE1",
                "season": 2024,
                "week": week,
                "team": team,
                "opponent": opp,
                "is_home": 1,
                "epa_play": 0.2,
                "epa_pass": 0.2,
                "epa_rush": 0.1,
                "success_rate": 0.5,
                "n_pass": 30,
                "n_rush": 25,
            }
            for week in (3, 4)
            for team, opp in (("E", "F"), ("F", "E"))
        ]
    )
    out = build_ratings(
        pd.concat([df, future_only], ignore_index=True), asof_season=2024, asof_week=3
    )

    assert sorted(out["team"]) == ["A", "B", "C", "D"]


def test_build_ratings_raises_when_no_prior_data():
    with pytest.raises(ValueError, match="no games before"):
        build_ratings(_games(), asof_season=2023, asof_week=1)


def test_ratings_by_week_covers_every_week():
    out = ratings_by_week(_games(), seasons=[2024])
    assert sorted(out["week"].unique()) == [1, 2, 3, 4]
    assert (out["season"] == 2024).all()
    assert len(out) == 4 * 4  # 4 weeks x 4 teams


def test_ratings_for_targets_builds_future_week_without_same_season_pbp():
    """Explicit future targets use completed history, even when no 2024 games are present."""
    history = _games().query("season < 2024").copy()

    out = ratings_for_targets(history, [(2024, 1), (2024, 2)])

    assert sorted(out[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)) == [
        (2024, 1),
        (2024, 2),
    ]
    assert out.groupby(["season", "week"])["team"].nunique().eq(4).all()


def test_ratings_for_targets_excludes_every_game_in_the_target_week(monkeypatch):
    """A target week is a pre-week cutoff, so Thursday cannot leak into Sunday."""
    seen = []

    def fake_build(team_games, asof_season, asof_week, **kwargs):
        cutoff = team_games.loc[
            (team_games["season"] < asof_season)
            | ((team_games["season"] == asof_season) & (team_games["week"] < asof_week))
        ]
        seen.append(set(zip(cutoff["season"], cutoff["week"], strict=True)))
        return pd.DataFrame({"team": ["AAA"], "off_rating": [0.0], "def_rating": [0.0]})

    monkeypatch.setattr("nfl_game.ratings.build.build_ratings", fake_build)

    ratings_for_targets(_games(), [(2024, 3)])

    assert all(week < 3 for season, week in seen[0] if season == 2024)


def test_ratings_for_targets_returns_documented_empty_schema():
    """No scheduled targets returns a predictable empty ratings boundary."""
    out = ratings_for_targets(_games(), [])

    assert out.empty
    assert list(out.columns) == ["season", "week", "team"]
