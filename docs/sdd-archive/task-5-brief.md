### Task 5: As-of rating table with recency decay

**Files:**
- Create: `src/nfl_game/ratings/build.py`
- Test: `tests/test_build_ratings.py`

**Interfaces:**
- Consumes: `epa.team_game_epa`, `epa.fit_ratings`.
- Produces:
  - `decay_weights(team_games, asof_season, asof_week, halflife_games=10.0, season_penalty=0.6) -> np.ndarray`
  - `build_ratings(team_games, asof_season, asof_week, halflife_games=10.0, season_penalty=0.6, alpha=1.0) -> pd.DataFrame` with columns `team, off_rating, def_rating, off_rating_pass, def_rating_pass, off_rating_rush, def_rating_rush`
  - `ratings_by_week(team_games, seasons, **kwargs) -> pd.DataFrame` — the above stacked, with `season` and `week` columns, for every week in `seasons`.

**Context for the implementer:** This is where the leak-prevention lives. `build_ratings` must use **only** games strictly before `(asof_season, asof_week)`.

There is no separate "preseason regression to the mean" step. Two mechanisms already do that job together: games from prior seasons are downweighted by `season_penalty` per season of age, and ridge shrinks thin-sample teams toward the league mean. In Week 1 a team's rating is therefore built almost entirely from last season's decayed games, shrunk toward average — which is the intended behavior.

- [ ] **Step 1: Write the failing test**

`tests/test_build_ratings.py`:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.ratings.build import build_ratings, decay_weights, ratings_by_week


def _games():
    rows = []
    gid = 0
    for season in (2023, 2024):
        for week in range(1, 5):
            for team, opp in (("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")):
                gid += 1
                rows.append(
                    {
                        "game_id": f"g{gid}", "season": season, "week": week,
                        "team": team, "opponent": opp, "is_home": 1,
                        "epa_play": 0.1 if team in ("A", "C") else -0.1,
                        "epa_pass": 0.1 if team in ("A", "C") else -0.1,
                        "epa_rush": 0.05 if team in ("A", "C") else -0.05,
                        "success_rate": 0.45, "n_pass": 30, "n_rush": 25,
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
        "team", "off_rating", "def_rating",
        "off_rating_pass", "def_rating_pass",
        "off_rating_rush", "def_rating_rush",
    }
    assert set(out.columns) == expected
    assert sorted(out["team"]) == ["A", "B", "C", "D"]


def test_build_ratings_orders_teams_correctly():
    out = build_ratings(_games(), asof_season=2024, asof_week=5).set_index("team")
    assert out.loc["A", "off_rating"] > out.loc["B", "off_rating"]


def test_build_ratings_raises_when_no_prior_data():
    with pytest.raises(ValueError, match="no games before"):
        build_ratings(_games(), asof_season=2023, asof_week=1)


def test_ratings_by_week_covers_every_week():
    out = ratings_by_week(_games(), seasons=[2024])
    assert sorted(out["week"].unique()) == [1, 2, 3, 4]
    assert (out["season"] == 2024).all()
    assert len(out) == 4 * 4  # 4 weeks x 4 teams
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_build_ratings.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.ratings.build'`.

- [ ] **Step 3: Write `src/nfl_game/ratings/build.py`**

```python
"""As-of team ratings: recency-weighted, opponent-adjusted, leak-free.

Every function here takes an (asof_season, asof_week) cutoff and uses only games
strictly before it. That property is what makes the backtest honest, so it is tested
directly in tests/test_build_ratings.py rather than assumed.
"""

import numpy as np
import pandas as pd

from nfl_game.ratings.epa import fit_ratings

RATING_TARGETS = {
    "epa_play": ("off_rating", "def_rating"),
    "epa_pass": ("off_rating_pass", "def_rating_pass"),
    "epa_rush": ("off_rating_rush", "def_rating_rush"),
}


def decay_weights(
    team_games: pd.DataFrame,
    asof_season: int,
    asof_week: int,
    halflife_games: float = 10.0,
    season_penalty: float = 0.6,
) -> np.ndarray:
    """Exponential recency weights; zero for anything at or after the cutoff.

    Weight falls by half every `halflife_games` weeks of age, and is multiplied by
    `season_penalty` for each completed season of distance. The season penalty is what
    makes Week 1 ratings lean on last year without treating it as current.
    """
    season = team_games["season"].to_numpy()
    week = team_games["week"].to_numpy()

    is_past = (season < asof_season) | ((season == asof_season) & (week < asof_week))

    seasons_back = np.maximum(asof_season - season, 0)
    weeks_back = np.where(season == asof_season, asof_week - week, asof_week + (18 - week))
    weeks_back = np.maximum(weeks_back, 0)

    w = 0.5 ** (weeks_back / halflife_games) * (season_penalty**seasons_back)
    return np.where(is_past, w, 0.0)


def build_ratings(
    team_games: pd.DataFrame,
    asof_season: int,
    asof_week: int,
    halflife_games: float = 10.0,
    season_penalty: float = 0.6,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Opponent-adjusted ratings as of a point in time, for overall/pass/rush EPA."""
    w = decay_weights(team_games, asof_season, asof_week, halflife_games, season_penalty)
    if not (w > 0).any():
        raise ValueError(f"no games before season {asof_season} week {asof_week}")

    used = team_games[w > 0].reset_index(drop=True)
    used_w = w[w > 0]

    out = None
    for target, (off_col, def_col) in RATING_TARGETS.items():
        fitted = fit_ratings(used, target=target, alpha=alpha, weights=used_w)
        fitted = fitted.rename(columns={"off_rating": off_col, "def_rating": def_col})
        out = fitted if out is None else out.merge(fitted, on="team", how="outer")

    return out.reset_index(drop=True)


def ratings_by_week(team_games: pd.DataFrame, seasons: list[int], **kwargs) -> pd.DataFrame:
    """Stack build_ratings across every week of every requested season."""
    frames = []
    for season in seasons:
        weeks = sorted(team_games.loc[team_games["season"] == season, "week"].unique())
        for week in weeks:
            r = build_ratings(team_games, asof_season=season, asof_week=int(week), **kwargs)
            r.insert(0, "week", int(week))
            r.insert(0, "season", season)
            frames.append(r)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_build_ratings.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_game/ratings/build.py tests/test_build_ratings.py
git commit -m "feat: as-of recency-weighted rating table with leak guard"
```

---

