### Task 4: Opponent-adjusted ratings

**Files:**
- Modify: `src/nfl_game/ratings/epa.py` (append `fit_ratings`)
- Test: `tests/test_fit_ratings.py`

**Interfaces:**
- Consumes: `team_game_epa` output.
- Produces: `fit_ratings(team_games: pd.DataFrame, target: str = "epa_play", alpha: float = 1.0, weights: np.ndarray | None = None) -> pd.DataFrame` with columns `team, off_rating, def_rating` plus attribute-free league mean returned as `.attrs["league_mean"]`.

**Context for the implementer:** This is the most important function in the project. Raw EPA mostly measures schedule quality — a team that played four bad defenses looks great. Regressing play-level EPA on offense-team and defense-team dummies separates the two effects.

Sign convention, and be careful here: the raw defense coefficient is *positive when that defense allows more EPA*, which means a good defense has a negative coefficient. **Negate it** so that in the returned frame, higher is better for both `off_rating` and `def_rating`. Every downstream consumer assumes that.

Ridge (L2) shrinkage is deliberate: teams with few observations get pulled toward the league mean, which is exactly the behavior needed early in a season.

- [ ] **Step 1: Write the failing test**

`tests/test_fit_ratings.py`:

```python
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
                "game_id": f"x{i}", "season": 2024, "week": 20 + i, "team": "C",
                "opponent": "A", "is_home": 0, "epa_play": -0.10 + 0.20,
                "epa_pass": 0.10, "epa_rush": 0.10, "success_rate": 0.45,
                "n_pass": 30, "n_rush": 25,
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
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_fit_ratings.py -v
```

Expected: `ImportError: cannot import name 'fit_ratings'`.

- [ ] **Step 3: Append `fit_ratings` to `src/nfl_game/ratings/epa.py`**

Add these imports at the top of the file:

```python
import numpy as np
from sklearn.linear_model import Ridge
```

Then append:

```python
def fit_ratings(
    team_games: pd.DataFrame,
    target: str = "epa_play",
    alpha: float = 1.0,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Opponent-adjusted offensive and defensive ratings via ridge regression.

    Regresses each team-game's `target` on offense-team and defense-team indicators.
    This separates a team's own quality from the quality of who it happened to play —
    without it, ratings mostly measure schedule luck.

    Ridge shrinkage pulls thin-sample teams toward the league mean, which is what gives
    early-season ratings a sane prior.

    Returns one row per team with `off_rating` and `def_rating`, both oriented so that
    **higher is better**. The raw defense coefficient means "EPA allowed", so it is
    negated here.

    The league mean (the fitted intercept) is on `.attrs["league_mean"]`.
    """
    df = team_games[team_games[target].notna()].copy()
    if df.empty:
        raise ValueError(f"no rows with non-null {target!r}")

    teams = sorted(set(df["team"]) | set(df["opponent"]))
    off = pd.get_dummies(pd.Categorical(df["team"], categories=teams), prefix="off")
    dfn = pd.get_dummies(pd.Categorical(df["opponent"], categories=teams), prefix="def")
    X = pd.concat([off, dfn], axis=1).astype(float).to_numpy()
    y = df[target].to_numpy(dtype=float)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=weights)

    n = len(teams)
    out = pd.DataFrame(
        {
            "team": teams,
            "off_rating": model.coef_[:n],
            "def_rating": -model.coef_[n:],
        }
    )
    out.attrs["league_mean"] = float(model.intercept_)
    return out
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_fit_ratings.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Sanity-check on a real season**

```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_pbp; from nfl_game.ratings.epa import team_game_epa, fit_ratings; r=fit_ratings(team_game_epa(load_pbp([2024], save=False))); print(r.sort_values('off_rating',ascending=False).head(5)); print(r.sort_values('def_rating',ascending=False).head(5))"
```

Expected: 2024's best offenses (BAL, DET, BUF) top the offense list and 2024's best defenses top the defense list. **If the defense list shows the worst defenses first, the negation is backwards — fix it before continuing**, because every later task inherits this sign.

- [ ] **Step 6: Commit**

```bash
git add src/nfl_game/ratings/epa.py tests/test_fit_ratings.py
git commit -m "feat: opponent-adjusted team ratings via ridge on team dummies"
```

---

