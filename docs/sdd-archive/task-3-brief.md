### Task 3: Team-game EPA aggregation

**Files:**
- Create: `src/nfl_game/ratings/epa.py`
- Test: `tests/test_epa.py`

**Interfaces:**
- Consumes: raw pbp frame from `data.nfl.load_pbp`.
- Produces: `team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame` with exactly these columns:
  `game_id, season, week, team, opponent, is_home, epa_play, epa_pass, epa_rush, success_rate, n_pass, n_rush`

**Context for the implementer:** One row per *offense* per game — the team in `posteam` with the EPA it generated against `defteam`. Defensive strength is not computed here; it falls out of the regression in Task 4 as the coefficient on the defense dummy.

Use the `pass` and `rush` **indicator** columns for the split, not `play_type`. In nflverse convention `pass == 1` includes sacks and scrambles (scrambles have `play_type == "run"`), which is the dropback split you want. Remember `df["pass"]`, not `df.pass`.

- [ ] **Step 1: Write the failing test**

`tests/test_epa.py`:

```python
import pandas as pd

from nfl_game.ratings import epa


def _pbp_fixture():
    """Two teams, one game. BUF offense: 2 pass (+1.0, +0.0), 1 rush (-0.6).
    ARI offense: 1 pass (+0.4), 1 rush (+0.2). Plus rows that must be excluded."""
    return pd.DataFrame(
        {
            "game_id": ["2024_01_ARI_BUF"] * 8,
            "season": [2024] * 8,
            "week": [1] * 8,
            "season_type": ["REG"] * 8,
            "home_team": ["BUF"] * 8,
            "away_team": ["ARI"] * 8,
            "posteam": ["BUF", "BUF", "BUF", "ARI", "ARI", "BUF", "BUF", None],
            "defteam": ["ARI", "ARI", "ARI", "BUF", "BUF", "ARI", "ARI", None],
            "play_type": ["pass", "pass", "run", "pass", "run", "punt", "pass", None],
            "pass": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "rush": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "epa": [1.0, 0.0, -0.6, 0.4, 0.2, 3.0, None, 0.0],
            "success": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, None, 0.0],
        }
    )


def test_one_row_per_offense_per_game():
    out = epa.team_game_epa(_pbp_fixture())
    assert len(out) == 2
    assert set(out["team"]) == {"BUF", "ARI"}


def test_excludes_special_teams_and_null_epa():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    # the punt (epa 3.0) and the null-epa pass are both dropped
    assert buf["n_pass"] == 2
    assert buf["n_rush"] == 1


def test_pass_rush_split_uses_indicator_columns():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    assert buf["epa_pass"] == 0.5          # (1.0 + 0.0) / 2
    assert buf["epa_rush"] == -0.6
    assert buf["epa_play"] == pytest.approx(0.4 / 3)   # (1.0 + 0.0 - 0.6) / 3


def test_opponent_and_home_flag():
    out = epa.team_game_epa(_pbp_fixture())
    buf = out[out["team"] == "BUF"].iloc[0]
    ari = out[out["team"] == "ARI"].iloc[0]
    assert buf["opponent"] == "ARI"
    assert buf["is_home"] == 1
    assert ari["opponent"] == "BUF"
    assert ari["is_home"] == 0


def test_success_rate():
    out = epa.team_game_epa(_pbp_fixture())
    ari = out[out["team"] == "ARI"].iloc[0]
    assert ari["success_rate"] == 1.0


def test_filters_to_regular_season():
    df = _pbp_fixture()
    df["season_type"] = "POST"
    out = epa.team_game_epa(df)
    assert out.empty
```

Add `import pytest` at the top of the file (used by `pytest.approx`).

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_epa.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.ratings.epa'`.

- [ ] **Step 3: Write `team_game_epa` in `src/nfl_game/ratings/epa.py`**

```python
"""Team strength from play-by-play EPA.

team_game_epa reduces raw plays to one row per offense per game. fit_ratings (Task 4)
turns those rows into opponent-adjusted offensive and defensive ratings.
"""

import pandas as pd

TEAM_GAME_COLS = [
    "game_id",
    "season",
    "week",
    "team",
    "opponent",
    "is_home",
    "epa_play",
    "epa_pass",
    "epa_rush",
    "success_rate",
    "n_pass",
    "n_rush",
]


def team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense) with that offense's EPA per play.

    Keeps regular-season scrimmage plays with a non-null EPA. The pass/rush split uses
    nflverse's `pass`/`rush` indicators rather than `play_type`, so scrambles and sacks
    count as dropbacks.
    """
    df = pbp[
        (pbp["season_type"] == "REG")
        & pbp["posteam"].notna()
        & pbp["epa"].notna()
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
    ].copy()

    df["is_pass"] = df["pass"] == 1
    df["is_rush"] = df["rush"] == 1

    grouped = df.groupby(["game_id", "season", "week", "posteam", "defteam"], dropna=True)

    out = grouped.apply(
        lambda g: pd.Series(
            {
                "epa_play": g["epa"].mean(),
                "epa_pass": g.loc[g["is_pass"], "epa"].mean(),
                "epa_rush": g.loc[g["is_rush"], "epa"].mean(),
                "success_rate": g["success"].mean(),
                "n_pass": int(g["is_pass"].sum()),
                "n_rush": int(g["is_rush"].sum()),
            }
        ),
        include_groups=False,
    ).reset_index()

    out = out.rename(columns={"posteam": "team", "defteam": "opponent"})

    home = pbp[["game_id", "home_team"]].dropna().drop_duplicates("game_id")
    out = out.merge(home, on="game_id", how="left")
    out["is_home"] = (out["team"] == out["home_team"]).astype(int)

    out["n_pass"] = out["n_pass"].astype(int)
    out["n_rush"] = out["n_rush"].astype(int)

    return out[TEAM_GAME_COLS].sort_values(["season", "week", "team"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_epa.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Sanity-check against a real season**

```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_pbp; from nfl_game.ratings.epa import team_game_epa; t=team_game_epa(load_pbp([2024], save=False)); print(t.shape); print(t.groupby('team')['epa_play'].mean().sort_values(ascending=False).head(5))"
```

Expected: roughly `(544, 12)` rows. The top-5 EPA/play teams should be recognizable 2024 offenses (BAL, DET, BUF near the top) — if the leaderboard looks random, the join is wrong.

- [ ] **Step 6: Commit**

```bash
git add src/nfl_game/ratings/epa.py tests/test_epa.py
git commit -m "feat: aggregate play-by-play EPA to team-game rows"
```

---

