### Task 7: Game feature assembly

**Files:**
- Create: `src/nfl_game/model/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `schedules` (from `data.nfl.load_schedules`), `ratings_by_week` output, `team_week_ngs` output.
- Produces:
  - `build_game_features(schedules, ratings, ngs, ngs_halflife=4.0) -> pd.DataFrame`
  - `FEATURE_COLS: list[str]` — the model input columns, in fixed order.
  - `TARGET_COLS = ["margin", "total_points"]`

**Feature list (fixed order):**

```
off_pass_edge_home, off_rush_edge_home, off_pass_edge_away, off_rush_edge_away,
net_rating_diff, rest_diff, is_dome, temp_outdoor, wind_outdoor, div_game,
cpoe_diff, ryoe_diff, separation_diff, ngs_imputed_any
```

Where `off_pass_edge_home = home off_rating_pass - away def_rating_pass` (and symmetrically), and `net_rating_diff` is the home team's overall off+def rating advantage.

**Context for the implementer:** Ratings must be joined **as of that game's week** — the row in `ratings` with matching `(season, week, team)`, which Task 5 already guarantees was built from strictly-prior data. NGS features use a trailing decay-weighted mean over prior weeks, never the current week.

For dome games `temp` and `wind` are null in the source; set both to 0 and rely on `is_dome` to carry that information. Do not impute a temperature for indoor games.

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.model.features'`.

- [ ] **Step 3: Write `src/nfl_game/model/features.py`**

```python
"""Game-level feature assembly.

Joins as-of team ratings and trailing NGS onto each scheduled game. Games that have not
been played yet are kept with null targets, so the same function serves both training
and prediction of an upcoming slate.
"""

import numpy as np
import pandas as pd

from nfl_game.ratings.ngs import NGS_METRICS

FEATURE_COLS = [
    "off_pass_edge_home",
    "off_rush_edge_home",
    "off_pass_edge_away",
    "off_rush_edge_away",
    "net_rating_diff",
    "rest_diff",
    "is_dome",
    "temp_outdoor",
    "wind_outdoor",
    "div_game",
    "cpoe_diff",
    "ryoe_diff",
    "separation_diff",
    "ngs_imputed_any",
]

TARGET_COLS = ["margin", "total_points"]

DOME_ROOFS = {"dome", "closed"}


def _trailing_ngs(ngs: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Decay-weighted mean of each team's NGS over weeks strictly before each week."""
    frames = []
    for (season, team), g in ngs.groupby(["season", "team"]):
        g = g.sort_values("week")
        weeks = g["week"].to_numpy()
        for i, week in enumerate(weeks):
            prior = g.iloc[:i]
            row = {"season": season, "team": team, "week": week}
            if prior.empty:
                for m in NGS_METRICS:
                    row[f"trail_{m}"] = np.nan
                row["trail_imputed_any"] = 1
            else:
                age = week - prior["week"].to_numpy()
                w = 0.5 ** (age / halflife)
                for m in NGS_METRICS:
                    row[f"trail_{m}"] = float(np.average(prior[m].to_numpy(), weights=w))
                flags = [f"{m}_imputed" for m in NGS_METRICS if f"{m}_imputed" in prior.columns]
                row["trail_imputed_any"] = int(prior[flags].to_numpy().max()) if flags else 0
            frames.append(row)
    out = pd.DataFrame(frames)
    for m in NGS_METRICS:
        out[f"trail_{m}"] = out[f"trail_{m}"].fillna(0.0)
    return out


def build_game_features(
    schedules: pd.DataFrame,
    ratings: pd.DataFrame,
    ngs: pd.DataFrame,
    ngs_halflife: float = 4.0,
) -> pd.DataFrame:
    """One row per regular-season game with model features and targets."""
    g = schedules[schedules["game_type"] == "REG"].copy()

    trail = _trailing_ngs(ngs, ngs_halflife)

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        r = ratings.rename(columns={"team": team_col})
        r = r.rename(columns={c: f"{side}_{c}" for c in r.columns if c.startswith(("off_", "def_"))})
        g = g.merge(r, on=["season", "week", team_col], how="left")

        t = trail.rename(columns={"team": team_col})
        t = t.rename(columns={c: f"{side}_{c}" for c in t.columns if c.startswith("trail_")})
        g = g.merge(t, on=["season", "week", team_col], how="left")

    g["off_pass_edge_home"] = g["home_off_rating_pass"] - g["away_def_rating_pass"]
    g["off_rush_edge_home"] = g["home_off_rating_rush"] - g["away_def_rating_rush"]
    g["off_pass_edge_away"] = g["away_off_rating_pass"] - g["home_def_rating_pass"]
    g["off_rush_edge_away"] = g["away_off_rating_rush"] - g["home_def_rating_rush"]
    g["net_rating_diff"] = (g["home_off_rating"] + g["home_def_rating"]) - (
        g["away_off_rating"] + g["away_def_rating"]
    )

    g["rest_diff"] = g["home_rest"] - g["away_rest"]
    g["is_dome"] = g["roof"].isin(DOME_ROOFS).astype(int)
    g["temp_outdoor"] = np.where(g["is_dome"] == 1, 0.0, g["temp"].fillna(60.0))
    g["wind_outdoor"] = np.where(g["is_dome"] == 1, 0.0, g["wind"].fillna(0.0))
    g["div_game"] = g["div_game"].fillna(0).astype(int)

    g["cpoe_diff"] = g["home_trail_cpoe"] - g["away_trail_cpoe"]
    g["ryoe_diff"] = g["home_trail_ryoe_per_att"] - g["away_trail_ryoe_per_att"]
    g["separation_diff"] = g["home_trail_separation"] - g["away_trail_separation"]
    g["ngs_imputed_any"] = (
        g[["home_trail_imputed_any", "away_trail_imputed_any"]].fillna(1).max(axis=1).astype(int)
    )

    g["margin"] = g["result"]
    g["total_points"] = g["total"]

    keep = [
        "game_id", "season", "week", "home_team", "away_team",
        "spread_line", "total_line", *FEATURE_COLS, *TARGET_COLS,
    ]
    out = g[keep].copy()
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)
    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_features.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_game/model/features.py tests/test_features.py
git commit -m "feat: assemble game-level features from ratings and trailing NGS"
```

---

