### Task 6: NGS offensive layer

**Files:**
- Create: `src/nfl_game/ratings/ngs.py`
- Test: `tests/test_ngs.py`

**Interfaces:**
- Consumes: raw NGS frames from `data.nfl.load_ngs`.
- Produces:
  - `team_week_ngs(passing: pd.DataFrame, rushing: pd.DataFrame, receiving: pd.DataFrame) -> pd.DataFrame` with columns `season, week, team` plus the eight metrics below and their `_imputed` flags.
  - `NGS_METRICS: list[str]` — the eight metric column names.

Metrics, all volume-weighted team aggregates:

| Column | Source | Weight |
|---|---|---|
| `cpoe` | passing `completion_percentage_above_expectation` | `attempts` |
| `time_to_throw` | passing `avg_time_to_throw` | `attempts` |
| `air_yards_to_sticks` | passing `avg_air_yards_to_sticks` | `attempts` |
| `aggressiveness` | passing `aggressiveness` | `attempts` |
| `ryoe_per_att` | rushing `rush_yards_over_expected_per_att` | `rush_attempts` |
| `pct_eight_defenders` | rushing `percent_attempts_gte_eight_defenders` | `rush_attempts` |
| `separation` | receiving `avg_separation` | `targets` |
| `yac_oe` | receiving `avg_yac_above_expectation` | `targets` |

**Context for the implementer:** Two traps. First, **`week == 0` rows are season aggregates**, not games — filter them or every team's totals double-count. Second, NGS has qualifier thresholds: measured on 2024, passing covers 539 of 544 team-games but **rushing only covers 468 (86%)**. Missing values are imputed with the league-week mean and flagged with a `<metric>_imputed` column, so the model can see that a value was a guess rather than being silently fed one.

- [ ] **Step 1: Write the failing test**

`tests/test_ngs.py`:

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_ngs.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.ratings.ngs'`.

- [ ] **Step 3: Write `src/nfl_game/ratings/ngs.py`**

```python
"""Next Gen Stats aggregated to team-weeks (offense only; NGS has no defensive table).

CPOE, rush yards over expected, and separation stabilize faster than box-score yardage,
which is where this layer earns its place: it identifies real team quality earlier in a
season than results alone do.

NGS applies qualifier thresholds, so coverage is incomplete — measured on 2024, passing
covers 539 of 544 team-games but rushing only 468. Missing values are imputed with the
league-week mean and flagged, so the model is told when a number is a guess.
"""

import pandas as pd

PASSING_MAP = {
    "completion_percentage_above_expectation": "cpoe",
    "avg_time_to_throw": "time_to_throw",
    "avg_air_yards_to_sticks": "air_yards_to_sticks",
    "aggressiveness": "aggressiveness",
}
RUSHING_MAP = {
    "rush_yards_over_expected_per_att": "ryoe_per_att",
    "percent_attempts_gte_eight_defenders": "pct_eight_defenders",
}
RECEIVING_MAP = {
    "avg_separation": "separation",
    "avg_yac_above_expectation": "yac_oe",
}

NGS_METRICS = list(PASSING_MAP.values()) + list(RUSHING_MAP.values()) + list(RECEIVING_MAP.values())


def _weighted_team_week(df: pd.DataFrame, mapping: dict[str, str], weight_col: str) -> pd.DataFrame:
    """Collapse player rows to one volume-weighted row per team-week."""
    d = df[(df["season_type"] == "REG") & (df["week"] > 0)].copy()
    if d.empty:
        return pd.DataFrame(columns=["season", "week", "team", *mapping.values()])

    d = d.rename(columns={"team_abbr": "team"})
    d["_w"] = d[weight_col].fillna(0.0)

    out = []
    for (season, week, team), g in d.groupby(["season", "week", "team"]):
        row = {"season": season, "week": week, "team": team}
        total = g["_w"].sum()
        for src, dest in mapping.items():
            if total > 0 and g[src].notna().any():
                valid = g[g[src].notna()]
                vw = valid["_w"].sum()
                row[dest] = (valid[src] * valid["_w"]).sum() / vw if vw > 0 else None
            else:
                row[dest] = None
        out.append(row)
    return pd.DataFrame(out)


def team_week_ngs(
    passing: pd.DataFrame, rushing: pd.DataFrame, receiving: pd.DataFrame
) -> pd.DataFrame:
    """One row per team-week with all eight NGS metrics, imputed and flagged."""
    p = _weighted_team_week(passing, PASSING_MAP, "attempts")
    r = _weighted_team_week(rushing, RUSHING_MAP, "rush_attempts")
    c = _weighted_team_week(receiving, RECEIVING_MAP, "targets")

    keys = ["season", "week", "team"]
    out = p
    for other in (r, c):
        out = out.merge(other, on=keys, how="outer")

    for metric in NGS_METRICS:
        if metric not in out.columns:
            out[metric] = None
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_imputed"] = out[metric].isna().astype(int)
        league_mean = out.groupby(["season", "week"])[metric].transform("mean")
        out[metric] = out[metric].fillna(league_mean).fillna(out[metric].mean()).fillna(0.0)

    ordered = keys + NGS_METRICS + [f"{m}_imputed" for m in NGS_METRICS]
    return out[ordered].sort_values(keys).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_ngs.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Verify coverage against real 2024 data**

```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_ngs; from nfl_game.ratings.ngs import team_week_ngs; p,r,c=[load_ngs([2024],s,save=False) for s in ('passing','rushing','receiving')]; t=team_week_ngs(p,r,c); print(t.shape); print(t[[col for col in t.columns if col.endswith('_imputed')]].mean().round(3))"
```

Expected: roughly 540 rows. `cpoe_imputed` near 0.01, `ryoe_per_att_imputed` near 0.14 — matching the measured 99% / 86% coverage. A rushing imputation rate far above 0.2 means the join key is wrong.

- [ ] **Step 6: Commit**

```bash
git add src/nfl_game/ratings/ngs.py tests/test_ngs.py
git commit -m "feat: aggregate Next Gen Stats to team-weeks with imputation flags"
```

---

