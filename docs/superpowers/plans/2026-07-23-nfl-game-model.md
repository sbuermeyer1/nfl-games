# NFL Game Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a market-blind NFL game model that predicts each game's margin and total from EPA + Next Gen Stats team ratings, then compares those predictions to the closing line with calibrated cover probabilities.

**Architecture:** Four layers with strictly one-directional flow — `data → ratings → model → market`. Team strength comes from a ridge regression of play-level EPA on offense/defense team dummies (which removes schedule effects), recency-weighted with exponential decay. NGS adds offensive quality features from 2016 on. A separate calibration layer converts model-vs-market point gaps into probabilities.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, nflreadpy, pyarrow, pytest, ruff.

## Global Constraints

- Package lives at `src/nfl_game/`, importable as `nfl_game`. Repo root is `~/Documents/NFL Game Model`.
- Python `>=3.11`. Line length 100 (ruff).
- **No reverse dependencies.** `data` must not import from `ratings`; `ratings` must not import from `model`; `model` must not import from `market`.
- **Do not import from `nfl_ffm`** (the fantasy project). Conventions are copied, code is not.
- All `nflreadpy` calls return Polars — convert with `.to_pandas()` immediately at the data layer. Nothing downstream of `data/` touches Polars.
- **`pass` is a Python keyword.** The pbp indicator column must be accessed as `df["pass"]`, never `df.pass`.
- Training window is **2016+** (NGS era). Never widen it without an explicit decision.
- **No test may hit the network.** All tests use synthetic in-memory DataFrames.
- Windows interpreter path: `.\.venv\Scripts\python.exe`.
- Every rating/feature function takes an as-of cutoff and must use **strictly prior** data. This is the project's central correctness property.

---

### Task 1: Repo scaffold and packaging

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/nfl_game/__init__.py`, `src/nfl_game/paths.py`, `data/raw/.gitkeep`, `data/processed/.gitkeep`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `nfl_game.paths.PROJECT_ROOT`, `RAW_DIR`, `PROCESSED_DIR` — all `pathlib.Path`.

- [ ] **Step 1: Create the directory skeleton**

```bash
cd ~/Documents/"NFL Game Model"
mkdir -p src/nfl_game/{data,ratings,model,market} tests scripts data/raw data/processed
touch data/raw/.gitkeep data/processed/.gitkeep
touch src/nfl_game/__init__.py
touch src/nfl_game/data/__init__.py src/nfl_game/ratings/__init__.py
touch src/nfl_game/model/__init__.py src/nfl_game/market/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "nfl-game"
version = "0.1.0"
description = "NFL game model for spreads and totals"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "scipy>=1.13",
    "nflreadpy>=0.1",
    "pyarrow>=16.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
.pytest_cache/
.ruff_cache/

# raw/processed data can be large and regeneratable — keep out of git by default
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

.ipynb_checkpoints/
```

- [ ] **Step 4: Write the failing test**

`tests/test_smoke.py`:

```python
from nfl_game import paths


def test_paths_resolve_under_project_root():
    assert paths.RAW_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.PROCESSED_DIR == paths.PROJECT_ROOT / "data" / "processed"


def test_data_dirs_exist():
    assert paths.RAW_DIR.is_dir()
    assert paths.PROCESSED_DIR.is_dir()
```

- [ ] **Step 5: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game'` (venv not built yet) or `ImportError` on `paths`.

- [ ] **Step 6: Write `src/nfl_game/paths.py`**

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
```

- [ ] **Step 7: Create the venv and install**

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 8: Run the test and confirm it passes**

```
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v
```

Expected: 2 passed.

- [ ] **Step 9: Write `README.md`**

```markdown
# NFL Game Model

Predicts NFL game margins and totals from EPA + Next Gen Stats team ratings, then compares
those predictions against the closing spread and total.

The model is **market-blind**: it never sees the betting line when predicting. A separate
layer compares model output to the market and reports calibrated cover probabilities.

Data source: [`nflreadpy`](https://nflreadpy.nflverse.com/). No API key required.

Design: `docs/superpowers/specs/2026-07-23-nfl-game-model-design.md`

## Setup

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

## Tests

    .\.venv\Scripts\python.exe -m pytest
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore README.md src/ tests/ data/
git commit -m "feat: scaffold nfl_game package and paths"
```

---

### Task 2: Data layer

**Files:**
- Create: `src/nfl_game/data/nfl.py`
- Test: `tests/test_data_nfl.py`

**Interfaces:**
- Consumes: `nfl_game.paths.RAW_DIR`.
- Produces:
  - `load_schedules(seasons: list[int] | None = None, save: bool = True) -> pd.DataFrame`
  - `load_pbp(seasons: list[int], save: bool = True) -> pd.DataFrame`
  - `load_ngs(seasons: list[int], stat_type: str, save: bool = True) -> pd.DataFrame`
  - `_seasons_label(seasons: list[int]) -> str`

**Context for the implementer:** `nflreadpy` returns Polars DataFrames. Convert immediately. `load_schedules()` with no seasons returns all seasons 1999–2026 including future games with null results — that is intentional and needed for predicting upcoming slates.

- [ ] **Step 1: Write the failing test**

`tests/test_data_nfl.py`:

```python
import pandas as pd
import pytest

from nfl_game.data import nfl


def test_seasons_label_single():
    assert nfl._seasons_label([2024]) == "2024"


def test_seasons_label_range():
    assert nfl._seasons_label([2024, 2016, 2020]) == "2016-2024"


def test_load_pbp_converts_and_saves(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"game_id": ["2024_01_ARI_BUF"], "epa": [0.5]})

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    out = nfl.load_pbp([2024])

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["game_id", "epa"]
    assert (tmp_path / "pbp_2024.parquet").exists()


def test_load_pbp_can_skip_save(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"game_id": ["x"], "epa": [0.1]})

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    nfl.load_pbp([2024], save=False)

    assert list(tmp_path.iterdir()) == []


def test_load_ngs_rejects_bad_stat_type():
    with pytest.raises(ValueError, match="stat_type"):
        nfl.load_ngs([2024], stat_type="kicking")
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_data_nfl.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.data.nfl'`.

- [ ] **Step 3: Write `src/nfl_game/data/nfl.py`**

```python
"""NFL data ingestion via nflreadpy (nflverse public data releases).

nflreadpy returns Polars DataFrames; everything here converts to pandas so nothing
downstream of this module has to know Polars exists.
"""

import nflreadpy
import pandas as pd

from nfl_game.paths import RAW_DIR

NGS_STAT_TYPES = ("passing", "rushing", "receiving")


def _seasons_label(seasons: list[int]) -> str:
    seasons = sorted(seasons)
    return f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])


def load_schedules(seasons: list[int] | None = None, save: bool = True) -> pd.DataFrame:
    """Game schedule, results, and closing betting lines.

    Passing seasons=None loads every season (1999+), including future games whose
    result/total are null but whose spread_line/total_line may already be posted.
    """
    df = nflreadpy.load_schedules().to_pandas()
    if seasons is not None:
        df = df[df["season"].isin(seasons)].reset_index(drop=True)
    if save:
        label = _seasons_label(seasons) if seasons else "all"
        df.to_parquet(RAW_DIR / f"schedules_{label}.parquet")
    return df


def load_pbp(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Play-by-play with EPA. Large: roughly 50k rows and 372 columns per season."""
    df = nflreadpy.load_pbp(seasons).to_pandas()
    if save:
        df.to_parquet(RAW_DIR / f"pbp_{_seasons_label(seasons)}.parquet")
    return df


def load_ngs(seasons: list[int], stat_type: str, save: bool = True) -> pd.DataFrame:
    """Next Gen Stats, 2016+ only. stat_type is one of passing/rushing/receiving.

    Note: rows with week == 0 are season aggregates, not week-zero games. Callers
    doing weekly joins must filter them out.
    """
    if stat_type not in NGS_STAT_TYPES:
        raise ValueError(f"stat_type must be one of {NGS_STAT_TYPES}, got {stat_type!r}")
    df = nflreadpy.load_nextgen_stats(seasons=seasons, stat_type=stat_type).to_pandas()
    if save:
        df.to_parquet(RAW_DIR / f"ngs_{stat_type}_{_seasons_label(seasons)}.parquet")
    return df
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_data_nfl.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Verify against real data once, by hand**

```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_schedules; d=load_schedules(save=False); print(d.shape); print(d[['spread_line','total_line','result','total']].notna().sum())"
```

Expected: shape around `(7548, 46)`; all four columns show thousands of non-null values.

- [ ] **Step 6: Commit**

```bash
git add src/nfl_game/data/nfl.py tests/test_data_nfl.py
git commit -m "feat: add nflreadpy data loaders for schedules, pbp, and NGS"
```

---

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

### Task 8: Margin and total models

**Files:**
- Create: `src/nfl_game/model/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `features.FEATURE_COLS`, `features.TARGET_COLS`.
- Produces:
  - `GameModel(estimator: str = "ridge", alpha: float = 1.0)` with `.fit(train_df) -> GameModel` and `.predict(df) -> pd.DataFrame` returning columns `game_id, model_margin, model_total`.
  - `ESTIMATORS: dict[str, callable]` mapping `"ridge"` and `"gbm"` to factory functions.

**Context for the implementer:** One class wrapping two fitted regressors — one for margin, one for total. Keeping both behind a single object means the backtest, calibration, and slate code never care which estimator won. Rows with a null target are dropped at fit time; rows with null targets are still predictable.

Ridge is the baseline and default. GBM is the challenger. Task 9 decides between them with evidence.

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import ESTIMATORS, GameModel


def _train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["game_id"] = [f"g{i}" for i in range(n)]
    # margin is a known linear function of two features plus noise
    df["margin"] = 3.0 * df["net_rating_diff"] + 1.5 * df["rest_diff"] + rng.normal(scale=0.5, size=n)
    df["total_points"] = 44.0 + 2.0 * df["off_pass_edge_home"] + rng.normal(scale=0.5, size=n)
    return df


def test_predict_returns_expected_columns():
    m = GameModel().fit(_train())
    out = m.predict(_train(n=10, seed=1))
    assert list(out.columns) == ["game_id", "model_margin", "model_total"]
    assert len(out) == 10


def test_recovers_a_known_linear_signal():
    train = _train()
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    pred = m.predict(train)
    corr = np.corrcoef(pred["model_margin"], train["margin"])[0, 1]
    assert corr > 0.95


def test_total_model_is_separate_from_margin():
    train = _train()
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    pred = m.predict(train)
    assert pred["model_total"].mean() == pytest.approx(44.0, abs=1.0)


def test_gbm_estimator_also_fits():
    train = _train()
    m = GameModel(estimator="gbm").fit(train)
    pred = m.predict(train)
    assert pred["model_margin"].notna().all()


def test_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        GameModel(estimator="magic")


def test_rows_with_null_targets_are_dropped_at_fit():
    train = _train()
    train.loc[:50, "margin"] = np.nan
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    assert m.n_train_margin_ == len(train) - 51


def test_can_predict_rows_with_null_targets():
    train = _train()
    future = _train(n=5, seed=2)
    future[["margin", "total_points"]] = np.nan
    m = GameModel().fit(train)
    out = m.predict(future)
    assert out["model_margin"].notna().all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GameModel().predict(_train(n=3))


def test_estimators_registry_exposes_both():
    assert set(ESTIMATORS) == {"ridge", "gbm"}
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.model.predict'`.

- [ ] **Step 3: Write `src/nfl_game/model/predict.py`**

```python
"""Margin and total regressors behind a single interface.

Ridge is the honest baseline; gradient boosting is the challenger. Task 9's backtest
picks between them on evidence. Everything downstream consumes GameModel and never
needs to know which one is in use.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from nfl_game.model.features import FEATURE_COLS

ESTIMATORS = {
    "ridge": lambda alpha: Ridge(alpha=alpha),
    "gbm": lambda alpha: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=0
    ),
}


class GameModel:
    """Fits one regressor for game margin and one for total points."""

    def __init__(self, estimator: str = "ridge", alpha: float = 1.0):
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}")
        self.estimator = estimator
        self.alpha = alpha
        self._margin = None
        self._total = None
        self.n_train_margin_ = 0
        self.n_train_total_ = 0

    def fit(self, train: pd.DataFrame) -> "GameModel":
        m = train[train["margin"].notna()]
        t = train[train["total_points"].notna()]
        self.n_train_margin_ = len(m)
        self.n_train_total_ = len(t)

        self._margin = ESTIMATORS[self.estimator](self.alpha)
        self._margin.fit(m[FEATURE_COLS].to_numpy(dtype=float), m["margin"].to_numpy(dtype=float))

        self._total = ESTIMATORS[self.estimator](self.alpha)
        self._total.fit(
            t[FEATURE_COLS].to_numpy(dtype=float), t["total_points"].to_numpy(dtype=float)
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._margin is None or self._total is None:
            raise RuntimeError("call fit() before predict()")
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "game_id": df["game_id"].to_numpy(),
                "model_margin": self._margin.predict(X),
                "model_total": self._total.predict(X),
            }
        )
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_game/model/predict.py tests/test_predict.py
git commit -m "feat: margin and total regressors with ridge and gbm estimators"
```

---

### Task 9: Walk-forward backtest

**Files:**
- Create: `src/nfl_game/backtest.py`, `scripts/build_dataset.py`, `scripts/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `GameModel`, `features.build_game_features`.
- Produces:
  - `walk_forward(features_df, test_seasons, estimator="ridge", alpha=1.0) -> pd.DataFrame` — per-game predictions for each test season, each trained only on prior seasons.
  - `evaluate(preds: pd.DataFrame) -> dict` — MAE and ATS metrics vs the market.
  - `market_comparison_regression(preds) -> dict` — the decisive test.

**Context for the implementer:** `evaluate` must report the model's MAE **and the market's MAE on the same games**, because the market is the benchmark. `market_comparison_regression` regresses actual margin on both the market line and the model line; if `model_coef` is near zero, the model adds nothing over the market. Report it honestly rather than burying it.

Sign convention for `spread_line` in nflverse: it is the **home team's** line, positive when the home team is favored. So the market's implied home margin is `spread_line` itself.

- [ ] **Step 1: Write the failing test**

`tests/test_backtest.py`:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.backtest import evaluate, market_comparison_regression, walk_forward
from nfl_game.model.features import FEATURE_COLS


def _features(seasons=(2021, 2022, 2023), n_per=100, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in seasons:
        df = pd.DataFrame({c: rng.normal(size=n_per) for c in FEATURE_COLS})
        df["game_id"] = [f"{s}_{i}" for i in range(n_per)]
        df["season"] = s
        df["week"] = rng.integers(1, 18, n_per)
        df["margin"] = 3.0 * df["net_rating_diff"] + rng.normal(scale=3.0, size=n_per)
        df["total_points"] = 44.0 + rng.normal(scale=5.0, size=n_per)
        df["spread_line"] = df["margin"] + rng.normal(scale=2.0, size=n_per)
        df["total_line"] = df["total_points"] + rng.normal(scale=2.0, size=n_per)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_walk_forward_only_scores_test_seasons():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    assert sorted(out["season"].unique()) == [2022, 2023]


def test_walk_forward_never_trains_on_the_test_season():
    """A model fit on its own test season scores better in-sample. Honest
    walk-forward error must be strictly worse than that leaked baseline —
    no slack, because any slack is exactly where a real leak would hide."""
    from nfl_game.model.predict import GameModel

    feats = _features()
    honest = walk_forward(feats, test_seasons=[2023], alpha=0.01)
    mae_honest = (honest["model_margin"] - honest["margin"]).abs().mean()

    test_rows = feats[feats["season"] == 2023]
    leaked_pred = GameModel(alpha=0.01).fit(test_rows).predict(test_rows)
    mae_leaked = np.abs(
        leaked_pred["model_margin"].to_numpy() - test_rows["margin"].to_numpy()
    ).mean()

    assert mae_honest > mae_leaked


def test_walk_forward_skips_season_with_no_prior_data():
    out = walk_forward(_features(), test_seasons=[2021, 2022])
    assert sorted(out["season"].unique()) == [2022]


def test_evaluate_reports_model_and_market_mae():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    m = evaluate(out)
    assert "margin_mae" in m and "market_margin_mae" in m
    assert "total_mae" in m and "market_total_mae" in m
    assert m["margin_mae"] > 0


def test_evaluate_reports_ats_hit_rate_and_n():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    m = evaluate(out)
    assert 0.0 <= m["ats_hit_rate"] <= 1.0
    assert m["ats_n"] > 0


def test_evaluate_excludes_pushes_from_ats():
    preds = pd.DataFrame(
        {
            "game_id": ["a", "b"], "season": [2023, 2023], "week": [1, 1],
            "margin": [7.0, 3.0], "total_points": [44.0, 44.0],
            "model_margin": [10.0, 1.0], "model_total": [45.0, 45.0],
            "spread_line": [7.0, 1.0], "total_line": [44.0, 44.0],
        }
    )
    m = evaluate(preds)
    # game "a" is an exact push against the spread and must not be counted
    assert m["ats_n"] == 1


def test_market_regression_returns_both_coefficients():
    out = walk_forward(_features(), test_seasons=[2022, 2023])
    r = market_comparison_regression(out)
    assert "market_coef" in r and "model_coef" in r
    # the synthetic market line is a near-perfect signal, so it must dominate
    assert r["market_coef"] > r["model_coef"]
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.backtest'`.

- [ ] **Step 3: Write `src/nfl_game/backtest.py`**

```python
"""Walk-forward evaluation against the market.

The market is the benchmark, not a strawman. Every accuracy number is reported next to
the closing line's own error on the same games, and market_comparison_regression answers
the only question that really matters: does the model add anything the line doesn't
already contain?
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nfl_game.model.predict import GameModel


def walk_forward(
    features_df: pd.DataFrame,
    test_seasons: list[int],
    estimator: str = "ridge",
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Predict each test season using a model trained only on strictly earlier seasons."""
    frames = []
    for season in sorted(test_seasons):
        train = features_df[features_df["season"] < season]
        test = features_df[features_df["season"] == season]
        if train.empty or test.empty:
            continue
        model = GameModel(estimator=estimator, alpha=alpha).fit(train)
        preds = model.predict(test)
        merged = test.merge(preds, on="game_id", how="left")
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=[*features_df.columns, "model_margin", "model_total"])
    return pd.concat(frames, ignore_index=True)


def evaluate(preds: pd.DataFrame) -> dict:
    """Accuracy and ATS metrics, each paired with the market's own performance.

    ATS: the model takes the home side when it predicts a bigger home margin than the
    line. Exact pushes are excluded, which is why ats_n is reported alongside the rate.
    Break-even at standard -110 juice is 52.4%.
    """
    d = preds[preds["margin"].notna()].copy()

    out = {
        "n_games": int(len(d)),
        "margin_mae": float((d["model_margin"] - d["margin"]).abs().mean()),
        "market_margin_mae": float((d["spread_line"] - d["margin"]).abs().mean()),
        "total_mae": float((d["model_total"] - d["total_points"]).abs().mean()),
        "market_total_mae": float((d["total_line"] - d["total_points"]).abs().mean()),
    }

    played = d[d["margin"] != d["spread_line"]]
    if len(played):
        picks_home = played["model_margin"] > played["spread_line"]
        home_covered = played["margin"] > played["spread_line"]
        out["ats_hit_rate"] = float((picks_home == home_covered).mean())
        out["ats_n"] = int(len(played))
    else:
        out["ats_hit_rate"] = float("nan")
        out["ats_n"] = 0

    ou = d[d["total_points"] != d["total_line"]]
    if len(ou):
        picks_over = ou["model_total"] > ou["total_line"]
        went_over = ou["total_points"] > ou["total_line"]
        out["ou_hit_rate"] = float((picks_over == went_over).mean())
        out["ou_n"] = int(len(ou))
    else:
        out["ou_hit_rate"] = float("nan")
        out["ou_n"] = 0

    out["ats_breakeven"] = 0.524
    return out


def market_comparison_regression(preds: pd.DataFrame) -> dict:
    """Regress actual margin on both the market line and the model line.

    If model_coef is indistinguishable from zero, the model contributes nothing beyond
    what the closing line already knows. This is the decisive test.
    """
    d = preds[preds["margin"].notna()]
    X = d[["spread_line", "model_margin"]].to_numpy(dtype=float)
    y = d["margin"].to_numpy(dtype=float)
    fit = LinearRegression().fit(X, y)
    return {
        "market_coef": float(fit.coef_[0]),
        "model_coef": float(fit.coef_[1]),
        "intercept": float(fit.intercept_),
        "r2": float(fit.score(X, y)),
        "n": int(len(d)),
    }


def ats_by_threshold(preds: pd.DataFrame, thresholds=(0, 1, 2, 3, 4, 6)) -> pd.DataFrame:
    """ATS hit rate bucketed by how far the model disagrees with the line."""
    d = preds[preds["margin"].notna()].copy()
    d["edge"] = (d["model_margin"] - d["spread_line"]).abs()
    rows = []
    for t in thresholds:
        sub = d[(d["edge"] >= t) & (d["margin"] != d["spread_line"])]
        if sub.empty:
            rows.append({"min_edge": t, "n": 0, "hit_rate": np.nan})
            continue
        picks_home = sub["model_margin"] > sub["spread_line"]
        home_covered = sub["margin"] > sub["spread_line"]
        rows.append(
            {"min_edge": t, "n": int(len(sub)), "hit_rate": float((picks_home == home_covered).mean())}
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Write `scripts/build_dataset.py`**

```python
"""Build and cache the full game-features dataset. Run before backtesting."""

import argparse

from nfl_game.data.nfl import load_ngs, load_pbp, load_schedules
from nfl_game.model.features import build_game_features
from nfl_game.paths import PROCESSED_DIR
from nfl_game.ratings.build import ratings_by_week
from nfl_game.ratings.epa import team_game_epa
from nfl_game.ratings.ngs import team_week_ngs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-season", type=int, default=2016)
    ap.add_argument("--end-season", type=int, default=2025)
    args = ap.parse_args()

    seasons = list(range(args.start_season, args.end_season + 1))
    print(f"loading pbp for {seasons[0]}-{seasons[-1]} (this takes a few minutes)...")
    pbp = load_pbp(seasons)
    team_games = team_game_epa(pbp)

    print("building as-of ratings...")
    ratings = ratings_by_week(team_games, seasons=seasons)

    print("building NGS team-weeks...")
    ngs = team_week_ngs(
        load_ngs(seasons, "passing"),
        load_ngs(seasons, "rushing"),
        load_ngs(seasons, "receiving"),
    )

    print("assembling features...")
    feats = build_game_features(load_schedules(), ratings, ngs)
    feats = feats[feats["season"].isin(seasons)]

    path = PROCESSED_DIR / "game_features.parquet"
    feats.to_parquet(path)
    print(f"wrote {len(feats)} games to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write `scripts/backtest.py`**

```python
"""Walk-forward backtest report."""

import argparse

import pandas as pd

from nfl_game.backtest import ats_by_threshold, evaluate, market_comparison_regression, walk_forward
from nfl_game.paths import PROCESSED_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", default="2021-2025")
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "gbm"])
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    lo, _, hi = args.test_seasons.partition("-")
    seasons = list(range(int(lo), int(hi or lo) + 1))

    feats = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    preds = walk_forward(feats, seasons, estimator=args.estimator, alpha=args.alpha)

    m = evaluate(preds)
    print(f"\n=== {args.estimator} | test seasons {seasons[0]}-{seasons[-1]} ===")
    print(f"games:            {m['n_games']}")
    print(f"margin MAE:       {m['margin_mae']:.3f}   market: {m['market_margin_mae']:.3f}")
    print(f"total  MAE:       {m['total_mae']:.3f}   market: {m['market_total_mae']:.3f}")
    print(f"ATS hit rate:     {m['ats_hit_rate']:.4f}  (n={m['ats_n']}, break-even 0.5240)")
    print(f"O/U hit rate:     {m['ou_hit_rate']:.4f}  (n={m['ou_n']})")

    print("\n--- ATS by edge threshold ---")
    print(ats_by_threshold(preds).to_string(index=False))

    r = market_comparison_regression(preds)
    print("\n--- does the model add anything to the line? ---")
    print(f"market coef: {r['market_coef']:.4f}")
    print(f"model  coef: {r['model_coef']:.4f}   <- near zero means it adds nothing")
    print(f"r2: {r['r2']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the real backtest**

```
.\.venv\Scripts\python.exe scripts\build_dataset.py --start-season 2016 --end-season 2025
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025 --estimator gbm
```

Expected and how to read it: market margin MAE lands around 9.8–10.3 points. A model MAE in the same range is a good result. **A model MAE far below the market's, or an ATS hit rate above roughly 0.56, is a leak, not an edge** — stop and audit the as-of joins before believing it. Record which estimator wins; that choice feeds Task 11's default.

- [ ] **Step 8: Commit**

```bash
git add src/nfl_game/backtest.py scripts/build_dataset.py scripts/backtest.py tests/test_backtest.py
git commit -m "feat: walk-forward backtest with market benchmark and leak checks"
```

---

### Task 10: Probability calibration

**Files:**
- Create: `src/nfl_game/model/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `walk_forward` output.
- Produces:
  - `Calibrator()` with `.fit(preds) -> Calibrator` and `.predict(preds) -> pd.DataFrame` returning `game_id, cover_prob, over_prob`.
  - `brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float`
  - `reliability_table(probs, outcomes, bins=5) -> pd.DataFrame`

**Context for the implementer:** This converts a raw point disagreement into a probability with an empirical basis. Fit logistic regression on a single feature — `model_margin - spread_line` — against whether the home team actually covered. Same shape for totals.

Critically, the calibrator must be fit on **walk-forward (out-of-sample) predictions**, not in-sample ones. In-sample gaps are systematically overconfident and would produce probabilities that look sharp and are wrong.

`cover_prob` is always the probability that the **home team covers**.

- [ ] **Step 1: Write the failing test**

`tests/test_calibrate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.model.calibrate import Calibrator, brier_score, reliability_table


def _preds(n=800, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.normal(scale=6.0, size=n)
    edge = rng.normal(scale=3.0, size=n)
    margin = spread + edge * 0.5 + rng.normal(scale=10.0, size=n)
    total_line = rng.normal(loc=45, scale=4.0, size=n)
    t_edge = rng.normal(scale=3.0, size=n)
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "spread_line": spread,
            "model_margin": spread + edge,
            "margin": margin,
            "total_line": total_line,
            "model_total": total_line + t_edge,
            "total_points": total_line + t_edge * 0.5 + rng.normal(scale=9.0, size=n),
        }
    )


def test_predict_returns_expected_columns():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=20, seed=1))
    assert list(out.columns) == ["game_id", "cover_prob", "over_prob"]


def test_probabilities_are_in_range():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=100, seed=1))
    assert out["cover_prob"].between(0, 1).all()
    assert out["over_prob"].between(0, 1).all()


def test_bigger_edge_means_higher_cover_probability():
    c = Calibrator().fit(_preds())
    df = _preds(n=2, seed=3)
    df["spread_line"] = [0.0, 0.0]
    df["model_margin"] = [1.0, 7.0]
    out = c.predict(df)
    assert out.iloc[1]["cover_prob"] > out.iloc[0]["cover_prob"]


def test_zero_edge_is_near_a_coin_flip():
    c = Calibrator().fit(_preds())
    df = _preds(n=1, seed=4)
    df["spread_line"] = [0.0]
    df["model_margin"] = [0.0]
    assert c.predict(df).iloc[0]["cover_prob"] == pytest.approx(0.5, abs=0.08)


def test_brier_score_rewards_accuracy():
    outcomes = np.array([1, 1, 0, 0])
    good = np.array([0.9, 0.8, 0.2, 0.1])
    bad = np.array([0.1, 0.2, 0.8, 0.9])
    assert brier_score(good, outcomes) < brier_score(bad, outcomes)


def test_reliability_table_shape():
    c = Calibrator().fit(_preds())
    p = _preds(n=400, seed=5)
    out = c.predict(p)
    covered = (p["margin"] > p["spread_line"]).astype(int).to_numpy()
    table = reliability_table(out["cover_prob"].to_numpy(), covered, bins=4)
    assert set(table.columns) == {"bin", "n", "mean_pred", "observed"}
    assert len(table) <= 4


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        Calibrator().predict(_preds(n=3))
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_calibrate.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.model.calibrate'`.

- [ ] **Step 3: Write `src/nfl_game/model/calibrate.py`**

```python
"""Turn model-vs-market point gaps into probabilities.

A 4-point disagreement means nothing on its own. Fitting the gap against historical
cover outcomes gives it an empirical hit rate.

Fit this on walk-forward predictions only. In-sample gaps are systematically
overconfident, and a calibrator trained on them produces probabilities that look sharp
and are wrong.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better; 0.25 is a coin flip."""
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def reliability_table(probs: np.ndarray, outcomes: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Predicted vs observed frequency per probability bucket. Well-calibrated means they match."""
    df = pd.DataFrame({"p": np.asarray(probs), "y": np.asarray(outcomes)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")
    )
    return grouped.reset_index()


class Calibrator:
    """Maps (model - market) disagreement to cover and over probabilities."""

    def __init__(self):
        self._cover = None
        self._over = None

    def fit(self, preds: pd.DataFrame) -> "Calibrator":
        d = preds[preds["margin"].notna() & preds["total_points"].notna()]

        spread_edge = (d["model_margin"] - d["spread_line"]).to_numpy(dtype=float).reshape(-1, 1)
        covered = (d["margin"] > d["spread_line"]).astype(int).to_numpy()
        self._cover = LogisticRegression().fit(spread_edge, covered)

        total_edge = (d["model_total"] - d["total_line"]).to_numpy(dtype=float).reshape(-1, 1)
        went_over = (d["total_points"] > d["total_line"]).astype(int).to_numpy()
        self._over = LogisticRegression().fit(total_edge, went_over)
        return self

    def predict(self, preds: pd.DataFrame) -> pd.DataFrame:
        if self._cover is None or self._over is None:
            raise RuntimeError("call fit() before predict()")
        spread_edge = (
            (preds["model_margin"] - preds["spread_line"]).to_numpy(dtype=float).reshape(-1, 1)
        )
        total_edge = (
            (preds["model_total"] - preds["total_line"]).to_numpy(dtype=float).reshape(-1, 1)
        )
        return pd.DataFrame(
            {
                "game_id": preds["game_id"].to_numpy(),
                "cover_prob": self._cover.predict_proba(spread_edge)[:, 1],
                "over_prob": self._over.predict_proba(total_edge)[:, 1],
            }
        )
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_calibrate.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_game/model/calibrate.py tests/test_calibrate.py
git commit -m "feat: calibrate model-market gaps into cover and over probabilities"
```

---

### Task 11: Market comparison and weekly slate CLI

**Files:**
- Create: `src/nfl_game/market/compare.py`, `scripts/slate.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `GameModel`, `Calibrator`.
- Produces:
  - `build_slate(features_df, preds, probs, edge_threshold=2.0) -> pd.DataFrame` with columns
    `game_id, season, week, away_team, home_team, model_spread, market_spread, spread_gap, cover_prob, model_total, market_total, total_gap, over_prob, edge_flag`
  - `slate_markdown(slate: pd.DataFrame) -> str`

**Context for the implementer:** `model_spread` and `market_spread` are both stated as **home-team margins**, matching nflverse's `spread_line` convention — positive means the home team is favored. Keeping one convention end-to-end avoids the sign errors that make betting tools quietly useless.

`edge_flag` is 1 when `abs(spread_gap) >= edge_threshold`. It is a flag, not advice — v1 deliberately ships no bet sizing.

The output schema is fixed here so a future web app can render it without rework.

- [ ] **Step 1: Write the failing test**

`tests/test_compare.py`:

```python
import pandas as pd

from nfl_game.market.compare import SLATE_COLS, build_slate, slate_markdown


def _inputs():
    feats = pd.DataFrame(
        {
            "game_id": ["2026_01_KC_BUF", "2026_01_MIA_NYJ"],
            "season": [2026, 2026], "week": [1, 1],
            "home_team": ["BUF", "NYJ"], "away_team": ["KC", "MIA"],
            "spread_line": [2.5, -1.0], "total_line": [48.5, 43.0],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["2026_01_KC_BUF", "2026_01_MIA_NYJ"],
            "model_margin": [6.0, -1.5], "model_total": [51.0, 43.2],
        }
    )
    probs = pd.DataFrame(
        {
            "game_id": ["2026_01_KC_BUF", "2026_01_MIA_NYJ"],
            "cover_prob": [0.58, 0.49], "over_prob": [0.55, 0.51],
        }
    )
    return feats, preds, probs


def test_slate_has_fixed_schema():
    out = build_slate(*_inputs())
    assert list(out.columns) == SLATE_COLS


def test_gap_is_model_minus_market():
    out = build_slate(*_inputs()).set_index("game_id")
    assert out.loc["2026_01_KC_BUF", "spread_gap"] == 3.5   # 6.0 - 2.5
    assert out.loc["2026_01_KC_BUF", "total_gap"] == 2.5    # 51.0 - 48.5


def test_edge_flag_respects_threshold():
    out = build_slate(*_inputs(), edge_threshold=2.0).set_index("game_id")
    assert out.loc["2026_01_KC_BUF", "edge_flag"] == 1   # gap 3.5
    assert out.loc["2026_01_MIA_NYJ", "edge_flag"] == 0  # gap 0.5


def test_higher_threshold_flags_fewer_games():
    feats, preds, probs = _inputs()
    assert build_slate(feats, preds, probs, edge_threshold=10.0)["edge_flag"].sum() == 0


def test_sorted_by_absolute_edge():
    out = build_slate(*_inputs())
    assert out.iloc[0]["game_id"] == "2026_01_KC_BUF"


def test_markdown_renders_every_game():
    md = slate_markdown(build_slate(*_inputs()))
    assert "KC" in md and "BUF" in md and "NYJ" in md
    assert md.startswith("|")
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_compare.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.market.compare'`.

- [ ] **Step 3: Write `src/nfl_game/market/compare.py`**

```python
"""Model vs market: the weekly slate.

Both model_spread and market_spread are stated as home-team margins, matching nflverse's
spread_line convention (positive = home favored). One convention end to end is what keeps
sign errors from quietly inverting every pick.

edge_flag marks disagreement above a threshold. It is a flag, not advice — v1 ships no
bet sizing, because staking is only as sound as the calibration underneath it.
"""

import pandas as pd

SLATE_COLS = [
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "model_spread",
    "market_spread",
    "spread_gap",
    "cover_prob",
    "model_total",
    "market_total",
    "total_gap",
    "over_prob",
    "edge_flag",
]


def build_slate(
    features_df: pd.DataFrame,
    preds: pd.DataFrame,
    probs: pd.DataFrame,
    edge_threshold: float = 2.0,
) -> pd.DataFrame:
    """Join predictions and probabilities onto the slate, flag disagreements."""
    df = features_df.merge(preds, on="game_id", how="inner").merge(probs, on="game_id", how="left")

    df["model_spread"] = df["model_margin"].round(2)
    df["market_spread"] = df["spread_line"]
    df["spread_gap"] = (df["model_margin"] - df["spread_line"]).round(2)
    df["model_total"] = df["model_total"].round(2)
    df["market_total"] = df["total_line"]
    df["total_gap"] = (df["model_total"] - df["total_line"]).round(2)
    df["cover_prob"] = df["cover_prob"].round(4)
    df["over_prob"] = df["over_prob"].round(4)
    df["edge_flag"] = (df["spread_gap"].abs() >= edge_threshold).astype(int)

    out = df[SLATE_COLS].copy()
    return out.reindex(out["spread_gap"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def slate_markdown(slate: pd.DataFrame) -> str:
    """Render the slate as a markdown table, edges first."""
    header = (
        "| Game | Model | Market | Gap | Cover% | Model O/U | Market O/U | Gap | Over% | Edge |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in slate.itertuples(index=False):
        rows.append(
            f"| {r.away_team} @ {r.home_team} | {r.model_spread:+.1f} | {r.market_spread:+.1f} "
            f"| {r.spread_gap:+.1f} | {r.cover_prob:.1%} | {r.model_total:.1f} "
            f"| {r.market_total:.1f} | {r.total_gap:+.1f} | {r.over_prob:.1%} "
            f"| {'*' if r.edge_flag else ''} |"
        )
    return header + "\n".join(rows)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_compare.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Write `scripts/slate.py`**

```python
"""Weekly slate report: model vs market for a given season/week."""

import argparse

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.market.compare import build_slate, slate_markdown
from nfl_game.model.calibrate import Calibrator
from nfl_game.model.predict import GameModel
from nfl_game.paths import PROCESSED_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "gbm"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--edge-threshold", type=float, default=2.0)
    args = ap.parse_args()

    feats = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")

    # Calibrate on out-of-sample predictions from every completed prior season.
    prior_seasons = sorted(s for s in feats["season"].unique() if s < args.season)
    oos = walk_forward(feats, prior_seasons[1:], estimator=args.estimator, alpha=args.alpha)
    calibrator = Calibrator().fit(oos)

    train = feats[feats["season"] < args.season]
    target = feats[(feats["season"] == args.season) & (feats["week"] == args.week)]
    if target.empty:
        raise SystemExit(f"no games found for {args.season} week {args.week}")

    model = GameModel(estimator=args.estimator, alpha=args.alpha).fit(train)
    preds = model.predict(target)
    probs = calibrator.predict(target.merge(preds, on="game_id"))

    slate = build_slate(target, preds, probs, edge_threshold=args.edge_threshold)

    csv_path = PROCESSED_DIR / f"slate_{args.season}_wk{args.week:02d}.csv"
    md_path = PROCESSED_DIR / f"slate_{args.season}_wk{args.week:02d}.md"
    slate.to_csv(csv_path, index=False)
    md_path.write_text(slate_markdown(slate), encoding="utf-8")

    print(slate_markdown(slate))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full suite and a real slate**

```
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\slate.py --season 2025 --week 1
```

Expected: all tests pass, ruff clean, and a printed markdown table of Week 1 2025 games with model and market numbers side by side. Cover probabilities should cluster near 50% — most games will not show an edge, and that is the correct behavior against an efficient market.

- [ ] **Step 7: Write `CLAUDE.md`**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

All four layers are implemented: `data/`, `ratings/`, `model/`, `market/`. Check contents
before assuming any extension exists.

## Commands

    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
    .\.venv\Scripts\python.exe -m pytest
    .\.venv\Scripts\python.exe -m ruff check .

## Data sourcing

All data comes from `nflreadpy`. No API key, no scraping. `load_schedules()` carries the
closing `spread_line` and `total_line` — complete back to 2000, moneylines from 2010.

Next Gen Stats is **2016+ and offense only** — there is no defensive NGS table, which is
why defensive strength comes from EPA. NGS also applies qualifier thresholds: passing
covers ~99% of team-games, rushing only ~86%. Missing values are imputed with the
league-week mean and flagged via `<metric>_imputed`.

## Architecture

Data flows one direction: `data` → `ratings` → `model` → `market`. Do not introduce
reverse dependencies.

- `ratings/epa.py` — the core. `fit_ratings` regresses play EPA on offense/defense team
  dummies, which is what separates team quality from schedule quality. **Both `off_rating`
  and `def_rating` are oriented so higher is better** — the raw defensive coefficient is
  negated. Every consumer depends on that.
- `ratings/build.py` — as-of ratings. Every function takes an `(asof_season, asof_week)`
  cutoff and uses strictly prior data. This is the project's central correctness property
  and is tested directly.
- `model/calibrate.py` — must be fit on walk-forward predictions, never in-sample.

## Reading the backtest

The market is the benchmark. Market margin MAE is around 9.8–10.3 points; matching it is a
good result. A model MAE far below the market's, or an ATS hit rate above ~0.56, is
overwhelmingly likely to be a data leak rather than an edge — audit the as-of joins first.
`market_comparison_regression` is the decisive test: if `model_coef` is near zero, the
model adds nothing the closing line doesn't already contain.
```

- [ ] **Step 8: Commit**

```bash
git add src/nfl_game/market/compare.py scripts/slate.py tests/test_compare.py CLAUDE.md
git commit -m "feat: weekly slate report comparing model to market"
```

---

## Self-Review

**Spec coverage:** Every spec section maps to a task — data sourcing (2), EPA backbone and opponent adjustment (3, 4), recency weighting and preseason prior (5), NGS layer with imputation flags (6), features including rest/roof/weather/div (7), margin and total models (8), walk-forward validation with the market baseline, ATS-by-threshold, and the coefficient test (9), calibration with Brier and reliability (10), slate output schema and CLI (11).

**Known deviations from the spec, both deliberate:**

1. The spec lists a QB-change flag among the features. It is **not** in Task 7's `FEATURE_COLS`. Building it correctly needs a per-team starter history derived from prior weeks, which is a task's worth of work on its own and is not required for a working end-to-end model. Recommend adding it as Task 12 after the baseline backtest establishes whether the model needs the help.
2. The spec describes preseason regression to the mean as its own step. Task 5 achieves it through two existing mechanisms instead — `season_penalty` decay plus ridge shrinkage — rather than adding a separate blending parameter. Same behavior, one less knob.

**Placeholder scan:** No TBDs. Every code step contains complete runnable code.

**Type consistency:** `team` is the column name throughout `ratings/`; `home_team`/`away_team` in `schedules` and features. `model_margin`/`model_total` are produced by `GameModel.predict` and consumed unchanged by `Calibrator` and `build_slate`. `spread_line`/`total_line` keep nflverse naming end to end. `NGS_METRICS` is defined in Task 6 and imported by Task 7.

---

## Verification

End-to-end, after all tasks:

```
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\build_dataset.py --start-season 2016 --end-season 2025
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe scripts\slate.py --season 2025 --week 1
```

The backtest output is the real acceptance test. Read it as described in Task 9 Step 7 and in CLAUDE.md: matching the market is success, and beating it decisively is a leak until proven otherwise.
