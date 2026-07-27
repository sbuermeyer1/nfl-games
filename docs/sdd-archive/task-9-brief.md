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

