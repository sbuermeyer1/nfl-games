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
