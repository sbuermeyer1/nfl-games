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

## Known issue: ryoe_diff near-zero variance poisons early walk-forward calibration

`scripts/slate.py` fits `Calibrator` on `walk_forward(feats, prior_seasons[1:], ...)`,
which for a season like 2025 includes test season 2017 — trained on 2016 alone. In 2016,
`ryoe_diff` is constant to floating-point noise (std ~1e-17, NGS rushing data is too new
for real signal yet). `GameModel`'s ridge pipeline standardizes features on the training
set, so `StandardScaler` divides by that near-zero std, and any 2017 test row with a
nonzero `ryoe_diff` explodes into a `model_margin`/`model_total` on the order of 1e15
(confirmed: `2017_02_CHI_TB` and `2017_02_MIA_LAC`, both models). Those two rows dominate
the `LogisticRegression` loss enough to drag both fitted coefficients to ~0, so
`Calibrator.predict` returns `cover_prob`/`over_prob` of exactly 0.500000 for every game
in the 2025 week 1 slate — a real bug, not genuine 50/50 calibration. `scripts/backtest.py`
is unaffected because Task 9's `--test-seasons 2021-2025` always trains on 5+ prior
seasons, where `ryoe_diff` already has healthy variance (std > 0.7 by 2018). Fixing this
needs either a variance floor in the ridge `StandardScaler` step or an as-of ratings fix
in `ratings/`, both reserved for human review — flagging here rather than patching around
it in `market/`.
