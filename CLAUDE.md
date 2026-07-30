# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

All four layers are implemented: `data/`, `ratings/`, `model/`, `market/`. Check contents
before assuming any extension exists.

**This file is authoritative for how the project works.** `docs/development-log.md` is the
verbatim working ledger from the build, preserved for the reasoning behind each decision —
it is a historical record, append-only and chronological, and some of its claims were
falsified by the very reviews it describes. Treat it as evidence, never as authority, and
prefer this file wherever the two disagree. Per-task briefs and reports are archived in
`docs/sdd-archive/`. `docs/webapp-development-log.md` is the equivalent ledger for the web
dashboard built on top of the model, under the same evidence-not-authority rule; the
dashboard's operational documentation lives in the "Web dashboard operations" section of
`README.md`.

## Commands

    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
    .\.venv\Scripts\python.exe -m pytest
    .\.venv\Scripts\python.exe -m ruff check .

`scripts/` is the project's entire user interface -- run in this order:

    .\.venv\Scripts\python.exe scripts\build_dataset.py --start-season 2016 --end-season 2025
    .\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
    .\.venv\Scripts\python.exe scripts\slate.py --season 2025 --week 1

- `build_dataset.py` loads pbp/schedules/NGS, builds as-of ratings and features, and
  writes `data/processed/game_features.parquet`. Run this first, and rerun it whenever
  `--start-season`/`--end-season` changes or upstream data is refreshed; everything else
  reads the cached parquet rather than hitting `nflreadpy` itself.
- `backtest.py` runs `walk_forward` over `--test-seasons` (a `lo-hi` range) and prints
  margin/total MAE against the market, ATS/O-U hit rates, ATS by edge threshold, and
  `market_comparison_regression`. This is the acceptance test -- see "Reading the
  backtest" below.
- `slate.py --season Y --week W` fits `GameModel` on every season before `Y`, calibrates
  on `walk_forward` predictions from every season before `Y` (letting `walk_forward` skip
  whatever it must -- see the degenerate-feature note below), and writes/prints a
  markdown table comparing model and market for that week's games. Both `--estimator`
  (`ridge`/`gbm`) and `--alpha` are shared with `backtest.py`; `--edge-threshold` controls
  when `edge_flag` fires (default 2.0 points).

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
- `data/teams.py` — canonical team abbreviations, applied by every loader in
  `data/nfl.py` at ingestion. The three feeds disagree about relocated franchises:
  schedules keep the code in use that season (`OAK`, `SD`, `STL`), play-by-play uses the
  modern code for every season, and NGS spells the Rams `LAR` where schedules and pbp both
  say `LA`. Because every downstream join is a left merge ending in `features.py`'s blanket
  `fillna(0.0)`, a mismatch produced a silently zero-filled feature and never an error —
  which is how the `LA`/`LAR` case survived unnoticed across every Rams game in all ten
  seasons, 85 of them inside the backtest window. `normalize_team_codes` raises on an
  unrecognised code rather than passing it through, so a future feed change fails loudly
  instead of silently zeroing features. Normalise at ingestion; do not add per-join fixups.
- `ratings/build.py` — as-of ratings. Every function takes an `(asof_season, asof_week)`
  cutoff and uses strictly prior data. This is the project's central correctness property
  and is tested directly.
- `model/calibrate.py` — must be fit on walk-forward predictions, never in-sample.
  `Calibrator.fit` excludes exact pushes (`margin == spread_line` / `total_points ==
  total_line`) from training for both targets, matching `backtest.evaluate`'s own
  treatment: a push returns the stake, so it is not a "did not cover" outcome.
- `market/compare.py` — `build_slate` joins `GameModel` predictions and `Calibrator`
  probabilities onto a week's games and flags disagreements; `slate_markdown` renders it.
  `model_spread`/`market_spread` are both home-team margins (nflverse's `spread_line`
  convention, positive = home favored) end to end — keeping one sign convention is what
  keeps this from quietly inverting every pick. `edge_flag` is 1 when
  `abs(spread_gap) >= edge_threshold`; it is a flag, not advice.

## Reading the backtest

The market is the benchmark. `scripts/backtest.py --test-seasons 2021-2025` measures
market margin MAE at 9.752 points; matching it is a good result. A model MAE meaningfully
below the market's, or an ATS hit rate above ~0.56, is overwhelmingly likely to be a data
leak rather than an edge — audit the as-of joins first. `market_comparison_regression` is
the decisive test: if `model_coef` is near zero, the model adds nothing the closing line
doesn't already contain.

**The current result is that the model ties the market without beating it, and that is
the success case, not a failure.** Do not "improve" it into an edge.

### Regression baseline

`scripts/backtest.py --test-seasons 2021-2025` (ridge, the default) must produce:

| metric | model | market |
| --- | --- | --- |
| games | 1359 | — |
| margin MAE | 10.274 | 9.752 |
| total MAE | 10.684 | 10.309 |
| ATS hit rate | 0.4977 (n=1326) | — |
| O/U hit rate | 0.5022 (n=1348) | — |
| model_coef | -0.0218 | market_coef 1.0755 |
| r² | 0.2083 | — |

Treat these as an invariant: any change that is not deliberately a behavior change must
leave them untouched. If they move, stop and find out why before going further — and read
the paragraph above before judging the direction, because a number that looks like an
improvement is the one that most needs auditing.

## Degenerate training features

Ridge is scale-sensitive, so `GameModel`'s ridge pipeline standardizes `FEATURE_COLS`
before fitting. Two different ways a training slice can defeat that safely are both
guarded against now:

- **`RobustStandardScaler`** (`model/predict.py`) floors any feature's fitted `scale_` to
  1.0 if it falls below `10 * eps` (~2.22e-15) — sklearn's own constant-feature fallback,
  which its own mean-relative check can't reach for a feature that is constant to
  floating-point noise but happens to have a mean near zero (e.g. `ryoe_diff` trained on
  2016 alone: std ~1e-17). Without this, `StandardScaler` divides by that near-zero
  number and any later row with a real value explodes into a `model_margin` on the order
  of 1e15.
- **The degenerate-feature guard** (`GameModel.fit`, via `_degenerate_features`) catches
  the milder case the scaler floor can't: a feature whose variance is nowhere near that
  epsilon floor but that still has too few distinct values to support a coefficient,
  because almost every row shares the same imputed default. `ryoe_diff` trained on
  2016+2017 (512 rows) has std ~1.05e-2 — comfortably above the scaler's floor — but only
  5 distinct values (416 rows at 0.0, ~94 at float noise around it, 2 rows with a real
  signal). The guard skips columns whose values are only 0/1 (legitimate indicator flags
  like `is_dome`/`div_game`/`ngs_imputed_any`, which always take just two values and are
  not degenerate) and raises `DegenerateFeatureError` if any other column has fewer than
  `MIN_DISTINCT_VALUES = 10` distinct values in the training slice — comfortably below
  `rest_diff`'s minimum in every fold that isn't already flagged by `ryoe_diff` (15, from
  the 2019 fold onward; the one exception, the 2017 fold, sees 13 but is already
  degenerate on `ryoe_diff` regardless), and comfortably above the two poisoned folds'
  3 and 5. `walk_forward` catches this and
  skips the fold, the same way it already skips a season with no prior training data, so
  a fold that can't be trained never contributes predictions to a caller (e.g. the
  calibration corpus `scripts/slate.py` builds).

Both guards only ever change behavior for a feature below their respective threshold;
neither alters a healthy multi-season fold, which is why the regression baseline above is
unchanged by either. A skipped fold emits a `RuntimeWarning` naming the season and the
offending feature — a fold vanishing silently would shrink the corpus with no signal.

## Failed joins vs. missing data

`build_game_features` joins ratings and NGS onto each game with left merges, so a key that
doesn't match yields nulls rather than an error, and the closing `fillna(0.0)` then makes
those nulls indistinguishable from real zeros. That is how a team-code mismatch cost this
project every Rams game's NGS features across all ten seasons before anyone noticed.

Two things guard it now, and they are deliberately different:

- **NGS features** are flagged, not fenced. A week-1 game genuinely has no prior NGS to
  average, so it is zero-filled and `ngs_imputed_any` is set to 1 — the model can see that
  the value is imputed. All 159 all-zero-NGS rows in the current dataset are week 1 and all
  159 carry the flag.
- **Rating features** carry no such flag, so a missed join there would be invisible.
  `_check_rating_joins` raises `MissingRatingJoinError` before the fill if a game has null
  rating features *in a week where other teams were rated*. That condition is what makes it
  safe to raise: a week with nobody rated is legitimate (week 1 has no strictly-prior games
  to rate from) and is ignored, while a week where the join worked for everyone else can
  only mean this game's key failed.

The upstream cause is that the three feeds spell relocated franchises differently;
`data/teams.py` normalizes them at ingestion and raises on any code it doesn't recognize.
See its module docstring for the three disagreements and what each one cost.
