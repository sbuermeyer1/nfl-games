# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

All six layers are implemented: `data/`, `ratings/`, `model/`, `market/`, `tracking/`,
`web/`. Check contents before assuming any extension exists.

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
    .\.venv\Scripts\python.exe scripts\build_tracker.py
    .\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
    .\.venv\Scripts\python.exe scripts\update_live_tracker.py --dry-run
    .\.venv\Scripts\python.exe scripts\slate.py --season 2026 --week 1

- `build_dataset.py` loads pbp/schedules/NGS, builds as-of ratings and features, and
  writes `data/processed/game_features.parquet`. Run this first, and rerun it whenever
  `--start-season`/`--end-season` changes or upstream data is refreshed; everything else
  reads the cached parquet rather than hitting `nflreadpy` itself.
- `backtest.py` runs `walk_forward` over `--test-seasons` (a `lo-hi` range) and prints
  margin/total MAE against the market, ATS/O-U hit rates, ATS by edge threshold, and
  `market_comparison_regression`. This is the acceptance test -- see "Reading the
  backtest" below.
- `build_tracker.py` rebuilds the offline historical Ridge `ridge-v1` tracker ledger from
  the cached feature artifact and accepts only the regression baseline below. It is not a
  routine live updater.
- `refresh_2026.py` downloads the current schedule, preserves the frozen through-2025
  feature corpus exactly, and atomically writes the complete 2026 schedule plus current/
  next prediction-week features. It defaults to dry-run; `--dry-run` and `--write` are
  mutually exclusive.
- `update_live_tracker.py` advances immutable 2026 publication/finalization facts while
  rechecking the exact 1,359-row historical baseline. It also defaults to dry-run and is
  the only artifact writer that may add official live records. `--write` remains approval
  gated through `ENABLE_OFFICIAL_TRACKER`; `--void-game GAME_ID=REASON` is the audited
  manual exception path.
- `slate.py --season Y --week W` fits `GameModel` on every season before `Y`, calibrates
  on `walk_forward` predictions from every season before `Y` (letting `walk_forward` skip
  whatever it must -- see the degenerate-feature note below), and writes/prints a
  markdown table comparing model and market for that week's games. Both `--estimator`
  (`ridge`/`gbm`) and `--alpha` are shared with `backtest.py`; `--edge-threshold` controls
  when `edge_flag` fires (default 2.0 points).

### Ridge v2 (research track, not the shipped path)

    .\.venv\Scripts\python.exe scripts\build_v2_dataset.py --dry-run
    .\.venv\Scripts\python.exe scripts\build_v2_dataset.py --write

`build_v2_dataset.py` builds the Ridge-v2 union feature artifact over the 2015-2025
historical seasons with 2021-2025 as the evaluation window, and writes
`data/processed/game_features_ridge_v2.parquet` plus `data/processed/ridge_v2_manifest.json`.
It defaults to dry-run, prints every source row count, coverage figure and digest, refuses a
Ridge-v1 destination, and replaces both files atomically or restores both. `ridge-v1` remains
the shipped model: nothing in `web/` reads either v2 file and neither is copied into the
Docker image. Runtime cost and the fresh-clone caveat are in `README.md`.

**The manifest's four digests are the reproducibility contract, and all four exclude the
build clock.** `schema_sha256` and `features_semantic_sha256` identify the output frame;
`source_manifest_sha256` and `manifest_semantic_sha256` identify the inputs and the manifest
itself. `build_timestamp` and each snapshot's `retrieved_at` remain in the manifest as
provenance but are never hashed, so two builds from identical inputs produce identical
digests and a digest that moves means data moved. `latest_event_at` is deliberately still
hashed -- `retrieved_at` records when we fetched, `latest_event_at` is a property of the
data itself. `_validate_artifacts` recomputes the digests from the stored manifest, which
makes it an internal integrity check against a tampered file, not a cross-run signal.

### The Ridge-v2 experiment result: the challenger TIES Ridge v1 exactly

Run 2026-08-25 on the locked artifact. **Gates 3, 4, 5, 6 and 9 FAIL, so Ridge v1 remains
official** and nothing about the shipped model, tracker or website changes.

**The headline is in the selections, not the gate table.** The nested selection chose **C0 --
the exact Ridge-v1 schema -- for both targets in all five evaluation seasons**, so the
challenger reproduces the champion: all **1,359 outer predictions are bit-identical** to
Ridge v1 (max absolute difference 0.0), and margin MAE agrees to full float precision at
10.273977625706554. Per-season improvement is exactly 0.0000 in every season. None of the
Ridge-v2 blocks -- ratings, quarterback, style, personnel continuity, PFR -- earned selection
in a single evaluation season. Only the 2019 and 2020 calibration seeds picked non-C0
configurations (C4/C1), on two or three seasons of training data, with an inner margin MAE
of 25.0 and 20.3.

> **CORRECTION (2026-08-26): the selection was not a fair test, so "no v2 block helps" is NOT
> supported.** The sentences above describe what the selection *did*, and the tie itself is
> real. What does not follow is the merit verdict — the blocks were never given an equal
> comparison, and the earlier "overfitting, not signal" reading of 2019/2020 is part of what
> is now in doubt.
>
> **Mechanism.** In `_inner_evaluations` (`src/nfl_game/experiments/v2_selection.py:337`) a
> `DegenerateFeatureError` re-raises for every candidate *except* C0, which is allowed to
> `continue` and silently drop that validation season. `mean_inner_mae`
> (`v2_selection.py:118`) is then a plain `np.mean` over whatever seasons each config happens
> to have. C0 is therefore scored on an easier season set than its challengers.
>
> **Evidence in the committed artifact.** In `data/processed/ridge_v2_evaluation.json` every
> C0 selection (2021-2025) records `validation_seasons` beginning at **2019**, while the
> non-C0 selections (2019, 2020) begin at **2017**. The 2017 and 2018 slices run MAE 17-25,
> so C1-C5 carried two catastrophic folds that C0 never faced.
>
> **Rescored on C0's own season set, a richer candidate wins every fold checked**, by more
> than the 0.05 tie-break tolerance:
>
> | fold | as run | on matched seasons |
> | --- | --- | --- |
> | 2021 margin | C0 10.5899 | **C2 10.4592** (C1, C3 also beat C0) |
> | 2021 total | C0 10.9971 | **C2 10.8399** |
> | 2025 margin | C0 10.3717 | **C3 10.2557** (C1, C2, C4 also beat) |
> | 2025 total | C0 10.8276 | **C2 10.7762** |
>
> The 1,359 bit-identical predictions remain correct; only the conclusion drawn from them is
> withdrawn. Whether the v2 features help is **unknown**, and settling it needs the selection
> fixed to score every candidate on a common fold set, then a re-run (~22 min). Ridge v1
> remains official either way — this correction promotes nothing.

**Do not read gates 1 and 2 as evidence of improvement.** They test `< 10.274` and `< 10.684`
-- the *rounded* literals from the recorded baseline -- so a challenger that is bit-identical
to the champion passes them by 2.2e-05 and 1.8e-04. Those two gates cannot distinguish a tie
from a win. Gate 9 (Brier) fails by 0.0001 *despite* identical point predictions, because the
v2 calibrator is seeded on 2019-2020 where the selection did pick C4/C1. Gate 6 fails on the
margin side only (coefficient -0.0218, the Ridge-v1 value); the total side passes at +0.2924.

**The C1 rating block cannot be ablated**, measured on live data: removing it raises
`rating variant maps canonical column(s) outside the target schema` for margin and
`total_points/C1 has no rating-variant canonical columns` for the total. The rating-variant
contract requires a non-C0 schema to declare its canonical rating columns, and relaxing that
would change the manifest the experiment exists to measure. Those rows are recorded as
`not_constructible` with the exact error rather than dropped. Ablations therefore exist only
for 2019-2020; the C0 seasons have no blocks to remove.

### FTN charting (E1) was measured separately and REJECTED

Run 2026-08-25. FTN begins in 2022, so only 2023-2025 have a prior charting season to train
on. Two arms trained on identical rows -- C0 alone versus C0 plus the FTN block:

| season | margin (core -> E1) | total (core -> E1) |
| --- | --- | --- |
| 2023 | 10.8592 -> 11.0589 | 10.7244 -> 10.5001 |
| 2024 | 10.3020 -> 10.4323 | 10.4293 -> 10.8638 |
| 2025 | 10.3356 -> 10.4002 | 10.5337 -> 10.5037 |

Pooled, FTN costs **-0.1315** MAE on margin (worse in 3 of 3 seasons) and **-0.0601** on total
(better in 2 of 3, but 2024 loses 0.43). Nothing here justifies adding charting to the model.

**The FTN result stands on its own; do not lean it on the core result.** An earlier version of
this section called it "consistent with the core result that no v2 block earned selection" —
that support is withdrawn per the correction above. E1 does not need it: these two arms were
trained on *identical rows* and differ only by the FTN block, so it is a genuine paired
comparison, unaffected by the unequal-fold defect in the nested selection.

**Two live-schema facts about this feed, both measured before the code was written.** The FTN
table has **no team column** -- it carries `nflverse_game_id`/`nflverse_play_id` only, so
offence comes from a play-by-play join on `posteam`; 100% of charted plays join, across
2022-2025. And **`date_pulled` is not an availability time**: it is the archive's snapshot
stamp, with 2022 rows carrying 2024 dates, so the as-of rule is week ordering like every other
block. Live coverage is 2,278 charted team-games, 32 teams per season, 73-81 charted plays per
team-game.

## Data sourcing

All data comes from `nflreadpy`. No API key, no scraping. `load_schedules()` carries the
closing `spread_line` and `total_line` — complete back to 2000, moneylines from 2010.

Next Gen Stats is **2016+ and offense only** — there is no defensive NGS table, which is
why defensive strength comes from EPA. NGS also applies qualifier thresholds: passing
covers ~99% of team-games, rushing only ~86%. Missing values are imputed with the
league-week mean and flagged via `<metric>_imputed`.

## Packaged artifacts

Exactly three reviewed Parquet files ship in the repository and Docker image:

- `data/processed/game_features.parquet` -- frozen historical model rows through 2025 plus
  only the currently active 2026 prediction weeks;
- `data/processed/schedule_2026.parquet` -- all 272 normalized 2026 regular-season games,
  with spread and total independently nullable, and the runtime's offline market fallback;
- `data/processed/tracker_ledger.parquet` -- exactly 1,359 accepted historical rows plus
  separately typed, immutable official live rows after Stage 2 begins.

Runtime startup fails closed if any file is missing, malformed, or if the schedule has no
2026 regular-season rows. `data/processed/ridge_v2_manifest.json` is also
committed, but it is a provenance record rather than a packaged artifact -- it is not read at
runtime and not copied into the image. The Ridge-v2 feature parquet is gitignored: at 7.3 MB
against v1's 251 KB it would be re-added whole on every rebuild. Artifact builders and workflow jobs may replace files atomically;
the web package is read-only and must never write them.

## Architecture

Data flows one direction and must remain:

```text
data -> ratings -> model -> market -> tracking -> web
```

Do not introduce reverse dependencies. Package ownership is strict:

- `data/` owns nflverse I/O, schemas, and team-code normalization;
- `ratings/` owns strictly-prior as-of team strength and cannot consume market lines;
- `model/` owns features, Ridge/GBM prediction, and calibration; `FEATURE_COLS` is the
  complete model input boundary and contains no spread, total, or market-derived value;
- `market/` overlays independently nullable lines after prediction and owns the bounded
  five-minute live cache/stale-snapshot behavior;
- `tracking/` owns historical summaries and immutable live publication/closing/grading
  transitions, but no web route may invoke an artifact writer;
- `web/` validates and reads packaged inputs, presents live/fallback metadata, and remains
  read-only.

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
- `tracking/` — tracker artifacts are built offline and read only at runtime. Only Ridge
  `ridge-v1` is official. Historical and live records must never aggregate together;
  thresholds are fixed at 2 points for qualified picks and cumulative 5/10/15-point
  spread cohorts. A model-version change must not rewrite published live history.

### Immutable and market-blind facts

- A target game's ratings use strictly earlier games; same-week and future results are
  forbidden inputs.
- Model training and prediction consume only `FEATURE_COLS`. A live schedule/line refresh
  may change presentation and edge calculations, never `model_margin` or `model_total`.
- For an official live record, model version, estimator, predictions, `published_at`, the
  original `kickoff_at`, every published/official line, and its observed time are frozen.
  A postponement may update only `current_kickoff_at` before finalization.
- Picks lock **4 days before kickoff**, floored so that no game is published until the
  features artifact has been rebuilt from a complete prior week. In practice the Sunday
  and Monday slate gets the full 4 days; Thursday games and Thanksgiving are held by the
  floor at roughly 2.3–2.6 days.
- Spread and total publication states progress independently. Missing, excluded, push,
  closing-line, CLV, and void facts are explicit; none may be inferred by rewriting an
  earlier field.
- Historical and live rows are always selected and summarized separately. A routine live
  update must preserve the 1,359 historical rows and the exact acceptance metrics below.
- Default CLI invocations and all web requests are non-mutating. Only explicit atomic
  builder/updater `--write` modes may replace packaged bytes.

## Reading the backtest

The market is the benchmark. `scripts/backtest.py --test-seasons 2021-2025` measures
market margin MAE at 9.752 points; matching it is a good result. A model MAE meaningfully
below the market's, or an ATS hit rate above ~0.56, is overwhelmingly likely to be a data
leak rather than an edge — audit the as-of joins first. `market_comparison_regression` is
the decisive test: if `model_coef` is near zero, the model adds nothing the closing line
doesn't already contain.

**The current result is that the model ties the CLOSING line without beating it, and that is
the success case, not a failure.** Do not "improve" it into an edge against the close.

**It also beats the EARLY line, but by far less than this file used to claim.** At the lead the
tracker actually publishes at -- a true 5 days before each game's own kickoff -- closing-line
value on spreads is **+0.1444 at edge >= 2 (z = 2.41, n = 703)**, out of 1,345 priced games in
the corpus, measured by
`scripts/analyze_line_value.py` against `line_history_combined_g05.parquet`. Totals are the
stronger side. The model anticipates a little of the movement between the early number and the
close, and none of what remains at the close.

> **CORRECTION (2026-08-31): the figures previously here (+0.254 at edge >= 1, +0.267 at
> edge >= 2, "about 55% of the way" to break-even) were measured at the wrong lead and are
> withdrawn.**
>
> **Mechanism.** `snapshot_timestamps` (`src/nfl_game/data/line_history.py`) takes one snapshot
> per (season, week), `days_before` ahead of **that week's first kickoff**. That is deliberate
> and correct for its own purpose -- it guarantees no game in the week has been played yet --
> but `PUBLISH_BEFORE` is measured from **each game's own kickoff**. Since the week's first
> game is usually Thursday night, a Sunday game's "5-day" snapshot sat ~7.7 days out.
>
> **Measured over 1,359 games:** the `_d05` cache has a mean lead of **7.51 days** (median 7.70,
> max 10.30). Only **98 of 1,359 (7%)** are within 0.1 days of a true 5-day lead; 1,139 are at
> 7 days or more. The file name asserted a property nobody had measured.
>
> **Re-measured at true per-game leads** (`--anchor game`, cache files tagged `_g<lead>`), on a
> fixed common set of 778 games with the model and games held constant so only the line changes:
>
> | true lead | CLV edge >= 2 | z |
> | --- | --- | --- |
> | 5d | +0.1280 | 1.86 |
> | 7d | +0.4123 | 4.32 |
> | 9d | +0.5103 | 5.02 |
> | 11d | +0.5445 | 5.18 |
>
> **Publishing earlier does NOT recover this, and the table above is itself contaminated.** It
> gave every arm current features -- week N-1 results that do not exist 9 days before a week-N
> game. Re-run with ratings lagged one week (`build_ratings(asof_week=W-1)`), which is what a
> 9-day lock would actually have, on a fixed 465-game set at edge >= 2: **5d +0.1409 (z 2.10)
> vs 9d +0.0032 (z 0.03)**. The entire apparent gain was look-ahead. One week of staleness
> costs only +0.0044 margin MAE, so it barely moves average accuracy -- it corrupts the tail,
> which is exactly where you bet.
>
> **Conclusion: the 5-day lock is roughly the right choice, but at ~30% of break-even, not
> 98%, and no publishable lead does better.** Do not reopen the lock length without new
> evidence, and do not quote any figure from a `_d<lead>` cache as a lead-specific result.

Do not evaluate this model by its ATS record. At ~141 qualified picks a season, demonstrating
a true 54% needs roughly **27 seasons**; the closing-line ATS figures (0.4977 at 0+, 0.4752 at
2+) are noise plus vig. CLV is continuous and answers the same question on far less data --
though at the corrected lead it resolves at **z = 2.41 at edge >= 2** (3.39 across all
predictions), not the z > 4 this file claimed before the anchoring fix.
**1 spread point = 4.93% win probability** and break-even at -110 needs ~0.48 points,
so the corrected +0.1444 is about **30%** of the way there — real, but well short of an edge,
and not a gap the publication lead can close.

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

**The tracker ledger reports a DIFFERENT ATS number, and that is not a regression.** Since
2026-08-31 `data/processed/tracker_ledger.parquet` grades the historical corpus at the true
5-day line, so it reads **0.4947 (n=1316)** with 14 games excluded for having no line at
publication. The 0.4977 above is `scripts/backtest.py`, which still grades at the close and is
unchanged. Both are pinned: `EXPECTED_BASELINE` holds the closing record, `EXPECTED_EARLY_BASELINE`
the early one, and `assert_historical_baseline` picks whichever a given corpus carries by
inspecting it. `scripts/build_tracker.py --no-early-lines` reproduces the old closing-line ledger.

### Line shopping is worth more than the model

Measured by `scripts/analyze_key_numbers.py` on the same 1,359-game corpus. **Getting half a
point better than the number you bet changes 4.56% of outcomes and is worth +4.34% EV per bet
at -110** — against the model's own **~0.71%** (its corrected +0.1444 points of early-line
value at 4.93% per point). Half a point of shopping is roughly **6x the entire model edge**,
costs nothing, and requires no modelling. The anchoring correction of 2026-08-31 roughly
doubled this ratio: it was recorded as 3.3x when the model's edge was thought to be +0.267.

The value is concentrated on one number, because NFL margins are not smooth:

| number bet | bets | outcome changed | EV gain |
| ---: | ---: | ---: | ---: |
| **3.0** | 197 | **10.15%** | **+9.23%** |
| 2.5 | 157 | 5.10% | +5.10% |
| 3.5 | 141 | 4.26% | +4.26% |
| 6.5 | 69 | 7.25% | +7.25% |

**14.64% of games end on a 3-point margin** and 7.73% on 7. Buying off 3 is worth several
times what it is worth anywhere else.

Two cautions. **Book-to-book dispersion is NOT measured here** — nflverse carries a single
consensus line and a multi-book feed is paid, so this quantifies what a better number is worth
*if you can get it*, not how often books actually differ. And note the push arithmetic: **33 of
1,359 games land exactly on the closing spread (2.43% overall, 5.45% of the 605 integer-spread
lines)**; no half-point line can ever push.

### Tracker acceptance records

The reviewed `ridge-v1` historical ledger covers 1,359 games in 2021–2025 and must
produce these exact all-season records:

| selection | record | n | win rate |
| --- | --- | ---: | ---: |
| Qualified ATS 2+ | 336-371-16 | 707 | 0.475248 |
| Qualified O/U 2+ | 396-407-6 | 803 | 0.493151 |
| All ATS | 660-666-33 | 1326 | 0.497738 |
| All O/U | 677-671-11 | 1348 | 0.502226 |
| Spread 5+ | 102-102-4 | 204 | 0.500000 |
| Spread 10+ | 9-11-0 | 20 | 0.450000 |
| Spread 15+ | 1-1-0 | 2 | 0.500000 |

Pushes are excluded from each win-rate denominator. The 5+/10+/15+ cohorts are
cumulative, not disjoint buckets.

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
