# Task 11: Market comparison and weekly slate CLI — report

## What I implemented

- `src/nfl_game/market/compare.py` — `SLATE_COLS`, `build_slate(features_df, preds, probs,
  edge_threshold=2.0)`, `slate_markdown(slate)`. Written from the brief's Step 3 code
  verbatim, with **one deliberate deviation** in `slate_markdown`'s NaN handling (see
  decision section below).
- `scripts/slate.py` — CLI per the brief's Step 5, verbatim. Fits `Calibrator` on
  `walk_forward` output from every completed prior season (never in-sample), exactly as
  specified. This construction was not touched.
- `tests/test_compare.py` — the brief's 6 tests verbatim, plus 2 of my own (see test
  discrimination section).
- `CLAUDE.md` — the brief's Step 7 content verbatim, plus one added section documenting a
  real bug I found while running Step 6 (see below).

## TDD Evidence

**RED** — `.\.venv\Scripts\python.exe -m pytest tests/test_compare.py -v`, before
`src/nfl_game/market/compare.py` existed:

```
collecting ... collected 0 items / 1 error
ERROR collecting tests/test_compare.py
tests\test_compare.py:3: in <module>
    from nfl_game.market.compare import SLATE_COLS, build_slate, slate_markdown
E   ModuleNotFoundError: No module named 'nfl_game.market.compare'
=========================== short test summary info ===========================
ERROR tests/test_compare.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.70s ===============================
```

Exactly the failure the brief predicted, for the expected reason: the module didn't exist.

**GREEN** — `.\.venv\Scripts\python.exe -m pytest tests/test_compare.py -v`, after writing
`src/nfl_game/market/compare.py`:

```
tests/test_compare.py::test_slate_has_fixed_schema PASSED                [ 12%]
tests/test_compare.py::test_gap_is_model_minus_market PASSED             [ 25%]
tests/test_compare.py::test_edge_flag_respects_threshold PASSED          [ 37%]
tests/test_compare.py::test_higher_threshold_flags_fewer_games PASSED    [ 50%]
tests/test_compare.py::test_sorted_by_absolute_edge PASSED               [ 62%]
tests/test_compare.py::test_sort_order_independent_of_input_row_order PASSED [ 75%]
tests/test_compare.py::test_markdown_renders_every_game PASSED           [ 87%]
tests/test_compare.py::test_markdown_handles_missing_cover_prob_without_rendering_nan PASSED [100%]
============================== 8 passed in 0.85s ==============================
```

(6 brief tests + 2 of my own added; all pass.) Also ran with
`-W error::FutureWarning -W error::DeprecationWarning`: still 8 passed, no warnings.

**Full suite:** `.\.venv\Scripts\python.exe -m pytest -q` → `87 passed in 5.86s` (the
pre-existing 79 plus 8 new; none of the original 79 were touched or weakened).

**Lint/format:**
```
.\.venv\Scripts\python.exe -m ruff check .
```
→ only the pre-existing, already-logged `test_epa.py` I001 nit from Task 3. No new
findings.
```
.\.venv\Scripts\python.exe -m ruff format --check src/nfl_game/market/compare.py scripts/slate.py tests/test_compare.py CLAUDE.md
```
→ `4 files already formatted` (after one `ruff format` pass on `test_compare.py` to
reformat the brief's multi-statement-per-line dict literals into one-key-per-line — a
whitespace-only change, no semantic difference from the brief's given code).

## Step 6 real-slate output, verbatim

```
.\.venv\Scripts\python.exe scripts\slate.py --season 2025 --week 1

| Game | Model | Market | Gap | Cover% | Model O/U | Market O/U | Gap | Over% | Edge |
|---|---|---|---|---|---|---|---|---|---|
| ARI @ NO | +1.6 | -6.0 | +7.5 | 50.0% | 48.3 | 44.5 | +3.8 | 50.0% | * |
| CIN @ CLE | -1.8 | -5.5 | +3.7 | 50.0% | 43.5 | 47.5 | -4.0 | 50.0% | * |
| DAL @ PHI | +5.0 | +8.5 | -3.5 | 50.0% | 46.6 | 47.5 | -0.9 | 50.0% | * |
| HOU @ LA | +6.2 | +3.5 | +2.8 | 50.0% | 47.2 | 43.5 | +3.7 | 50.0% | * |
| MIA @ IND | -1.2 | +1.5 | -2.7 | 50.0% | 48.0 | 47.5 | +0.5 | 50.0% | * |
| TB @ ATL | -4.0 | -1.5 | -2.5 | 50.0% | 48.1 | 47.5 | +0.6 | 50.0% | * |
| BAL @ BUF | +0.9 | -1.5 | +2.4 | 50.0% | 48.0 | 50.5 | -2.5 | 50.0% | * |
| PIT @ NYJ | -1.1 | -3.0 | +1.9 | 50.0% | 42.2 | 37.5 | +4.7 | 50.0% |  |
| KC @ LAC | -1.4 | -3.0 | +1.6 | 50.0% | 48.4 | 47.5 | +0.9 | 50.0% |  |
| CAR @ JAX | +3.0 | +4.5 | -1.4 | 50.0% | 48.1 | 45.5 | +2.6 | 50.0% |  |
| NYG @ WAS | +4.9 | +6.0 | -1.1 | 50.0% | 44.8 | 45.5 | -0.7 | 50.0% |  |
| DET @ GB | +2.6 | +1.5 | +1.1 | 50.0% | 49.0 | 48.5 | +0.5 | 50.0% |  |
| MIN @ CHI | -2.5 | -1.5 | -1.0 | 50.0% | 41.9 | 43.5 | -1.6 | 50.0% |  |
| TEN @ DEN | +8.0 | +8.5 | -0.5 | 50.0% | 45.1 | 42.5 | +2.6 | 50.0% |  |
| SF @ SEA | -2.6 | -2.5 | -0.1 | 50.0% | 46.9 | 43.5 | +3.4 | 50.0% |  |
| LV @ NE | +2.6 | +2.5 | +0.1 | 50.0% | 41.9 | 44.5 | -2.6 | 50.0% |  |

wrote C:\Users\sbuer\Documents\NFL Game Model\data\processed\slate_2025_wk01.csv
wrote C:\Users\sbuer\Documents\NFL Game Model\data\processed\slate_2025_wk01.md
```

`model_spread`/`model_total` themselves look sane (in the same range as the market's
numbers, no crazy values) — 7 of 16 games flag as edges at the default 2.0-point
threshold, and the spread gaps range from 0.1 to 7.5, which is a plausible spread of
model-vs-market disagreement for a real week.

### My reading against "what the honest output looks like"

**This is not the good, expected result — it's a bug, and I want to be explicit that I am
not presenting it as success.** Every single `cover_prob` and `over_prob` in the table
above is **exactly** 50.0%, with zero variation across 16 games whose spread gaps range
from 0.1 to 7.5 points. "Clustering near 50%" (the brief's expected honest outcome) means
probabilities like 48%, 53%, 46% scattered around a coin flip because the market is
efficient — it does not mean *literally identical* to six decimal places regardless of
how large the disagreement is. A calibrator that assigns the same probability to a
7.5-point gap and a 0.1-point gap has learned nothing; that is qualitatively different
from "learned a small, honest signal."

I traced the root cause rather than taking the number at face value (see the "Instructions
say ask if unclear... report BLOCKED with specifics" and "treat that as a signal something
is wrong" guidance — this is exactly that situation, just manifesting as suspiciously flat
0.5 instead of suspiciously confident numbers):

1. `scripts/slate.py`'s calibration step (which I preserved exactly, unmodified) calls
   `walk_forward(feats, prior_seasons[1:], estimator="ridge", alpha=1.0)` where
   `prior_seasons = [2016..2024]`, so `prior_seasons[1:] = [2017..2024]`. The 2017 test
   season trains on 2016 alone (`train = features_df[features_df["season"] < 2017]`).
2. In the 2016 season, `ryoe_diff` (a `FEATURE_COLS` feature) is constant to
   floating-point noise: `std ≈ 1.177e-17`, only 3 distinct values (`0`, `±2.78e-17`) —
   NGS rushing coverage is too sparse that early for real signal, so nearly every row
   gets the same imputed value.
3. `GameModel`'s ridge pipeline (`make_pipeline(StandardScaler(), Ridge(alpha=alpha))`)
   fits `StandardScaler` on the training set, so it divides by that ~1e-17 std. Any 2017
   test-season row with a real (non-imputed) `ryoe_diff` — e.g. `-0.168577` for
   `2017_02_CHI_TB` — gets standardized to a value of order 1e16, and the ridge
   coefficient on that column, however small, multiplies it into a `model_margin` of
   **-1.226773e+15** and a `model_total` of **1.410309e+15** for that single game (its
   mirror game `2017_02_MIA_LAC` gets the opposite sign, same magnitude). Verified
   directly:
   ```
   game_id           season week home_team away_team  model_margin  spread_line  margin
   2017_02_CHI_TB       2017    2        TB       CHI -1.226773e+15          7.0    22.0
   2017_02_MIA_LAC      2017    2       LAC       MIA  1.226773e+15          3.5    -2.0
   ```
4. Those two rows (and to a much smaller degree, 2018-season rows near the same fragile
   boundary, e.g. `2018_02_MIA_NYJ` at ~-210, `2018_02_NYG_DAL` at ~+183) dominate
   `LogisticRegression`'s loss surface enough that fitting drags both the cover and over
   coefficients to numerically zero:
   ```
   cover coef: [[-3.81193437e-13]] intercept [-1.57625866e-16]
   over  coef: [[6.10060458e-27]] intercept [-4.02052648e-25]
   ```
   A near-zero coefficient and near-zero intercept is exactly what produces
   `predict_proba` of 0.500000... for every input regardless of its edge — which is what
   the 2025-week-1 slate shows.
5. `scripts/backtest.py` (Task 9's own acceptance test, `--test-seasons 2021-2025`) is
   **not** affected: every one of those test seasons trains on 5+ prior seasons, and
   `ryoe_diff`'s std is already 0.77+ by 2018. This bug is specific to `slate.py`'s wider
   calibration window reaching back to `2017` (trained on `2016` alone) — a code path
   Task 9's backtest never exercised, since Task 9 never tested against 2017 or 2018.

This is a real, reproducible bug in the interaction between `ratings`/`features` (a
near-zero-variance early-season feature) and `model/predict.py`'s unguarded
`StandardScaler` (no variance floor), surfaced for the first time by Task 11's script
because it is the first caller to walk `walk_forward` back to a single-prior-season
training window. **I did not fix it**, because:
- `model/features.py` and everything under `ratings/` are explicitly off-limits ("reserved
  for a human decision at the final review").
- `model/predict.py` (the `StandardScaler`/`Ridge` pipeline) is described as "already
  built, tested, and reviewed clean" — outside Task 11's stated scope to modify, and a
  fix there (e.g. a variance floor) is a design decision, not a slate-layer concern.
- The brief explicitly told me to preserve the `walk_forward`-based calibration
  construction in `scripts/slate.py` exactly, and warned against "simplifying" it (e.g. by
  narrowing `prior_seasons` to avoid thin training windows) — which is the only fix I
  could make from inside `market/` without touching reserved files.

I documented this fully in `CLAUDE.md` under a new "Known issue" section (see file) rather
than silently shipping a slate whose flat probabilities could be mistaken for a genuine,
good calibration result. **Status implication: I'm flagging this as a real defect for the
final review, not folding it into "expected honest output."**

## `test_sorted_by_absolute_edge` finding

**It does not discriminate.** I verified by temporarily replacing the sort in
`build_slate` with a no-op (`return out.reset_index(drop=True)`) and rerunning
`tests/test_compare.py`: all 6 brief tests still passed, including
`test_sorted_by_absolute_edge`. The reason is exactly what was flagged in the task
brief: in `_inputs()`'s fixture, `2026_01_KC_BUF` (gap 3.5) is already first in every
input DataFrame, and `2026_01_MIA_NYJ` (gap 0.5) is second — so "no sort" and "sort
descending by absolute gap" produce the identical row order for this fixture. The test
cannot tell them apart.

I did **not** silently rewrite the brief-mandated test. Instead I added a new test,
`test_sort_order_independent_of_input_row_order`, which reuses the exact same numbers
but builds fresh input frames with the smaller-gap game (`MIA_NYJ`) listed *first* and
the larger-gap game (`KC_BUF`) listed *second*. I confirmed this new test:
- **passes** with the real sort in place (`assert list(out["game_id"]) ==
  ["2026_01_KC_BUF", "2026_01_MIA_NYJ"]`), and
- **fails** when I reintroduce the no-sort stub, with a clear diff showing
  `2026_01_MIA_NYJ` first instead of `2026_01_KC_BUF`.

That confirms the new test does discriminate correct sorting from "happens to already be
sorted." I left the brief's original `test_sorted_by_absolute_edge` untouched — flagging
this for the reviewer's decision on whether to keep, replace, or supersede it, per
instructions not to silently change brief-mandated tests.

## Decision: NaN probability rendering in `slate_markdown`

`Calibrator.predict` degrades per-row: a row missing `spread_line` (e.g. an
upcoming-week game whose line hasn't posted yet) returns `NaN` for `cover_prob` while
still returning a real `over_prob` (and vice versa for `total_line`/`over_prob`). The
brief's literal `slate_markdown` code formats every probability with `f"{r.cover_prob:.1%}"`
unconditionally, which on a NaN input in Python renders as the literal string `"nan%"`.

**Decision: this is not acceptable output, and I changed it.** `"nan%"` in a markdown
table reads exactly like a data-quality bug to anyone consuming the report (a bettor, a
future web app rendering this schema) — indistinguishable from "the pipeline broke,"
when the actual meaning is "no line was available for this side yet, so no probability
could be computed." I changed `slate_markdown` to render `"n/a"` for either probability
when it is NaN, leaving every other column (including `spread_gap`/`total_gap`, which can
still be real if only one side of the line is posted) unaffected. This is a small,
targeted change: `SLATE_COLS`, `build_slate`'s column math, and every other line of the
brief's `slate_markdown` are unchanged from the brief's given code — only the two
probability-formatting lines changed from unconditional `f"{x:.1%}"` to a NaN-guarded
version.

I added a new test, `test_markdown_handles_missing_cover_prob_without_rendering_nan`,
which injects a NaN `cover_prob` into the `_inputs()` fixture and asserts `"nan%"` does
not appear in the rendered markdown. It passes against my implementation; I did not
re-verify it fails against the brief's literal (unguarded) code, but the reasoning is
direct: Python's `f"{float('nan'):.1%}"` evaluates to `"nan%"` unconditionally, so the
brief's literal code would fail this test by construction.

## Files changed

- `src/nfl_game/market/compare.py` (new) — `SLATE_COLS`, `build_slate`, `slate_markdown`.
- `scripts/slate.py` (new) — weekly slate CLI, verbatim from the brief.
- `tests/test_compare.py` (new) — 6 brief tests + `test_sort_order_independent_of_input_row_order`
  + `test_markdown_handles_missing_cover_prob_without_rendering_nan`.
- `CLAUDE.md` (new) — brief's Step 7 content + a "Known issue" section documenting the
  `ryoe_diff`/`StandardScaler` calibration bug found in Step 6.

No changes to `src/nfl_game/model/features.py` or anything under `src/nfl_game/ratings/`,
per instructions. No changes to `model/predict.py`, `model/calibrate.py`, or
`backtest.py` — the bug found in Step 6 lives in the interaction between those and is
reported, not patched, per scope.

## Self-review findings

- Confirmed all 79 pre-existing tests still pass unchanged (87 total after adding 8).
- Confirmed `ruff check .` shows no new findings — only the pre-existing, already-logged
  `test_epa.py` I001 nit.
- Confirmed `ruff format --check` is clean on all four files I touched/created.
- Confirmed no stray warnings: ran `tests/test_compare.py` and the full suite under
  `-W error::FutureWarning -W error::DeprecationWarning` — all pass, no warnings.
- Confirmed the real `scripts/slate.py` run emits nothing on stderr and no warnings.
- Verified the generated `data/processed/slate_2025_wk01.{csv,md}` files are correctly
  excluded by the existing `data/processed/*` gitignore rule (not staged, not committed).
- Verified `market/__init__.py` already existed (empty), consistent with the other
  packages (`ratings/__init__.py`, `model/__init__.py`), so no new `__init__.py` was
  needed.
- Re-read `compare.py` end to end against the brief's Step 3 code: identical except for
  the two NaN-guard lines in `slate_markdown`, which is the one deliberate, documented
  deviation.
- Checked for YAGNI creep: added nothing beyond the brief's schema/CLI plus the two
  test additions and the NaN-guard; no speculative options, no unused parameters.

## Scaling fix pass

Follow-up pass fixing the `ryoe_diff`/`StandardScaler` bug documented above (found while
implementing Task 11, root cause diagnosed in the "Step 6 real-slate output" section).
Work done on top of `a2c37ef` (branch `feat/game-model`).

### RED — failing test written first

Added `test_near_constant_feature_with_near_zero_mean_does_not_explode_predictions` to
`tests/test_predict.py`, constructing a 256-row training frame where `ryoe_diff` is
`1e-17 * rng.normal(...)` (mean near zero, variance ~1e-34, mirroring the real 2016
training slice) and a test frame where `ryoe_diff` varies normally (mirroring a later
season with real NGS coverage). Ran against the pre-fix code:

```
.\.venv\Scripts\python.exe -m pytest tests\test_predict.py -q -k near_constant

FAILED tests/test_predict.py::test_near_constant_feature_with_near_zero_mean_does_not_explode_predictions
AssertionError: assert np.float64(5231873374134372.0) < 1000
 +  where np.float64(5231873374134372.0) = max()
 +    where max = 0    5.002065e+15
                 1    2.691764e+15
                 2    1.168178e+15
                 3    1.313043e+15
                 4    5.231873e+15
                 5    1.059581e+15
                 6    3.929268e+15
                 7    1.926033e+15
                 8    2.316740e+15
                 9    2.396829e+15
Name: model_margin, dtype: float64.max
1 failed, 11 deselected in 4.16s
```

Predictions of order 1e15, the same signature as the real bug (game_margin ~1.2e15 for
`2017_02_CHI_TB`) — confirms the RED failure is for the right reason, not an unrelated
assertion mistake.

### The fix

In `src/nfl_game/model/predict.py`, added `RobustStandardScaler(StandardScaler)`, used in
place of plain `StandardScaler` in the `"ridge"` pipeline. It overrides `partial_fit`
(which `StandardScaler.fit` calls internally, so both entry points are covered) to, after
calling `super().partial_fit(...)`, floor `scale_` to `1.0` wherever `scale_ < 10 * eps`
(`eps = np.finfo(scale_.dtype).eps`, i.e. ~2.22e-16 for float64 — so the floor threshold
is ~2.22e-15).

**Why this threshold, and why it isn't a new invention:** sklearn's own
`_handle_zeros_in_scale` helper (`sklearn/preprocessing/_data.py`) already contains
exactly this absolute check — `constant_mask = scale < 10 * xp.finfo(scale.dtype).eps` —
but that branch only runs `if constant_mask is None`. `StandardScaler.partial_fit` always
computes and passes a `constant_mask` itself, via `_is_constant_feature(var_, mean_,
n_samples_seen_)`, which tests `var <= n_samples*eps*var + (n_samples*mean*eps)**2` — i.e.
variance relative to the *feature's mean*. For `ryoe_diff` in the 2016 slice
(`var_=1.380030e-34`, `mean_` near zero), the `(n*mean*eps)**2` term vanishes and the test
reduces to `var <= n*eps*var`, which is false for any nonzero `var_`, however tiny — so
the near-zero-mean case defeats sklearn's own guard and it never gets to apply the
absolute fallback it already wrote. `RobustStandardScaler` just re-applies that same
already-authored absolute check unconditionally after fitting, OR'd on top of sklearn's
mean-relative one. Verified: `1.174747e-17 < 2.22e-15` (the real `ryoe_diff` scale from
the diagnosis) — caught. Real signal features (e.g. `off_rush_edge_home`, scale
`7.07e-2`) sit ~13 orders of magnitude above the floor — untouched.

**Alternatives considered and rejected:**
- *Absolute variance threshold hand-picked for this dataset (e.g. `var_ < 1e-12`).*
  Rejected: arbitrary, undocumented magic number tuned to today's feature scales; the
  `10*eps` floor is already sklearn's own considered choice for "this is numerical noise,
  not a real constant," so reusing it needs no new justification and no re-tuning if
  feature scales change.
- *Threshold relative to the largest variance among the other features in the same fit
  (e.g. `var_ < eps * max(var_)`).* Rejected: couples one feature's constant-detection to
  whatever unrelated features happen to be in the training slice that fold, which is
  harder to reason about and not obviously more correct than an absolute machine-epsilon
  floor — and the absolute floor already solves the problem without that coupling.
- *`VarianceThreshold` filtering step in the pipeline before scaling.* Rejected: drops the
  feature/column shape entirely (removes it from `X`), which would change `FEATURE_COLS`'
  effective contract fold-to-fold (a feature present in one fold's matrix and absent in
  another's) and break the fixed-width assumption `GameModel.predict` relies on
  (`df[FEATURE_COLS].to_numpy(...)`) unless carefully re-padded — much more invasive than
  necessary just to neutralize one feature's contribution.
- *Fix in `features.py`/`ratings/` (e.g. change how `ryoe_diff` is imputed for
  sparse-NGS seasons).* Out of scope per instructions, and the wrong layer regardless —
  the bug is that `StandardScaler` divides by near-zero, not that the imputation choice
  is wrong.

This keeps the fix local to `predict.py`, requires no new constants, is correct for a
feature that's constant only within one training slice (recomputed fresh on every `fit`
call, nothing persists across walk-forward folds), and is exactly the behavior sklearn's
own code already intends but doesn't reach for a near-zero-mean constant feature.

### GREEN — new test and full `test_predict.py` suite

```
.\.venv\Scripts\python.exe -m pytest tests\test_predict.py -q
............
12 passed in 12.11s
```

All 12 tests pass, including the two pre-existing scaling regression tests from
`5ddf154`:
- `test_predictions_are_invariant_to_feature_units` — PASSED (unit-invariance intact:
  rescaling `temp_outdoor` by 100x still produces identical `model_margin`).
- `test_ridge_recovers_signal_carried_by_a_small_scale_feature` — PASSED (small-scale
  `net_rating_diff` signal still recovered, MAE < 1.0; the original `5ddf154` fix — ridge
  no longer shrinks small-scale signal features — is not undone).

Full suite:
```
.\.venv\Scripts\python.exe -m pytest -q
........................................................................ [ 81%]
................                                                         [100%]
88 passed in 12.67s
```
(87 pre-existing + 1 new test, none weakened.) Also clean under
`-W error::FutureWarning -W error::DeprecationWarning` (88 passed, no warnings).

### Unchanged-backtest confirmation (critical regression check)

Did **not** rebuild `data/processed/game_features.parquet` — confirmed its mtime
(`Jul 24 17:29`) was unchanged before and after the run.

```
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025

=== ridge | test seasons 2021-2025 ===
games:            1359
margin MAE:       10.267   market: 9.752
total  MAE:       10.691   market: 10.309
ATS hit rate:     0.4992  (n=1326, break-even 0.5240)
O/U hit rate:     0.5022  (n=1348)

--- ATS by edge threshold ---
 min_edge    n  hit_rate
        0 1326  0.499246
        1 1008  0.495040
        2  714  0.478992
        3  515  0.469903
        4  336  0.488095
        6  129  0.550388

--- does the model add anything to the line? ---
market coef: 1.0673
model  coef: -0.0102   <- near zero means it adds nothing
r2: 0.2083  n=1359
```

**Identical to the expected numbers** (n=1359, margin MAE 10.267, market 9.752, ATS
0.4992 n=1326, model_coef -0.0102). Confirms the fix is a no-op on healthy multi-season
training data, exactly as expected — `RobustStandardScaler` only changes behavior for a
feature whose scale falls below the `10*eps` floor, which none of `FEATURE_COLS` do once
5+ seasons of real NGS variance are in the training set.

### New slate output, verbatim

```
.\.venv\Scripts\python.exe scripts\slate.py --season 2025 --week 1

| Game | Model | Market | Gap | Cover% | Model O/U | Market O/U | Gap | Over% | Edge |
|---|---|---|---|---|---|---|---|---|---|
| ARI @ NO | +1.6 | -6.0 | +7.5 | 47.8% | 48.3 | 44.5 | +3.8 | 48.4% | * |
| CIN @ CLE | -1.8 | -5.5 | +3.7 | 47.5% | 43.5 | 47.5 | -4.0 | 47.0% | * |
| DAL @ PHI | +5.0 | +8.5 | -3.5 | 46.9% | 46.6 | 47.5 | -0.9 | 47.6% | * |
| HOU @ LA | +6.2 | +3.5 | +2.8 | 47.4% | 47.2 | 43.5 | +3.7 | 48.4% | * |
| MIA @ IND | -1.2 | +1.5 | -2.7 | 46.9% | 48.0 | 47.5 | +0.5 | 47.8% | * |
| TB @ ATL | -4.0 | -1.5 | -2.5 | 46.9% | 48.1 | 47.5 | +0.6 | 47.8% | * |
| BAL @ BUF | +0.9 | -1.5 | +2.4 | 47.4% | 48.0 | 50.5 | -2.5 | 47.3% | * |
| PIT @ NYJ | -1.1 | -3.0 | +1.9 | 47.3% | 42.2 | 37.5 | +4.7 | 48.6% |  |
| KC @ LAC | -1.4 | -3.0 | +1.6 | 47.3% | 48.4 | 47.5 | +0.9 | 47.9% |  |
| CAR @ JAX | +3.0 | +4.5 | -1.4 | 47.0% | 48.1 | 45.5 | +2.6 | 48.2% |  |
| NYG @ WAS | +4.9 | +6.0 | -1.1 | 47.1% | 44.8 | 45.5 | -0.7 | 47.6% |  |
| DET @ GB | +2.6 | +1.5 | +1.1 | 47.2% | 49.0 | 48.5 | +0.5 | 47.8% |  |
| MIN @ CHI | -2.5 | -1.5 | -1.0 | 47.1% | 41.9 | 43.5 | -1.6 | 47.4% |  |
| TEN @ DEN | +8.0 | +8.5 | -0.5 | 47.1% | 45.1 | 42.5 | +2.6 | 48.2% |  |
| SF @ SEA | -2.6 | -2.5 | -0.1 | 47.1% | 46.9 | 43.5 | +3.4 | 48.4% |  |
| LV @ NE | +2.6 | +2.5 | +0.1 | 47.2% | 41.9 | 44.5 | -2.6 | 47.3% |  |

wrote C:\Users\sbuer\Documents\NFL Game Model\data\processed\slate_2025_wk01.csv
wrote C:\Users\sbuer\Documents\NFL Game Model\data\processed\slate_2025_wk01.md
```

**The degenerate flatness is gone.** `Cover%` now ranges 46.9%–47.8% and `Over%` ranges
47.0%–48.6%, and both move *monotonically* with their respective gap column: sorting by
`spread_gap`, `Cover%` rises strictly from 46.9% (gap -3.5) to 47.8% (gap +7.5). This is
the correct qualitative shape — the calibrator is now responsive to the size of the
model/market disagreement instead of returning a constant.

**Consistent with, not contradicting, Task 9.** All values sit within ~2.2 points of 50%,
and all 16 `Cover%` values happen to fall slightly below 50% — a mild, uniform-ish shift
plausible given `model_coef ≈ -0.0102` (near zero, slightly negative) rather than a sign
of newly-invented confident edge. Nothing here claims the model beats the market: 7 of 16
games flag at the default 2.0-point edge threshold (same games as before the fix; the
`edge_flag` math is driven by `spread_gap`/`total_gap`, which were never touched), and
probabilities stay in a tight band around 50%, not "confidently far from 50%." This
matches the expected outcome: the *flatness* was the bug; the *modest spread* is real and
unchanged.

### `scripts/slate.py`: skipping single-season-trained calibration folds

Changed `prior_seasons[1:]` to `prior_seasons[2:]` when building the `test_seasons` list
passed to `walk_forward` for calibration. Previously, for a target season like 2025 with
`prior_seasons = [2016..2024]`, `prior_seasons[1:] = [2017..2024]` still let `walk_forward`
train the 2017 fold on `2016` alone (`train = features_df[features_df["season"] < 2017]`)
— exactly the single-season, thin-data fold that exposed this bug. `prior_seasons[2:]`
drops 2017 too, so every remaining calibration fold trains on 2+ prior seasons.

**Reasoning for making this change (not just leaving it as "consider"):** the root-cause
fix in `predict.py` already makes every fold numerically safe regardless of how many
training seasons it has — so this change is not required for correctness anymore. But a
model trained on a single season is still a *low-quality* calibration data point on its
own terms (noisier ridge fit, thinner NGS coverage), independent of the scaling bug, and
there is no cost to excluding it: `walk_forward`'s and `Calibrator`'s **load-bearing
contract is unchanged** — the calibrator is still fit exclusively on `walk_forward`'s
out-of-sample predictions from strictly-prior seasons, never in-sample ones, and every
season from `prior_seasons[2:]` onward is still trained on strictly-earlier data than
its own test season, same as before. This is defense-in-depth on top of the real fix, not
a substitute for it — verified `scripts/backtest.py` (which never calls this code path)
and all other tests are unaffected by this change.

`src/nfl_game/backtest.py` (the `walk_forward` function itself) was left untouched, since
it is out of this task's stated scope and Task 9's backtest never exercises a
single-prior-season fold in the first place (`--test-seasons 2021-2025` always has 5+
prior seasons available).

## Issues or concerns

- **The Step 6 real-slate output is not the clean "matches the honest expectation"
  result the brief describes.** All probabilities are flat at exactly 0.500000 due to a
  real bug (near-zero-variance `ryoe_diff` in the 2016 training season causing a
  `StandardScaler` divide-by-near-zero blowup in two 2017-week-2 rows, which then
  poisons the `LogisticRegression` calibration fit). This is fully traced and documented
  in `CLAUDE.md` and above. I recommend the fix (a variance floor in the ridge pipeline's
  `StandardScaler`, or excluding thin-training-window seasons from the calibration
  corpus) be a follow-up decision at the final review, since it touches files reserved
  for human review (`ratings/`) or described as already-reviewed-clean (`model/predict.py`).
  The model's own predictions (`model_spread`/`model_total`) for the actual 2025-week-1
  slate are unaffected and look sane — only the calibrator's probabilities are corrupted.
- `test_sorted_by_absolute_edge` (brief-mandated) does not discriminate against a
  no-sort implementation, confirmed by direct experiment. I added a test that does
  discriminate rather than rewriting the original, per instructions; the reviewer should
  decide whether to keep, supersede, or retire the original.
- One deliberate, documented deviation from the brief's literal code: `slate_markdown`
  renders `"n/a"` instead of `"nan%"` for a NaN `cover_prob`/`over_prob`.
