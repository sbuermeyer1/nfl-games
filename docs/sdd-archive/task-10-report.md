# Task 10: Probability calibration — report

## What I implemented

- `src/nfl_game/model/calibrate.py` — `Calibrator` (`.fit`, `.predict`), `brier_score`,
  `reliability_table`. Written from the brief's Step 3 code with **one deliberate
  deviation** in `Calibrator.fit`'s row filter (see design-point section below).
- `tests/test_calibrate.py` — written verbatim from the brief's Step 1 code.

## TDD Evidence

**RED** — `.\.venv\Scripts\python.exe -m pytest tests/test_calibrate.py -v`, before
`src/nfl_game/model/calibrate.py` existed:

```
collecting ... collected 0 items / 1 error
ERROR collecting tests/test_calibrate.py
tests\test_calibrate.py:5: in <module>
    from nfl_game.model.calibrate import Calibrator, brier_score, reliability_table
E   ModuleNotFoundError: No module named 'nfl_game.model.calibrate'
=========================== short test summary info ===========================
ERROR tests/test_calibrate.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 1.35s ===============================
```

This is exactly the failure the brief predicted, for the expected reason: the module
didn't exist yet.

**GREEN** — `.\.venv\Scripts\python.exe -m pytest tests/test_calibrate.py -v`, after
writing `src/nfl_game/model/calibrate.py`:

```
tests/test_calibrate.py::test_predict_returns_expected_columns PASSED    [ 14%]
tests/test_calibrate.py::test_probabilities_are_in_range PASSED          [ 28%]
tests/test_calibrate.py::test_bigger_edge_means_higher_cover_probability PASSED [ 42%]
tests/test_calibrate.py::test_zero_edge_is_near_a_coin_flip PASSED       [ 57%]
tests/test_calibrate.py::test_brier_score_rewards_accuracy PASSED        [ 71%]
tests/test_calibrate.py::test_reliability_table_shape PASSED             [ 85%]
tests/test_calibrate.py::test_predict_before_fit_raises PASSED           [100%]
============================== 7 passed in 2.85s ==============================
```

Matches the brief's "Expected: 7 passed" exactly.

Also ran with `-W error::FutureWarning -W error::DeprecationWarning`: still 7 passed,
no warnings raised (the `pd.cut`/`groupby(observed=True)` combination in
`reliability_table` is warning-clean).

**Full suite before committing:**
`.\.venv\Scripts\python.exe -m pytest -q` → `72 passed in 10.02s` (the pre-existing 65
plus the 7 new tests; none of the original 65 were touched).

**Lint/format on changed files:**
```
.\.venv\Scripts\python.exe -m ruff format --check src/nfl_game/model/calibrate.py tests/test_calibrate.py
2 files already formatted
.\.venv\Scripts\python.exe -m ruff check src/nfl_game/model/calibrate.py tests/test_calibrate.py
All checks passed!
```

## Design decision: null line values in `Calibrator.fit`

**Decision: filter on the full required-column set (margin, total_points, spread_line,
total_line, model_margin, model_total), not just the two the brief's literal code
checks. This is a deviation from the brief's Step 3 code.**

Reasoning:

- `spread_edge = model_margin - spread_line`. If `spread_line` is NaN, the edge is NaN.
  `LogisticRegression().fit` does not tolerate NaN in `X` — it raises `ValueError:
  Input contains NaN` at fit time, not silently. So the brief's literal filter
  (`margin.notna() & total_points.notna()`) doesn't just risk quietly wrong results the
  way the Task 9 `evaluate()` bug did — with any row missing a posted line, it would
  make `fit()` hard-crash on real walk-forward data (any season/week with an unposted
  line, e.g. preseason artifacts or a data gap, produces this pattern). That's worse
  than Task 9's bug, not better: Task 9's `NaN != NaN` bug silently produced *wrong*
  numbers; this one would silently produce *no calibrator at all* the first time real
  data has a hole in it.
- The failure mode is exactly the shape described in the Task 9 review: a partial-column
  null check lets rows with missing lines slip past a filter that was clearly intended
  to gate them out (the brief's own docstring twice states this fits on lines-vs-model
  disagreement, which is meaningless without both a line and a model prediction).
- I extended the filter to require all six columns non-null (same set as
  `backtest._REQUIRED_COLS`), applied via `preds[list(_REQUIRED_COLS)].notna().all(axis=1)`
  — same pattern as `_valid_games`, just reimplemented locally rather than imported.
- I deliberately did **not** import `_valid_games`/`_REQUIRED_COLS` from
  `nfl_game.backtest` into `calibrate.py`. Reasoning: `calibrate.py` lives under
  `nfl_game.model` and is meant to consume `walk_forward`'s output shape, not to depend
  on the backtest module's internals. `backtest.py` already imports from
  `nfl_game.model.predict` (model → backtest is the existing dependency direction);
  adding `calibrate.py → backtest.py` would point a dependency the other way for a
  6-line tuple, and would make `model/calibrate.py` fail to import in isolation if
  `backtest.py` ever grew a heavier import (e.g. imports GameModel, sklearn's
  LinearRegression). The duplication is small (6 strings + a `.notna().all(axis=1)`
  one-liner) and I judged it cheaper than the coupling. I flag this for the reviewer in
  case they'd rather see the constant/helper shared (e.g. pulled up into a small
  `nfl_game/_validation.py`) instead of duplicated — I didn't do that myself since it
  touches file organization beyond what Task 10 asked for.
- I left `Calibrator.predict` unchanged from the brief (no added filtering). Predict
  must return exactly one row per input row (`game_id` 1:1), so silently dropping rows
  with a NaN line would break that contract; a NaN edge there just produces a NaN
  `cover_prob`/`over_prob` for that row, which is the honest answer ("no line, no
  prediction") rather than a crash or a silently wrong number. Verified this manually:
  injecting a NaN `spread_line` into `fit()`'s input no longer raises (rows silently
  excluded from training), and `predict()` on clean data afterward works normally. No
  test exercises predict-with-NaN-line explicitly since it wasn't in the brief's test
  list and doesn't change any documented interface behavior.

## Brief-mandated test discrimination review

Went through each test asking "would it fail if the described behavior broke?":

- `test_predict_returns_expected_columns` — discriminates (wrong columns/order fails it).
- `test_probabilities_are_in_range` — weak on its own (any constant in [0,1] passes) but
  is not meant to catch logic errors alone; it's a sanity/contract check. Not
  non-discriminating in a harmful way — it correctly guards against `predict_proba`
  misuse (e.g. returning raw margins instead of column 1 of predict_proba, which would
  push values outside [0, 1] range in this synthetic setup) — actually verified: I
  temporarily changed `[:, 1]` to `[:, 0]` and reran; `test_bigger_edge_means_higher...`
  failed (direction flipped) rather than this one, since both columns sum to 1 and both
  lie in [0,1]. So this test alone would NOT catch a `[:, 0]` vs `[:, 1]` swap — that's
  caught by `test_bigger_edge_means_higher_cover_probability` instead. Flagging only for
  completeness; the suite as a whole does catch it.
- `test_bigger_edge_means_higher_cover_probability` — discriminates well; verified by
  swapping `predict_proba[:, 1]` → `[:, 0]` locally, which flips the comparison and
  fails this test (reverted after checking).
- `test_zero_edge_is_near_a_coin_flip` — discriminates; a badly-fit or reversed-sign
  model would place this far from 0.5. `abs=0.08` gives it real teeth given
  `LogisticRegression`'s default L2 regularization pulls toward 0 at zero-edge with this
  much data.
- `test_brier_score_rewards_accuracy` — discriminates: a broken `brier_score`
  (e.g. `outcomes` and `probs` swapped in sign, or squared-error computed incorrectly)
  is fairly likely to fail this, though I'll note a `brier_score` that always returns a
  constant would trivially fail it too (good, since `good < bad` requires distinct
  values) — so this test does catch the "replaced with constant" failure mode Task 9's
  review warned about, unlike some Task 9 tests.
- `test_reliability_table_shape` — discriminates on column names/bin count but not on
  whether `mean_pred`/`observed` are correctly computed per bin (a swapped `mean_pred`
  and `observed` column would still pass this test, since it only checks the column
  *set*, not values). I consider this a real but minor gap — flagging it rather than
  silently strengthening the brief-mandated test. I did not add an additional test for
  this since the brief specifies this test verbatim and instructs me not to silently
  strengthen brief-mandated tests.
- `test_predict_before_fit_raises` — discriminates (removing the guard fails it
  immediately with `AttributeError`/`TypeError` instead of `RuntimeError` matching
  "fit").

No test was found where a clearly-plausible one-line regression (sign flip, swapped
column, constant substitution) leaves the *entire* suite green — every mutation I tried
by hand was caught by at least one test, even where an individual test wouldn't have
caught it alone.

## Files changed

- `src/nfl_game/model/calibrate.py` (new)
- `tests/test_calibrate.py` (new)

## Self-review findings

- Verified `_REQUIRED_COLS` filter change doesn't affect the synthetic test fixture
  (`_preds()` never produces NaNs), so all 7 brief tests pass unchanged with the
  stricter filter — the deviation is purely defensive, not a behavior change on the
  brief's own test data.
- Manually confirmed (see decision section) that a NaN in `spread_line`/`total_line`
  no longer crashes `fit()` and is silently excluded from training, matching the
  `_valid_games` precedent in `backtest.py`.
- Confirmed no ruff format/check issues, no pytest warnings under strict warning
  filters.
- Did not touch `src/nfl_game/model/features.py` or anything under
  `src/nfl_game/ratings/`, per instructions.
- File size: `calibrate.py` is 75 lines, well within a single clear responsibility —
  no split needed.

## Issues or concerns

- The one open question I'm flagging rather than resolving myself: whether
  `_REQUIRED_COLS`/`_valid_games` should be pulled into a shared location (e.g. a small
  `nfl_game/_validation.py`) instead of being duplicated between `backtest.py` and
  `calibrate.py`. I chose duplication over a `calibrate.py → backtest.py` import to keep
  `model/calibrate.py`'s dependency graph pointed the same direction as the rest of
  `nfl_game.model` (features/predict have no reverse dependency on `backtest.py`
  either). Six lines of duplicated constant + a one-line boolean mask felt like the
  cheaper cost versus a new cross-module coupling, but this is a judgment call the
  reviewer may weigh differently.
- `test_reliability_table_shape`'s weak spot noted above (doesn't verify
  `mean_pred`/`observed` are the right way round) is a pre-existing brief-mandated test
  gap, not something I introduced.

## Fix pass

Addressed four review findings (I1/I2 important, I3/I4 minor). TDD throughout: test
first, watch it fail for the right reason (or, for I1/I4 where the implementation was
already correct, confirm the new test passes and then prove it discriminates by
mutation), then fix.

### Finding 1 — missing directional coverage for `over_prob`

Added two tests mirroring the existing cover-side pair:
- `test_bigger_edge_means_higher_over_probability` — fixes `total_line` at 0, varies
  `model_total` from 1.0 to 7.0 across two rows, asserts `over_prob` strictly increases.
- `test_zero_total_edge_is_near_a_coin_flip` — `total_line=0`, `model_total=0`, asserts
  `over_prob == pytest.approx(0.5, abs=0.08)`.

Both passed immediately (the implementation was already correct; only the test
coverage was missing). Mutation evidence — each of the three mutations from the
review's table applied to `src/nfl_game/model/calibrate.py`, confirmed the new tests
fail, then reverted:

| Mutation | Before (7-test suite) | After (new tests added) |
|---|---|---|
| `went_over = total_points > total_line` flipped to `<` | SURVIVES | `test_bigger_edge_means_higher_over_probability` FAILS (`0.279 > 0.457` assertion error) |
| `over_prob` uses `predict_proba(...)[:, 0]` instead of `[:, 1]` | SURVIVES | `test_bigger_edge_means_higher_over_probability` FAILS (same assertion, direction flipped) |
| `over_prob` hardcoded to constant `0.5` | SURVIVES | `test_bigger_edge_means_higher_over_probability` FAILS (`0.5 > 0.5` assertion error) |

All three mutations reverted afterward; `git diff` on `calibrate.py` was empty before
moving to the next finding.

### Finding 2 — `predict()` crashed the whole batch on one NaN

Wrote three tests first (RED against the original code):
- `test_predict_all_clean_batch_has_no_nan_probs` — passed immediately (no regression
  risk here).
- `test_predict_with_nan_spread_line_yields_nan_cover_prob_only` — FAILED with
  `sklearn.ValueError: Input X contains NaN`, reproducing the reviewer's report exactly.
- `test_predict_with_nan_total_line_yields_nan_over_prob_only` — same crash, symmetric
  case.

Fix: `Calibrator.predict` now builds `cover_prob`/`over_prob` as `np.full(len(preds),
np.nan)` arrays, computes a boolean mask per target from that target's own two inputs
(`model_margin`/`spread_line` for cover, `model_total`/`total_line` for over), and only
calls `predict_proba` on the masked-in subset, writing results back into the
pre-sized NaN array by boolean-mask assignment. Output is still one row per input row,
same order, columns exactly `game_id, cover_prob, over_prob` — the two probabilities
are computed from fully independent masks, so a row missing only `total_line` still
gets a real `cover_prob`.

All three tests pass after the fix; re-ran full `test_calibrate.py` — no regressions.

### Finding 3 — `fit()` coupled the two training sets

Replaced the single `_REQUIRED_COLS` (union of all six columns) with `_COVER_COLS =
("margin", "spread_line", "model_margin")` and `_OVER_COLS = ("total_points",
"total_line", "model_total")`, filtered independently before each model's `.fit()`.

Test written first (RED against original code):
`test_fit_trains_cover_model_independently_of_total_line_nulls` fits calibrator A on a
40-row clean dataset and calibrator B on the same data with `total_line` set to NaN
for 5 rows (rows otherwise fully valid for the cover target). Because `total_line`
never enters the cover computation (`model_margin - spread_line`), the two cover
models must be numerically identical if trained on the same rows — asserted via
`pd.testing.assert_series_equal` on `cover_prob` predictions from a shared probe set.

Before the fix: FAILED — the union filter dropped those 5 rows from cover training
too, so calibrator B's cover model differed measurably from A's (e.g. probe row 0:
`0.467` vs `0.485`). After the fix: PASSED — cover model B trains on all 40 rows
(only `total_line` is missing, which cover doesn't need), identical to A.

### Finding 4 — `reliability_table` aggregation semantics untested

First attempt used `Calibrator.predict()` output as test data and asserted
`mean_pred` falls within its own bin's edges. That test passed against both the
correct implementation *and* the `mean_pred`/`observed` swap mutation — the synthetic
data turned out to be well-calibrated enough that `mean(outcome)` per bin coincidentally
landed in the same range as `mean(prob)` per bin, so the mutation didn't change the
pass/fail outcome. Discarded that version rather than ship a non-discriminating test.

Replacement — `test_reliability_table_mean_pred_within_bin_edges_but_observed_need_not_be`
— calls `reliability_table` directly with deliberately miscalibrated synthetic data:
`probs = [0.05, 0.1, 0.9, 0.95]`, `outcomes = [1, 1, 0, 0]` (low-probability rows all
turned out correct, high-probability rows all turned out wrong). Asserts every bin's
`mean_pred` falls within that bin's own edges (true by construction — `mean_pred` is a
mean of the same values used to build the bin), and additionally pins down the lowest
bin numerically (`mean_pred == approx(0.075)`, `observed == approx(1.0)`, explicitly
outside that bin's edges) to make the asymmetry obvious.

Mutation evidence: swapped `mean_pred=("p","mean"), observed=("y","mean")` to
`mean_pred=("y","mean"), observed=("p","mean")` in `reliability_table`. New test
FAILED (`assert 1.0 <= 0.25` — `mean_pred` for the low bin became the outcome mean
`1.0`, outside the bin's own `[≈0, 0.25]` edges). Reverted; `git diff` on
`calibrate.py` confirmed clean.

### Verification

```
.\.venv\Scripts\python.exe -m pytest -q
79 passed in 13.60s
```
(72 original + 7 new: 2 for Finding 1, 3 for Finding 2, 1 for Finding 3, 1 for Finding 4)

```
.\.venv\Scripts\python.exe -m pytest -q -W error::FutureWarning -W error::DeprecationWarning
79 passed in 712.00s
```

```
.\.venv\Scripts\python.exe -m ruff format --check src/nfl_game/model/calibrate.py tests/test_calibrate.py
2 files already formatted

.\.venv\Scripts\python.exe -m ruff check src/nfl_game/model/calibrate.py tests/test_calibrate.py
All checks passed!
```

`git status` confirmed clean of leftover mutations before committing (only the
intentional `calibrate.py`/`test_calibrate.py` diffs staged).

### Files changed

- `src/nfl_game/model/calibrate.py` — split `_REQUIRED_COLS` into `_COVER_COLS`/
  `_OVER_COLS`; `fit()` filters each target independently; `predict()` computes
  `cover_prob`/`over_prob` on independent per-row masks instead of one shared array,
  degrading to NaN per-row-per-target instead of crashing the batch.
- `tests/test_calibrate.py` — added 7 tests: 2 for Finding 1 (over-side directional
  coverage), 3 for Finding 2 (NaN batch handling: all-clean, NaN spread, NaN total),
  1 for Finding 3 (independent training-set filtering), 1 for Finding 4 (reliability
  table aggregation semantics).
