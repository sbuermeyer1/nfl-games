# Task 7 Report: Game feature assembly

## What was implemented

Created `src/nfl_game/model/features.py` with:

- `FEATURE_COLS` — the 14 model input columns, in fixed order, exactly as specified in the brief.
- `TARGET_COLS = ["margin", "total_points"]`.
- `build_game_features(schedules, ratings, ngs, ngs_halflife=4.0) -> pd.DataFrame` — joins
  as-of ratings and trailing (strictly-prior-week) decay-weighted NGS onto each REG-season
  scheduled game, producing one row per game with fixed-order features plus targets.
- `_trailing_ngs(ngs, halflife)` — internal helper computing, per (season, team), a
  decay-weighted mean of each NGS metric over weeks strictly before the row's own week
  (uses `g.iloc[:i]` after sorting by week, so the current week's own data is never included
  in its own trailing feature). Weeks with no prior data get `NaN` (later filled to 0.0) and
  `trail_imputed_any = 1`.

Implementation follows the brief's Step 3 code verbatim, with one addition (see "Deviation
from brief" below).

Created `tests/test_features.py` with the brief's 8 test cases verbatim, plus one additional
trivial test (`test_target_cols_fixed`) — 9 tests total.

## Deviation from brief

The brief's verbatim test file imports `TARGET_COLS` but never asserts against it, which
ruff flags as `F401 imported but unused`. Since my task instructions require ruff-clean new
files, I resolved this by adding a small test:

```python
def test_target_cols_fixed():
    assert TARGET_COLS == ["margin", "total_points"]
```

This preserves the brief's exact import line, gives it real use, and adds a legitimate
(if trivial) check on the public contract, rather than silently deleting the import. No
other deviation from the brief's specified code or test cases.

## TDD evidence

**Step 1/2 — RED.** Wrote `tests/test_features.py` (brief's exact content), then ran:

```
.\.venv\Scripts\python.exe -m pytest tests/test_features.py -v
```

Output:

```
ERROR tests/test_features.py
ImportError while importing test module '...\tests\test_features.py'.
...
tests\test_features.py:4: in <module>
    from nfl_game.model.features import FEATURE_COLS, TARGET_COLS, build_game_features
E   ModuleNotFoundError: No module named 'nfl_game.model.features'
=========================== short test summary info ===========================
ERROR tests/test_features.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.99s ===============================
```

Matches the brief's expected failure exactly (`ModuleNotFoundError: No module named
'nfl_game.model.features'`) — the module did not exist yet.

**Step 3.** Implemented `src/nfl_game/model/features.py` per the brief.

**Step 4 — GREEN.**

```
.\.venv\Scripts\python.exe -m pytest tests/test_features.py -v
```

```
tests/test_features.py::test_produces_one_row_per_game PASSED            [ 12%]
tests/test_features.py::test_all_feature_columns_present_and_numeric PASSED [ 25%]
tests/test_features.py::test_targets_computed_from_scores PASSED         [ 37%]
tests/test_features.py::test_rating_edges_use_opposing_defense PASSED    [ 50%]
tests/test_features.py::test_rest_diff_is_home_minus_away PASSED         [ 62%]
tests/test_features.py::test_dome_zeroes_weather_and_sets_flag PASSED    [ 75%]
tests/test_features.py::test_ngs_features_exclude_current_week PASSED    [ 87%]
tests/test_features.py::test_future_games_kept_with_null_targets PASSED  [100%]
============================== 8 passed in 0.70s ==============================
```

8 passed — matches brief's expected "8 passed" exactly, before the `TARGET_COLS` test was
added. After adding `test_target_cols_fixed`:

```
9 passed in 0.75s
```

**Full suite (after commit):**

```
.\.venv\Scripts\python.exe -m pytest -q
..........................................                               [100%]
42 passed in 2.16s
```

No regressions; all prior tests (test_build_ratings, test_data_nfl, test_epa,
test_fit_ratings, test_ngs, test_smoke) still pass.

**Ruff on new files:**

```
.\.venv\Scripts\python.exe -m ruff check src/nfl_game/model/features.py tests/test_features.py
All checks passed!
```

A full-repo `ruff check src tests` turned up one pre-existing, unrelated issue in
`tests/test_epa.py` (import-sort ordering, `I001`) that predates this task and was not
touched — left as-is, out of scope.

## Files changed

- `src/nfl_game/model/features.py` (new)
- `tests/test_features.py` (new)

## Commit

```
2bf6cba feat: assemble game-level features from ratings and trailing NGS
```

## Self-review

**Leak guard (trailing NGS).** Verified airtight:
- `_trailing_ngs` sorts each (season, team) group by week and, for the row at position `i`,
  uses only `g.iloc[:i]` — strictly the rows before position `i` in sorted-week order, never
  including the current week's own data.
- `test_ngs_features_exclude_current_week` poisons week-2 CPOE to 99.0 and confirms
  `cpoe_diff` for the week-2 game is unchanged versus baseline — passes, confirming the
  week-2 game's trailing feature is built only from week-1 data.
- Ratings are joined via a plain merge on `(season, week, team)` against the `ratings`
  frame the caller passes in; Task 5's `ratings_by_week`/`build_ratings` already guarantees
  each `(season, week)` row was built from strictly-prior games (verified by reading
  `src/nfl_game/ratings/build.py`'s `decay_weights`, which zeroes weight for anything at or
  after the as-of cutoff). Task 7 does not re-derive or weaken that guarantee — it only
  consumes it.

**Future/unplayed games.** Verified:
- `build_game_features` filters only on `game_type == "REG"`, never on score presence, so
  unplayed games are never dropped.
- `margin`/`total_points` pass through `result`/`total` directly, so null scores produce
  null targets (`pd.isna` confirmed by test).
- Features (ratings edges, rest, dome/weather, div_game, NGS diffs) depend only on
  schedule/ratings/NGS fields keyed by `(season, week, team)`, never on the game's own
  score — so as long as the caller supplies ratings/NGS rows for that week (which Task
  5/6's builders do for every week including future ones, since they only require game
  data strictly before the cutoff), features are fully populated for unplayed games. This
  is exactly what makes the same function usable for both training and predicting an
  upcoming slate.

**Dome handling.** `is_dome` is derived from `roof in {"dome", "closed"}`. For dome games,
`temp_outdoor`/`wind_outdoor` are forced to `0.0` regardless of the (null) source values —
no temperature is imputed for indoor games; `is_dome` is the only carrier of that
information, as required. For outdoor games with a missing temp (e.g., a future game
without a weather forecast yet), `temp.fillna(60.0)` supplies a neutral placeholder so the
same function can still produce features for prediction — this is a deliberate fallback,
not a leak, since it's a fixed constant rather than any future-derived value.

**Completeness / YAGNI.** Implementation matches the brief's fixed feature list and column
order exactly; no extra features, no extra parameters, no speculative generalization beyond
what the tests require. The one addition (`test_target_cols_fixed`) is minimal and only
exists to satisfy the ruff-clean requirement without deleting a brief-specified import.

**Test output.** Pristine — no warnings, no deprecation notices, no stderr noise from
pytest or ruff on the new files.

## Concerns

None blocking. Two minor, non-blocking notes:
1. `out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)` at the end of `build_game_features`
   is a safety net that would silently zero-fill any feature that ended up NaN for reasons
   other than the dome case (e.g., if a caller passed `ratings`/`ngs` missing a row for some
   `(season, week, team)`). This matches the brief's code exactly and is a reasonable
   defensive default for a leak-free-by-construction pipeline, but it could mask a genuine
   upstream data gap rather than surfacing it loudly. Flagging for awareness, not fixing,
   since it's explicitly what the brief specified and no test exercises that failure mode.
2. Pre-existing `ruff I001` in `tests/test_epa.py` (unrelated to this task) — not touched.

## Formatting fix (post-completion)

**Issue:** Two lines in `tests/test_features.py` exceeded the 100-char Global Constraint:
- Line 139: 106 chars
- Line 141: 101 chars

**Fix applied:**

```
.\.venv\Scripts\python.exe -m ruff format tests/test_features.py src/nfl_game/model/features.py
2 files reformatted
```

**Verification:**

Ruff format --check:
```
.\.venv\Scripts\python.exe -m ruff format --check tests/test_features.py src/nfl_game/model/features.py
2 files already formatted
```

Full suite:
```
.\.venv\Scripts\python.exe -m pytest -q
..........................................                               [100%]
42 passed in 3.56s
```

All lines now within 100-char limit. No behavioral changes — all 42 tests still passing.

**Commit:** `c1af473 style: ruff-format Task 7 files to 100-char line length`
