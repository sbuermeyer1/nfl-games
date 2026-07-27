# Task 5 Report: As-of rating table with recency decay

## Summary

Implemented `src/nfl_game/ratings/build.py` with `decay_weights`, `build_ratings`, and
`ratings_by_week`, plus `tests/test_build_ratings.py`, exactly per the task brief. Followed
TDD: wrote the test file first, confirmed the expected `ModuleNotFoundError`, implemented the
module verbatim from the brief, then confirmed green (7 passed).

## Files changed

- Created: `src/nfl_game/ratings/build.py`
- Created: `tests/test_build_ratings.py`

Both staged and committed together.

## TDD evidence

### Step 1/2 — RED

Command:
```
.\.venv\Scripts\python.exe -m pytest tests/test_build_ratings.py -v
```

Output (abridged):
```
ERROR collecting tests/test_build_ratings.py
ImportError while importing test module ...
tests\test_build_ratings.py:5: in <module>
    from nfl_game.ratings.build import build_ratings, decay_weights, ratings_by_week
E   ModuleNotFoundError: No module named 'nfl_game.ratings.build'
=========================== short test summary info ===========================
ERROR tests/test_build_ratings.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.98s ===============================
```

This is exactly the expected failure per Step 2 of the brief — the module didn't exist yet.

### Step 3 — implement

Wrote `src/nfl_game/ratings/build.py` verbatim from the brief (imports only `numpy`, `pandas`,
and `nfl_game.ratings.epa.fit_ratings` — no `model`/`market`/`nfl_ffm` imports).

### Step 4 — GREEN

Command:
```
.\.venv\Scripts\python.exe -m pytest tests/test_build_ratings.py -v
```

Output:
```
collected 7 items

tests/test_build_ratings.py::test_excludes_current_and_future_weeks PASSED [ 14%]
tests/test_build_ratings.py::test_recent_games_weigh_more PASSED         [ 28%]
tests/test_build_ratings.py::test_prior_season_downweighted_by_penalty PASSED [ 42%]
tests/test_build_ratings.py::test_build_ratings_returns_all_rating_columns PASSED [ 57%]
tests/test_build_ratings.py::test_build_ratings_orders_teams_correctly PASSED [ 71%]
tests/test_build_ratings.py::test_build_ratings_raises_when_no_prior_data PASSED [ 85%]
tests/test_build_ratings.py::test_ratings_by_week_covers_every_week PASSED [100%]

============================== 7 passed in 2.15s
```

### Full suite (Tasks 3-4 regressions)

Command:
```
.\.venv\Scripts\python.exe -m pytest -q
```
Output:
```
...........................                                              [100%]
27 passed in 2.28s
```
Ran again after the ruff fix (see below) with the same result: `27 passed`.

### Ruff

Command:
```
.\.venv\Scripts\python.exe -m ruff check src/nfl_game/ratings/build.py tests/test_build_ratings.py
```

First run flagged one issue in the test file: `F401 numpy imported but unused` (the brief's
verbatim test file imports `numpy as np` but never calls `np.*` directly — all array
operations go through the returned `w` and pandas boolean masks). Removed the unused
`import numpy as np` line from `tests/test_build_ratings.py` (no other change) since the task
explicitly required ruff to be clean on new files, and the import wasn't load-bearing for any
assertion. Re-ran:
```
All checks passed!
```
Re-ran the focused suite after the edit to confirm nothing broke: `7 passed in 2.10s`.

## Commit

```
4df1e98 feat: as-of recency-weighted rating table with leak guard
 2 files changed, 158 insertions(+)
 create mode 100644 src/nfl_game/ratings/build.py
 create mode 100644 tests/test_build_ratings.py
```

## Self-review (leak-guard focus)

Read `build.py` back end-to-end with fresh eyes, specifically stress-testing the
strictly-before cutoff:

- `is_past = (season < asof_season) | ((season == asof_season) & (week < asof_week))` is a
  correct, direct encoding of "strictly before (asof_season, asof_week)" — equality on both
  season and week (i.e. the as-of week itself) is excluded, as is any later week or season.
- The final line of `decay_weights`, `np.where(is_past, w, 0.0)`, unconditionally zeroes any
  row that isn't `is_past`, **regardless of what the decay arithmetic computed for that row**.
  So even if `weeks_back`/`seasons_back` produced a nonsensical or negative value for a
  future/current-week row (they don't, since both are clamped with `np.maximum(..., 0)`, but
  even if they hadn't been), the gate still forces the weight to exactly 0. This is a
  belt-and-suspenders design: the decay math and the leak gate are independent, and the gate
  wins.
- `build_ratings` doesn't stop at zero weights — it physically drops those rows before ever
  calling `fit_ratings`: `used = team_games[w > 0]`. So the leak guard is enforced twice: once
  as a weight of zero, and once by removing the rows from the training frame entirely. Even if
  a future change to `fit_ratings` or the underlying `Ridge` model treated a zero
  `sample_weight` in some unexpected way, the excluded rows are never in `used` to be affected.
- `test_excludes_current_and_future_weeks` directly asserts `w[future] == 0` for all rows at
  or after the cutoff and `w[~future] > 0` for all rows before it — this is the assertion the
  brief calls out as most important, and it passed.
- `ratings_by_week` calls `build_ratings` with `asof_week=week` for each week present in the
  data — i.e., "ratings entering week W" use only games strictly before week W, never week W
  itself. Confirmed via `test_ratings_by_week_covers_every_week` (16 rows = 4 weeks x 4 teams,
  season 2024 only, no leakage of the target week's own games).

No correctness or leak-guard issues found. The only change from the brief's exact code was
removing one unused import in the test file to satisfy the ruff-clean requirement; the
production module (`build.py`) was implemented and committed unmodified from the brief.

Other checks:
- Column contract (`test_build_ratings_returns_all_rating_columns`) and team ordering
  (`test_build_ratings_orders_teams_correctly`) both pass.
- `test_build_ratings_raises_when_no_prior_data` confirms the `ValueError("no games before...")`
  path fires correctly when the as-of point has zero prior data (2023 week 1, no earlier
  season in the fixture).
- No imports from `model`/`market`/`nfl_ffm` in the new file — only `numpy`, `pandas`, and
  `nfl_game.ratings.epa.fit_ratings`.
- Line length and other ruff rules clean on both new files.
- No test hits the network — both files use only the synthetic in-memory `_games()` fixture.

## Concerns

None blocking. One minor, pre-existing-in-spec note (not a defect, just worth flagging for
whoever reviews): in `build_ratings`, the three per-target fits (`epa_play`, `epa_pass`,
`epa_rush`) are combined with `outer` merges on `team`. `fit_ratings` internally filters to
non-null rows per target, so if a target had null values for some team's games (not the case
in this task's synthetic fixture, where all three EPA columns are always populated), that
team could end up with `NaN` ratings for the affected target column(s) rather than being
dropped or imputed. This matches the brief's design exactly and isn't exercised by any test
here or in Tasks 3-4, so it's flagged for awareness rather than treated as a task-5 defect.
