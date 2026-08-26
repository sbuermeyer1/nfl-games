# Task 5 report: as-of quarterback context

## RED / GREEN

- RED: `tests/test_qb.py` initially failed during collection with
  `ModuleNotFoundError: No module named 'nfl_game.ratings.qb'`.
- GREEN: the focused suite passes: 8 passed.

## Decisions

- Aggregate weekly QB usage with `attempts + sacks_suffered` as dropbacks.
- Regress every QB rate to an as-of league prior using 200 prior dropbacks.
- Use rank-one QB depth-chart rows.  Pre-2025 charts use their assigned season/week;
  2025-and-later charts use their latest `dt` at or before the target kickoff.
- Never consult schedule `home_qb_id` or `away_qb_id`.  If chart history is unavailable,
  use the prior game's most-used team QB and set `qb_uncertain=1` before rate imputation.
- Calculate starter-change, rookie/unproven, and uncertainty features only from strictly
  prior weekly statistics.

## Verification

- `python -m pytest tests/test_qb.py -v --basetemp .venv\\pytest-tmp`: 8 passed.
- `python -m pytest -q --basetemp .venv\\pytest-tmp`: 494 passed (3 pre-existing dependency/runtime warnings).
- `python -m ruff check .`: passed.
- `git diff --check`: passed.

## Commit

`feat: add as-of quarterback context`

## Self-review and concerns

- Reviewed the feature boundary against the Task 5 brief, including future depth snapshots,
  prior-game fallback, schedule-QB leakage, and target output keys.
- The depth source's documented 2025+ `dt` cadence is treated as the availability timestamp;
  older charts remain bounded by their supplied season/week because they lack equivalent history.

## Review fix round 1

### RED / GREEN evidence

- Added a public-composition test that passes normalized 2025 depth history containing
  rank-1 `z-starter` and rank-2 `a-backup` into `qb_features_for_targets()`. RED selected
  `a-backup`; GREEN preserves the normalized `rank` and selects `z-starter`.
- Added regular-season coverage with a week-zero REG row and a week-19 POST row. RED emitted
  both rows; GREEN retains only the three positive-week REG baseline QBs.

### Verification

- `python -m pytest tests/test_qb.py -v --basetemp .venv\\pytest-tmp`: 10 passed.
- `python -m pytest -q --basetemp .venv\\pytest-tmp`: 496 passed (3 pre-existing dependency/runtime warnings).
- `python -m ruff check .`: passed.
- `git diff --check HEAD`: passed.

### Self-review

- `qb_week_stats()` filters `season_type == "REG"` only when that source column exists and
  always removes nonpositive weeks.
- Re-normalizing an already normalized depth history now recognizes `rank` before raw source
  rank fields, preserving the source chronology and rank-one starter selection.
## Review fix round 2

### RED / GREEN evidence

- Added a mixed normalized/raw 2025 depth-history regression. It combines a normalized
  rank-1 `z-starter` row with raw rank-2 `a-backup` and rank-1 opposing-QB rows, each at the
  same eligible timestamp. RED returned no A rows because global rank-column selection and
  position filtering discarded mixed-source values.
- GREEN coalesces `rank`, `depth_chart_position`, `depth_chart_rank`,
  `depth_chart_order`, and `depth` per row (in that precedence order), preserving normalized
  rank first and retaining both eligible A rows. The expected starter remains `z-starter`.

### Verification

- `python -m pytest tests/test_qb.py -v --basetemp .venv\\pytest-tmp`: 11 passed.
- `python -m pytest -q --basetemp .venv\\pytest-tmp`: 497 passed (3 pre-existing dependency/runtime warnings).
- `python -m ruff check .`: passed.
- `git diff --check HEAD`: passed.

### Self-review

- Rank precedence is now applied per row, preventing sparse normalized `rank` values from
  blanking valid legacy rank values in a concatenated history.
- A source `position` column only excludes explicitly non-QB rows; normalized rows that do
  not carry this raw-only column remain valid when mixed with raw rows.
- Timestamp and season/week chronology remain unchanged; the fix only normalizes field
  representation before eligibility is evaluated.