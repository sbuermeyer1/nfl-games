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
