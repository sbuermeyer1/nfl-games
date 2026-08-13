# Task 6 report: style and hidden-yardage features

## Scope

Added `src/nfl_game/ratings/style.py` and `tests/test_style.py`.

`team_game_style` produces one regular-season offensive row per team-game with neutral
pass tendency, pace, turnover and explosive-play rates, starting field position, and
posteam-assigned special-teams EPA. `style_features_for_targets` produces strictly
pre-target, exponentially weighted features, shrinks turnovers toward the league rate
with a 200-play prior, and marks teams with no eligible history as imputed.

## TDD evidence

- **RED:** `python -m pytest tests/test_style.py -v --basetemp .pytest-tmp-task6-red`
  failed at collection with the expected `ModuleNotFoundError: nfl_game.ratings.style`.
- **GREEN:** after implementation and formatting, the focused suite passed: **5 passed**.

The tests pin the formula denominators, pace cutoff, lost-fumble/interception turnover
definition, first-offensive-play field position, posteam special-teams attribution,
exponential as-of history, 200-play turnover shrinkage, target/future exclusion, and
no-history imputation.

## Validation

- `python -m pytest tests/test_style.py -v --basetemp .pytest-tmp-task6-postformat`:
  **5 passed**.
- `python -m pytest -q --basetemp .pytest-tmp-task6-full`: **502 passed**, with three
  pre-existing dependency/runtime warnings.
- `python -m ruff check .`: **All checks passed**.
- `git diff --check`: clean.

## Self-review and concerns

- The implementation is limited to Task 6's two source/test files plus this required
  report.
- The functions deliberately return numeric league/default values with `style_imputed=1`
  when no prior games exist, avoiding null model inputs while retaining the uncertainty
  signal.
- No concerns found in the final scope and diff review.

## Commit

`feat: add game style and hidden-yardage features`

## Review fix round 1

### RED / GREEN evidence

- RED: added a bye/cross-season ordinal-recency regression, ninth-game exclusion,
  shuffled-play ordering, and overlapping-turnover-flags coverage. The focused suite
  failed exactly on the five affected expectations: calendar-age weighting, no
  eight-game cap, input-order-dependent pace, input-order-dependent field position,
  and a double-counted turnover play.
- GREEN: the focused suite passes: **9 passed**.

### Decisions

- History is now sorted per team by `(season, week, game_id)` when present, restricted
  to the eight newest strictly pre-target games, and weighted by game ordinal with
  `0.5 ** (age / halflife)`. It has no prior-season penalty.
- Pace and first-valid-drive-play field position use numeric `play_id` ordering when
  available; absent that, quarter then descending game-clock order provides a
  deterministic fallback.
- A play flagged as both an interception and a lost fumble counts once via a boolean
  OR indicator.

### Verification

- `python -m pytest tests/test_style.py -v --basetemp .pytest-tmp-task6-round1-final-focused`:
  **9 passed**.
- `python -m pytest -q --basetemp .pytest-tmp-task6-round1-full`: **506 passed**,
  with three pre-existing dependency/runtime warnings.
- `python -m ruff check .`: passed.
- `git diff --check`: passed.

### Self-review

- The review fixes remain contained to `style.py`, `test_style.py`, and this required
  report. Strict target/future exclusion remains the first history filter, before
  per-team ranking and truncation.

## Review fix round 2

### RED / GREEN evidence

- RED: `python -m pytest tests/test_style.py -v --basetemp
  .pytest-tmp-task6-round2-red` produced **2 failed, 9 passed** before the production
  edit. The partial-ID regression got `pace_seconds=NaN` instead of `27.0`; the tied-clock
  reversal changed starting field position from `32.5` to `35.0`.
- GREEN targeted: `python -m pytest tests/test_style.py -k "partial_play_ids or
  tied_fallback_clocks" -v --basetemp .venv\pytest-tmp-task6-round2-targeted` produced
  **2 passed, 9 deselected**.
- GREEN focused: `python -m pytest tests/test_style.py -v --basetemp
  .venv\pytest-tmp-task6-round2-focused` produced **11 passed**.

### Decision and self-review

- Complete, unique numeric `play_id` values remain the preferred order.
- Partial, missing, or duplicate IDs now fall back to quarter/game-clock chronology,
  then use per-row numeric IDs and a canonical row-content key to break ties without
  consulting input position. Missing quarter/clock/ID values sort deterministically
  after present values.
- Canonically identical tied rows may retain stable input order, but they are identical
  for pace and starting-field-position inputs, so exchanging them cannot change either
  result.
- The change does not touch trailing-eight ordinal history, strict as-of filtering, or
  turnover-OR behavior.

### Validation

- `python -m pytest -q --basetemp .venv\pytest-tmp-task6-round2-full-final`:
  **508 passed**, with three pre-existing dependency/runtime warnings.
- `python -m ruff check .`: passed.
- `git diff --check`: passed.

### Concerns
- None. Reviewer-created untracked pytest directories were left untouched.
