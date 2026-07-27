# Task 4 Report: Opponent-adjusted ratings

## What was implemented

Appended `fit_ratings(team_games, target="epa_play", alpha=1.0, weights=None) -> pd.DataFrame`
to `src/nfl_game/ratings/epa.py`, exactly as specified in the task brief. Added
`import numpy as np` and `from sklearn.linear_model import Ridge` to the top of the file, and
updated the module docstring (previously said "fit_ratings (Task 4)" as a forward reference;
now describes it as implemented).

`fit_ratings` regresses each team-game's target EPA value on offense-team and defense-team
one-hot indicators using ridge regression (L2 shrinkage toward the league mean, controlled by
`alpha`; optional `sample_weight` support via the `weights` param). It returns one row per team
with `off_rating` and `def_rating`, both oriented so **higher is always better** — the raw
defense coefficient represents "EPA allowed" (positive = bad defense), so it is negated before
being returned. The league mean (fitted intercept) is exposed via `.attrs["league_mean"]`.

Also created `tests/test_fit_ratings.py` with the 6 tests specified in the brief, verbatim.

## Files changed
- `src/nfl_game/ratings/epa.py` (modified — added imports, `fit_ratings`, docstring update)
- `tests/test_fit_ratings.py` (new)

## TDD evidence

**Step 1/2 — RED.** Command:
```
.\.venv\Scripts\python.exe -m pytest tests/test_fit_ratings.py -v
```
Output (abridged):
```
ERROR collecting tests/test_fit_ratings.py
ImportError: cannot import name 'fit_ratings' from 'nfl_game.ratings.epa'
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```
This is the expected failure: `fit_ratings` did not exist yet in `epa.py`, so the test module
fails to import, exactly as the brief predicted.

**Step 3 — implement.** Added the imports and the `fit_ratings` function verbatim from the brief
to `src/nfl_game/ratings/epa.py`.

**Step 4 — GREEN.** Command:
```
.\.venv\Scripts\python.exe -m pytest tests/test_fit_ratings.py -v
```
Output:
```
tests/test_fit_ratings.py::test_recovers_offensive_ordering PASSED       [ 16%]
tests/test_fit_ratings.py::test_higher_def_rating_means_better_defense PASSED [ 33%]
tests/test_fit_ratings.py::test_returns_one_row_per_team PASSED          [ 50%]
tests/test_fit_ratings.py::test_league_mean_available PASSED             [ 66%]
tests/test_fit_ratings.py::test_opponent_adjustment_beats_raw_average PASSED [ 83%]
tests/test_fit_ratings.py::test_sample_weights_shift_ratings PASSED      [100%]

============================= 6 passed in 14.69s ==============================
```

**Full suite** (confirms Task 3's tests still pass):
```
.\.venv\Scripts\python.exe -m pytest -q
...................                                                      [100%]
19 passed in 1.89s
```

## Step 5: real-2024 sanity check

Command:
```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_pbp; from nfl_game.ratings.epa import team_game_epa, fit_ratings; r=fit_ratings(team_game_epa(load_pbp([2024], save=False))); print('OFFENSE'); print(r.sort_values('off_rating',ascending=False).head(10)); print(); print('DEFENSE'); print(r.sort_values('def_rating',ascending=False).head(10)); print(); print('league_mean', r.attrs['league_mean'])"
```

Output:
```
OFFENSE
   team  off_rating  def_rating
2   BAL    0.190905    0.022695
10  DET    0.155235    0.065162
3   BUF    0.154855    0.011404
31  WAS    0.114320   -0.039501
29   TB    0.098592    0.002085
11   GB    0.090866    0.090188
6   CIN    0.084142   -0.073503
25  PHI    0.064880    0.102297
0   ARI    0.057224   -0.010676
15   KC    0.050606    0.003653

DEFENSE
   team  off_rating  def_rating
20  MIN    0.035659    0.109364
25  PHI    0.064880    0.102297
9   DEN   -0.007353    0.099886
11   GB    0.090866    0.090188
10  DET    0.155235    0.065162
17  LAC    0.008907    0.052264
12  HOU   -0.063987    0.051109
27  SEA   -0.021802    0.046232
26  PIT   -0.042794    0.039463
5   CHI   -0.074598    0.033652

league_mean 0.009643801400010296
```

**Judgment: sign convention is CORRECT.**
- Offense leaderboard top 3: BAL, DET, BUF — exactly the three teams the brief calls out as
  2024's best offenses, and they are in fact ranked #1-3.
- Defense leaderboard top 5: MIN, PHI, DEN, GB, DET — these are widely-regarded elite 2024
  defenses (Vikings under Brian Flores, Eagles, Broncos, Packers, Lions), not the league's worst.
  No sign flip needed.

## Self-review (fresh eyes)

- **Completeness**: All 6 brief-specified tests present and passing; function signature,
  docstring, and body match the brief verbatim; module docstring updated to remove the stale
  forward-reference to "Task 4."
- **Quality**: `ruff check` on both changed files passes with no findings. Full test suite output
  is pristine — no warnings, no skipped tests, 19/19 passed.
- **YAGNI**: No extra parameters, helpers, or abstractions added beyond what the brief specifies.
  Did not touch `team_game_epa` logic.
- **Constraints check**: `ratings/epa.py` imports only `numpy`, `pandas`, and
  `sklearn.linear_model.Ridge` — no import from `model`, `market`, or `nfl_ffm`. No test hits the
  network (all 6 new tests use the synthetic `_round_robin` fixture built in-memory); the Step 5
  network-touching sanity check was run manually outside the test suite, as instructed.
  All lines in the new code are within the 100-char limit per ruff.

No concerns. The implementation, tests, and sanity check all line up with the brief's central
correctness property: higher `off_rating` and higher `def_rating` both mean "better team."

## Commit
```
c6b9379 feat: opponent-adjusted team ratings via ridge on team dummies
 2 files changed, 139 insertions(+), 2 deletions(-)
 create mode 100644 tests/test_fit_ratings.py
```

## Review fix: weights misalignment on null-target rows

**Finding (Important, from Task 4 review).** `fit_ratings` filtered out null-`target` rows to
build `X`/`y` but passed the `weights` argument to `Ridge.fit(..., sample_weight=weights)`
UNFILTERED. If a caller supplies `weights` aligned to the original (pre-filter) `team_games`
frame and that frame contains any null-`target` rows, `weights` misaligns with `X`/`y`.

### RED evidence

Added `test_weights_aligned_to_prefilter_rows_with_null_target` to `tests/test_fit_ratings.py`
(builds a `team_games` frame with one row's `epa_play` set to `NaN`, supplies `weights` aligned
to the original, pre-filter frame length, and asserts the result equals calling `fit_ratings` on
the frame with the null row and its corresponding weight both dropped).

Ran against the pre-fix code:
```
.\.venv\Scripts\python.exe -m pytest tests/test_fit_ratings.py -v -k weights_aligned
```
Failed exactly as predicted — sklearn's `_check_sample_weight` raises a length-mismatch error
because `X`/`y` were filtered to 23 rows while `weights` was still 24 long:
```
E               ValueError: sample_weight.shape == (24,), expected (23,)!
.venv\Lib\site-packages\sklearn\utils\validation.py:2191: ValueError
FAILED tests/test_fit_ratings.py::test_weights_aligned_to_prefilter_rows_with_null_target
1 failed, 6 deselected in 2.74s
```

### Fix

In `src/nfl_game/ratings/epa.py::fit_ratings`, extracted the boolean mask so it can be reused:
```python
mask = team_games[target].notna()
df = team_games[mask].copy()
if df.empty:
    raise ValueError(f"no rows with non-null {target!r}")
...
if weights is not None:
    weights = np.asarray(weights)[mask.to_numpy()]

model = Ridge(alpha=alpha, fit_intercept=True)
model.fit(X, y, sample_weight=weights)
```
Also added one sentence to the docstring noting `weights`, when given, is expected to be aligned
to `team_games` (pre-filter) rows. No other behavior changed.

### GREEN evidence

```
.\.venv\Scripts\python.exe -m pytest tests/test_fit_ratings.py -v
```
```
tests/test_fit_ratings.py::test_recovers_offensive_ordering PASSED       [ 14%]
tests/test_fit_ratings.py::test_higher_def_rating_means_better_defense PASSED [ 28%]
tests/test_fit_ratings.py::test_returns_one_row_per_team PASSED          [ 42%]
tests/test_fit_ratings.py::test_league_mean_available PASSED             [ 57%]
tests/test_fit_ratings.py::test_opponent_adjustment_beats_raw_average PASSED [ 71%]
tests/test_fit_ratings.py::test_sample_weights_shift_ratings PASSED      [ 85%]
tests/test_fit_ratings.py::test_weights_aligned_to_prefilter_rows_with_null_target PASSED [100%]
7 passed in 1.29s
```

Full suite:
```
.\.venv\Scripts\python.exe -m pytest -q
....................                                                     [100%]
20 passed in 2.06s
```

Ruff on changed files:
```
.\.venv\Scripts\python.exe -m ruff check src/nfl_game/ratings/epa.py tests/test_fit_ratings.py
All checks passed!
```

### Files changed
- `src/nfl_game/ratings/epa.py` (modified — mask reused to filter `weights`; docstring note added)
- `tests/test_fit_ratings.py` (modified — added the regression test; the original 6 tests are
  untouched)

### Commit
```
b3b5108 fix: filter sample weights to match non-null target rows in fit_ratings
 2 files changed, 30 insertions(+), 1 deletion(-)
```
