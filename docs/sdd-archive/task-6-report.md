# Task 6 Report: NGS offensive layer

## Status: DONE

## What was implemented

- `src/nfl_game/ratings/ngs.py` — new module, following the brief verbatim.
  - `NGS_METRICS: list[str]` — the eight metric names (cpoe, time_to_throw,
    air_yards_to_sticks, aggressiveness, ryoe_per_att, pct_eight_defenders,
    separation, yac_oe).
  - `team_week_ngs(passing, rushing, receiving) -> pd.DataFrame` — collapses
    player-level NGS frames to one volume-weighted row per team-week, imputes
    missing metrics with the league-week mean, and flags each with a
    `<metric>_imputed` column.
  - Internal helper `_weighted_team_week(df, mapping, weight_col)` filters to
    `season_type == "REG"` and `week > 0` (dropping season-aggregate rows),
    renames `team_abbr` -> `team`, and computes the volume-weighted mean per
    (season, week, team) group.
- `tests/test_ngs.py` — the 6 tests specified in the brief, copied verbatim
  (synthetic in-memory DataFrames only, no network access).

## TDD evidence

**RED** — command:
```
.\.venv\Scripts\python.exe -m pytest tests/test_ngs.py -v
```
Output (relevant excerpt):
```
ImportError while importing test module '...\tests\test_ngs.py'.
tests\test_ngs.py:3: in <module>
    from nfl_game.ratings.ngs import NGS_METRICS, team_week_ngs
E   ModuleNotFoundError: No module named 'nfl_game.ratings.ngs'
=========================== short test summary info ===========================
ERROR tests/test_ngs.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```
This matches the brief's expected failure exactly — the module didn't exist yet.

**GREEN** — command:
```
.\.venv\Scripts\python.exe -m pytest tests/test_ngs.py -v
```
Output:
```
tests/test_ngs.py::test_drops_week_zero_aggregates PASSED                [ 16%]
tests/test_ngs.py::test_attempt_weighted_aggregation PASSED              [ 33%]
tests/test_ngs.py::test_one_row_per_team_week PASSED                     [ 50%]
tests/test_ngs.py::test_missing_rushing_is_imputed_and_flagged PASSED    [ 66%]
tests/test_ngs.py::test_all_metrics_and_flags_present PASSED             [ 83%]
tests/test_ngs.py::test_postseason_passing_rows_are_excluded PASSED      [100%]

============================== 6 passed in 0.70s ==============================
```

## Full suite

```
.\.venv\Scripts\python.exe -m pytest -q
```
```
.................................                                        [100%]
33 passed in 2.97s
```
No regressions in the existing epa/build/data_nfl/fit_ratings/smoke tests.

## Ruff

```
.\.venv\Scripts\python.exe -m ruff check src/nfl_game/ratings/ngs.py tests/test_ngs.py
```
```
All checks passed!
```
Manually verified the two longest lines in `ngs.py` are exactly 100 chars
(the `NGS_METRICS` assembly line and the `_weighted_team_week` signature) —
within the configured `line-length = 100` limit.

## Step 5: real 2024 coverage check

Command:
```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_ngs; from nfl_game.ratings.ngs import team_week_ngs; p,r,c=[load_ngs([2024],s,save=False) for s in ('passing','rushing','receiving')]; t=team_week_ngs(p,r,c); print(t.shape); print(t[[col for col in t.columns if col.endswith('_imputed')]].mean().round(3))"
```
Output:
```
(544, 19)
cpoe_imputed                   0.009
time_to_throw_imputed          0.009
air_yards_to_sticks_imputed    0.009
aggressiveness_imputed         0.009
ryoe_per_att_imputed           0.140
pct_eight_defenders_imputed    0.140
separation_imputed             0.024
yac_oe_imputed                 0.024
dtype: float64
```

**Judgment: join key is correct.** 544 team-weeks (brief said "roughly 540" —
consistent; this is the full outer-join universe across all three NGS tables
for the 2024 season, 32 teams x 17 weeks). `cpoe_imputed` = 0.009, matching
the brief's "near 0.01" and the measured passing coverage (539/544 = 99.1%).
`ryoe_per_att_imputed` = 0.140 is an exact match to the brief's expected value
and to the measured rushing coverage (468/544 = 86.0% covered -> 14.0% missing).
This is well below the "far above 0.2" red-flag threshold the brief calls out,
so there is no join-key bug — the imputation rate reflects real NGS qualifier
thresholds, not a broken (season, week, team) join.

## Files changed

- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\ratings\ngs.py` (new)
- `C:\Users\sbuer\Documents\NFL Game Model\tests\test_ngs.py` (new)

## Commit

```
a6efaeb feat: aggregate Next Gen Stats to team-weeks with imputation flags
 2 files changed, 167 insertions(+)
 create mode 100644 src/nfl_game/ratings/ngs.py
 create mode 100644 tests/test_ngs.py
```

## Self-review (fresh eyes)

- **Completeness**: All 6 required tests present and passing; both required
  public interfaces (`NGS_METRICS`, `team_week_ngs`) implemented as specified.
  Step 5 coverage check run against real data and judged, not just executed.
- **Constraint compliance**: `ngs.py` imports only `pandas` — no import from
  `model`, `market`, or `nfl_ffm`, satisfying the `ratings` package boundary.
  No test touches the network; all fixtures are synthetic in-memory frames
  built with plain `pd.DataFrame(...)` calls.
- **Quality / YAGNI**: The module is a single cohesive unit — one grouping
  helper reused for all three input frames (passing/rushing/receiving), one
  imputation loop reused for all eight metrics. No speculative generalization
  (e.g. no defensive-side handling, matching the module docstring's explicit
  note that NGS has no defensive table).
- **Imputation fallback chain**: `fillna(league_mean).fillna(overall_mean).fillna(0.0)`
  guards against the edge case of an entire league-week missing a metric
  (league_mean all-NaN for that week) by falling back to the metric's global
  mean, and finally to 0.0 if the metric is missing everywhere. This path is
  untested by the brief's fixtures (all synthetic weeks have at least one real
  value per metric) but is inexpensive insurance and was verbatim brief code,
  so left as-is per instructions to implement the brief's code exactly.
- **Pristine test output**: `pytest -q` full-suite run is clean — no
  warnings, no skips, no deselected tests, 33/33 passed.
- **Test file note**: A comment in the brief's given test
  (`# league mean of 1`) doesn't match the actual computed value (0.8, since
  BUF's real `ryoe_per_att` is 0.8 and KC's imputed value is filled to that
  same league-week mean). The assertion itself only checks that KC's and
  BUF's values are equal, so the test is correct and passes; the comment is
  just a leftover/imprecise annotation in the brief. Not changed since the
  test was to be copied verbatim and the assertion semantics are unaffected.

## Concerns

None blocking. The one cosmetic nit is the stale `# league mean of 1` comment
in the brief's own test file (see above) — flagging it for visibility, not
treating it as a defect since the test's actual assertion is correct.
