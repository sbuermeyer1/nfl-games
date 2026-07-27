# Task 9: Walk-forward backtest — report

## What I implemented

- `src/nfl_game/backtest.py` — `walk_forward`, `evaluate`, `market_comparison_regression`,
  `ats_by_threshold`. Written verbatim from the brief's Step 3 code.
- `scripts/build_dataset.py` — builds and caches `data/processed/game_features.parquet`.
  Written from the brief's Step 5 code with **one necessary deviation** (see "Deviation
  from the brief" below): it now loads one extra prior season of play-by-play purely to
  seed rating history, so week 1 of the first requested season has strictly-prior games
  to rate off of.
- `scripts/backtest.py` — CLI backtest report. Written verbatim from the brief's Step 6
  code.
- `tests/test_backtest.py` — written verbatim from the brief's Step 1 code (ruff's
  `--fix` later removed an unused `pytest` import that was in the brief's own listing but
  never used).

## TDD Evidence

**RED** — `.\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v`, before
`src/nfl_game/backtest.py` existed:

```
ERROR collecting tests/test_backtest.py
ImportError while importing test module '...\tests\test_backtest.py'.
...
tests\test_backtest.py:5: in <module>
    from nfl_game.backtest import evaluate, market_comparison_regression, walk_forward
E   ModuleNotFoundError: No module named 'nfl_game.backtest'
=========================== short test summary info ===========================
ERROR tests/test_backtest.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 1.30s ===============================
```

This is exactly the failure the brief predicted (`ModuleNotFoundError: No module named
'nfl_game.backtest'`), for the expected reason: the module didn't exist yet.

**GREEN** — `.\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v`, after
writing `src/nfl_game/backtest.py`:

```
collected 7 items

tests/test_backtest.py::test_walk_forward_only_scores_test_seasons PASSED [ 14%]
tests/test_backtest.py::test_walk_forward_never_trains_on_the_test_season PASSED [ 28%]
tests/test_backtest.py::test_walk_forward_skips_season_with_no_prior_data PASSED [ 42%]
tests/test_backtest.py::test_evaluate_reports_model_and_market_mae PASSED [ 57%]
tests/test_backtest.py::test_evaluate_reports_ats_hit_rate_and_n PASSED  [ 71%]
tests/test_backtest.py::test_evaluate_excludes_pushes_from_ats PASSED    [ 85%]
tests/test_backtest.py::test_market_regression_returns_both_coefficients PASSED [100%]

============================== 7 passed in 3.43s ==============================
```

Full suite after: `60 passed in 8.99s` (53 pre-existing + 7 new), no warnings, both before
and after a ruff format/fix pass.

## Deviation from the brief (read this before trusting Step 7's numbers)

Running `scripts/build_dataset.py --start-season 2016 --end-season 2025` exactly as
written in the brief fails immediately:

```
ValueError: no games before season 2016 week 1
```

This is `build_ratings` (in `ratings/build.py`, a Task 8 file, untouched by me) correctly
refusing to rate a team on zero prior games — `ratings_by_week` loops over the requested
`seasons` (2016-2025), and since `load_pbp(seasons)` only pulls those same seasons, there
are, by construction, no games strictly before season-2016-week-1 for it to weight. This
is a real gap between the brief's default parameters and `build_ratings`'s (intentional,
tested) hard-fail behavior, not something I introduced.

Fix, scoped only to my own new file (`scripts/build_dataset.py`), leaving `features.py`,
`ratings/build.py`, and every other Task 1-8 file untouched: load one extra season of
play-by-play (`args.start_season - 1` through `args.end_season`) purely to seed
`team_game_epa`/`ratings_by_week` with history, while the final feature set is still
filtered down to the originally requested `seasons` via the existing
`feats = feats[feats["season"].isin(seasons)]` line. NGS loading is untouched (still uses
`seasons`, matching the brief) — NGS already degrades gracefully to imputed
zero-with-a-flag when there's no trailing history, so it didn't need a warm-up season.

I'm flagging this explicitly rather than silently patching around it, per the brief's own
instruction to treat unexpected behavior as something to investigate, not guess past.

## Step 7: the real backtest, verbatim

### `build_dataset.py`

```
loading pbp for 2015-2025 (this takes a few minutes)...
building as-of ratings...
building NGS team-weeks...
assembling features...
wrote 2639 games to C:\Users\sbuer\Documents\NFL Game Model\data\processed\game_features.parquet
```

### `backtest.py --test-seasons 2021-2025` (ridge, alpha=1.0 default)

```
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

### `backtest.py --test-seasons 2021-2025 --estimator gbm`

```
=== gbm | test seasons 2021-2025 ===
games:            1359
margin MAE:       10.671   market: 9.752
total  MAE:       11.061   market: 10.309
ATS hit rate:     0.4759  (n=1326, break-even 0.5240)
O/U hit rate:     0.4963  (n=1348)

--- ATS by edge threshold ---
 min_edge    n  hit_rate
        0 1326  0.475867
        1 1083  0.466297
        2  867  0.465975
        3  676  0.463018
        4  525  0.459048
        6  273  0.527473

--- does the model add anything to the line? ---
market coef: 1.0973
model  coef: -0.0535   <- near zero means it adds nothing
r2: 0.2086  n=1359
```

### Reading these against "how to read the results"

Market margin MAE on these 1359 games is 9.752 — squarely inside the stated 9.8-10.3
reference band (a hair below it, which is fine; it's the market's own number, not the
model's). Both estimators' margin MAE (10.267 ridge, 10.671 gbm) land just **above** the
market's, i.e. the model is slightly worse than the closing line, which is the expected,
non-leaked outcome for a market-blind model. Total MAE shows the same pattern (10.691 /
11.061 vs market 10.309).

ATS hit rates are 0.4992 (ridge) and 0.4759 (gbm) — both near a coin flip and well below
the 0.524 break-even, nowhere close to the ~0.56 leak-suspicion threshold. `ats_by_threshold`
shows no meaningful improvement as the model's disagreement with the line grows (ridge
climbs to 0.550 only at the n=129 edge>=6 bucket, gbm to 0.527 at n=273 — both plausibly
noise at that sample size, not a real signal, especially since the trend across smaller
thresholds is flat-to-down first).

`market_comparison_regression` is the decisive test and it's unambiguous: `market_coef` is
~1.07-1.10 (essentially "the market's line is the actual margin, linearly, coefficient
near 1" as expected), while `model_coef` is -0.0102 (ridge) and -0.0535 (gbm) — both
indistinguishable from zero and, if anything, slightly negative. **The model adds nothing
over the closing line, honestly reported.**

None of this reads as a leak. It reads as a market-blind model built from EPA/NGS team
ratings performing about as well as such a model should against the most efficient line in
US sports: competitive but not competitive with the market itself.

**Ridge wins.** It has lower margin MAE (10.267 vs 10.671), lower total MAE (10.691 vs
11.061), an ATS hit rate closer to 50% (gbm's 0.4759 is further below coin-flip), and a
model_coef closer to (but still on the "adds nothing" side of) zero. GBM shows no
advantage here and is arguably overfitting slightly harder to noise (its more negative
model_coef and worse ATS rate both point that way, though at n=1359 neither is a strong
claim). Recommendation for Task 11's default: **ridge**.

## Zero-filled-features evidence (the known open question)

I checked how many rows in the built 2639-game dataset have zero-filled rating/NGS
features, and whether early-season weeks look different from later ones.

**NGS side (legitimate, expected effect):** 387 of 2639 rows (14.7%) have all three NGS
diff features (`cpoe_diff`, `ryoe_diff`, `separation_diff`) exactly 0.0. These are heavily
concentrated in week 1 of each season (2016: 16 of 17 week-1 games; 2017: 15 of 16; 2018:
16 of 16 games have some zero NGS metric at week 1) and thin out fast — by week 3-4 onward
it's typically 2-3 games per week, presumably teams on a bye or with sparse trailing
history. This matches the brief's description exactly: `_trailing_ngs` groups by
`(season, team)` with no season-to-season carryover, so week 1 of *every* season for
*every* team starts from zero trailing history, not just 2016.

Separately, the `ngs_imputed_any` flag is 1 for 2345 of 2639 rows (89%) and is 100% in
2016-2017, staying at 78-93% even in 2020-2025. This looked alarming at first but turned
out to be a sticky-flag artifact, not a broken join: `_trailing_ngs` computes
`trail_imputed_any` as the **max** of every prior week's per-metric imputed flag, and
individual NGS metrics (e.g. rushing, which per the module's own docstring covers only
468 of 544 team-games in a sample season) are imputed often enough that once a team hits
one imputed metric early in a season, the flag stays 1 for every subsequent week that
season, even though the underlying `trail_cpoe`/`trail_ryoe_per_att`/`trail_separation`
values themselves are real (or league-mean-imputed, not zero) for most of those rows. So
`ngs_imputed_any` is a much noisier/broader signal than "this row has no real trailing
data" — worth knowing if it's ever used to filter or weight rows.

**Rating side (a separate, more concerning finding — not just a "no history yet" effect):**
78 rows have all five rating-diff features (`net_rating_diff` and the four
`off_*_edge_*` columns) exactly 0.0. These are **not** spread evenly across early weeks of
the whole dataset — they are concentrated entirely in games involving Oakland (`OAK`,
2016-2019, i.e. every season before their move to Las Vegas) and San Diego (`SD`, 2016
only, before their move to LA), spanning every week of those seasons, not just week 1
(e.g. `2016_17_KC_SD`, week 17). I traced this to a team-abbreviation mismatch between
data sources: `nflreadpy`'s play-by-play normalizes historical franchise codes to their
*current* abbreviation (`posteam`/`defteam` show `LAC`/`LV` even for 2016-2019 rows),
while `nflreadpy`'s schedules keep the *historical* labels for those seasons (`SD`, `OAK`
appear literally in `home_team`/`away_team`). Since `ratings_by_week` is built from pbp
(labeled `LAC`/`LV`) and joined onto schedules (labeled `SD`/`OAK`) by team code in
`build_game_features`, every OAK/SD game in those years misses the join, becomes NaN, and
is then silently zero-filled by the blanket `fillna(0.0)` the brief already flagged —
except this isn't the "genuinely no prior data yet" case the brief was asking about, it's
a join-key bug that affects mid- and late-season games too. It's a small slice of the
dataset (78/2639 = 3.0%) and I did not fix it (it lives in `ratings/`/`features.py`, both
out of scope for this task), but it's a distinct, more actionable bug than the blanket
fillna question, and I'd flag it for the human's attention at final review alongside that
question.

Net read: the blanket zero-fill materially affects roughly 15% of rows via legitimate
lack-of-trailing-NGS-history (concentrated at each season's week 1, as expected), plus a
separate 3% via the OAK/SD team-code join miss (not early-season-specific). Neither looks
large enough by itself to be driving the headline MAE numbers (1359 test games in
2021-2025, where the OAK/LV, SD/LAC code collision doesn't recur since those relocations
happened before 2021), but the NGS week-1 effect does touch every test season and is worth
the human's attention if the model is ever pushed to weight early-season games more
heavily.

## Files changed

- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\backtest.py` (new)
- `C:\Users\sbuer\Documents\NFL Game Model\scripts\build_dataset.py` (new, with the
  documented one-line warm-up-season deviation from the brief)
- `C:\Users\sbuer\Documents\NFL Game Model\scripts\backtest.py` (new)
- `C:\Users\sbuer\Documents\NFL Game Model\tests\test_backtest.py` (new)
- `C:\Users\sbuer\Documents\NFL Game Model\data\processed\game_features.parquet` (built
  artifact, not committed — see `.gitignore` status below)
- `C:\Users\sbuer\Documents\NFL Game Model\data\raw\pbp_2015-2025.parquet`,
  `schedules_all.parquet`, `ngs_*_2016-2025.parquet` (cached raw pulls, not committed)

## Self-review findings

- Test file, `backtest.py`, and both scripts match the brief's code verbatim except: (1)
  the documented warm-up-season fix in `build_dataset.py`, required to make Step 7 runnable
  at all with the brief's own default parameters, and (2) an unused `pytest` import in the
  test file and five redundant `int(len(...))` casts in `backtest.py`/`ats_by_threshold`
  that `ruff check --fix` removed — pure style, no behavior change (`len()` already
  returns `int`).
- Confirmed test output is pristine: `60 passed in 8.99s`, no warnings, on both the
  focused file and the full suite.
- `ruff format --check` and `ruff check` both clean on all four files I created/edited.
- No YAGNI creep: I did not add anything beyond the brief's four functions and two
  scripts; `ats_by_threshold` was in the brief's own Step 3 code and is used by
  `scripts/backtest.py`, so it's in scope.
- Verified `walk_forward`'s per-season loop and `evaluate`'s push-exclusion logic against
  the synthetic tests and against realized behavior; nothing subtle needed correcting.

## Issues or concerns

1. **Deviation from the brief's `build_dataset.py`** (described above) was necessary for
   Step 7 to run at all — flagging prominently since I was told to use the brief's code
   verbatim, but the brief's own default parameters can't work as literally written
   against a hard-failing `build_ratings`. The fix only affects which pbp seasons feed
   rating history; nothing about `walk_forward`/`evaluate`/`market_comparison_regression`
   or the two files matching the brief verbatim.
2. **OAK/SD team-code join bug** (described above): 78 games (3.0% of the dataset) get
   fully zero-filled rating features due to a historical-vs-current team-abbreviation
   mismatch between pbp and schedules, not due to legitimate lack of trailing history.
   This is separate from, and more concrete than, the brief's "known open question" about
   the blanket fillna — worth the human's attention at final review, but out of scope for
   me to fix (touches `ratings/`/`features.py`).
3. Backtest numbers read as honest/non-leaked per the "how to read the results" section:
   model MAE for both estimators is slightly *worse* than the market's on the same games,
   ATS hit rates sit near a coin flip and below break-even, and `model_coef` in the
   decisive regression is near zero for both estimators. No leak-audit action taken
   because none of the leak triggers (MAE far below market's, ATS hit rate above ~0.56)
   fired.

## Fix pass

Code review on this task raised four Important findings (I1-I4) plus three Minor
cleanups. All were fixed in `src/nfl_game/backtest.py` and `tests/test_backtest.py` only
— `model/features.py` and `ratings/` were not touched, per instructions.

### I1 — `evaluate` mishandles games with no posted line

**Bug:** the only filter was `preds["margin"].notna()`. `spread_line`/`total_line` were
never checked, so (a) `margin_mae` and `market_margin_mae` could be computed over
different denominators, and (b) the push filter `d["margin"] != d["spread_line"]` kept
NaN-line rows (`NaN != NaN` is `True` in pandas), and since `NaN > x` is `False` on both
sides of the ATS comparison, such a row scored as a spurious hit.

**Fix:** added a single `_valid_games()` helper that requires all six of
`margin, total_points, spread_line, total_line, model_margin, model_total` to be
non-null, and switched `evaluate`, `market_comparison_regression`, and
`ats_by_threshold` to build their working frame `d` from it, so model and market are
always measured on the identical game set in every function that's reachable from
`walk_forward`'s output.

**TDD:** wrote `test_evaluate_excludes_games_with_no_posted_line` first (3-row frame, one
row with `spread_line=NaN`). Confirmed RED against the old code:
```
>       assert m["n_games"] == 2
E       assert 3 == 2
```
Applied the fix; test now passes (`n_games=2`, `ats_n=2`, `ats_hit_rate=0.5`), matching
the finding's own reproduction numbers exactly.

### I2 — the leak test couldn't detect a leak

**Bug:** `test_walk_forward_never_trains_on_the_test_season` compared MAE of the honest
walk-forward against a baseline fit on the test season alone. Mutation testing (per the
finding) showed a `walk_forward` that trains on the test season *plus* all prior seasons
still passes, because the leaked baseline (100 rows, in-sample) is always tighter than
even a leaky 300-row model.

**Fix:** replaced the test (keeping its name) with an exact prediction-equality check:
`walk_forward`'s output for the test season must be byte-for-byte identical
(`np.testing.assert_allclose`) to a `GameModel` fit directly on
`features_df[features_df["season"] < test_season]` and applied to that season's rows.
Rewrote the docstring to explain why the MAE-comparison version was toothless and what
the new assertion actually checks.

**Mutation evidence (required verification):**
- Temporarily changed `walk_forward`'s train filter from `season < season` to
  `season <= season`. Re-ran the new test alone: **FAILED**, 100/100 elements mismatched
  (e.g. `-0.678771` vs `-0.115956` for the first prediction) — confirms the test
  discriminates the leak.
- Reverted the mutation. Re-ran the new test alone: **PASSED**.

### I3 — nothing pinned the ATS/O-U sign convention

Added `test_evaluate_ats_hit_rate_pins_sign_convention` and
`test_evaluate_ou_hit_rate_pins_sign_convention`. Each has an all-correct sub-case
(asserts `hit_rate == 1.0`) and a 3-game mixed sub-case with an exact non-boundary
expected rate (`2/3`) chosen so that flipping the pick comparison (`>` to `<`) inverts
the result to `1/3` rather than leaving it inside the previously-tautological `[0, 1]`
range.

### I4 — the market benchmark was untested

Added `test_evaluate_market_mae_matches_hand_computed_values`: a 3-row hand-built frame
with known `spread_line`/`margin` and `total_line`/`total_points` pairs, asserting exact
`market_margin_mae == 10/3` and `market_total_mae == 4.0`. This fails if
`market_margin_mae` is negated, swapped, or hard-coded.

### Minor cleanups

- Added `test_ats_by_threshold_buckets_by_edge_size`: a 4-game hand-built frame pinning
  bucket membership (`n`) and `hit_rate` at two thresholds.
- `walk_forward`'s `test.merge(preds, on="game_id", how="left")` now passes
  `validate="one_to_one"`, so a duplicated `game_id` raises instead of silently fanning
  out rows.
- Removed the unnecessary `.copy()` in `evaluate` (`d` is only read there, never
  mutated); kept `.copy()` in `ats_by_threshold`, which does assign a new `edge` column.

### Test commands run

```
.\.venv\Scripts\python.exe -m pytest -q
```
```
.................................................................        [100%]
65 passed in 8.35s
```
(60 pre-existing + 5 new: I1's NaN-line test, I2's replacement, I3's ATS and O/U
sign-convention tests, I4's market-MAE test, and the ats_by_threshold coverage test —
net +5 since I2 replaced rather than added a test.)

```
.\.venv\Scripts\python.exe -m ruff format --check src/nfl_game/backtest.py tests/test_backtest.py
```
```
2 files already formatted
```

```
.\.venv\Scripts\python.exe -m ruff check src/nfl_game/backtest.py tests/test_backtest.py
```
```
All checks passed!
```

### Backtest re-run (I1 should be a no-op on this dataset — confirmed)

```
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
```
```
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

Matches the expected values exactly (n=1359, margin MAE 10.267/market 9.752, ATS 0.4992
n=1326, model_coef -0.0102). Confirms the built parquet has no NaN lines in the
2021-2025 test range, so I1's fix changes nothing here — as expected, since the dataset
was already known to be free of unplayed/no-line games in that range.

### Files changed

- `src/nfl_game/backtest.py` — `_valid_games()` helper; `evaluate`,
  `market_comparison_regression`, `ats_by_threshold` now derive their working frame from
  it; `validate="one_to_one"` on the `walk_forward` merge; removed unnecessary `.copy()`
  in `evaluate`.
- `tests/test_backtest.py` — added `test_evaluate_excludes_games_with_no_posted_line`,
  replaced `test_walk_forward_never_trains_on_the_test_season`, added
  `test_evaluate_ats_hit_rate_pins_sign_convention`,
  `test_evaluate_ou_hit_rate_pins_sign_convention`,
  `test_evaluate_market_mae_matches_hand_computed_values`,
  `test_ats_by_threshold_buckets_by_edge_size`.
