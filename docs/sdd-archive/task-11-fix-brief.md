# Task 11 fix pass — re-review brief

Range under review: `44fb9c1..aa6bf4e` (single commit `aa6bf4e`).
Review package: `.superpowers/sdd/review-44fb9c1..aa6bf4e.diff`

`aa6bf4e` is a review-fix pass responding to the findings below. It was committed
**unreviewed** (see Process note at the end). Your job: verify each finding is
actually fixed, in the code, by mutation — not by trusting the account in
"What aa6bf4e claims to have done".

---

## PART 1 — The original review findings that aa6bf4e was supposed to fix

### I3 (Important) — degenerate-feature guard in `GameModel.fit`. HUMAN-APPROVED APPROACH.

Background: `RobustStandardScaler`'s floor only engages below `10*eps ~ 2.22e-15`, so it is
a machine-epsilon guard, not a degeneracy guard. A milder case was still live: the 2018
calibration fold (trained on 2016+2017) has `ryoe_diff` std 1.05e-2 — far above the floor —
but only **5 distinct values across 512 rows**. Its predictions run -209.8..+183.1 where
every other fold maxes at |18.8|. Those 256 rows are 13.8% of the 1855-row calibration
sample and measurably flatten the calibrator:

    folds 2018-2024 (as shipped): n=1855, cover coef 0.00342, slate range 46.86-47.80%
    folds 2019-2024:              n=1599, cover coef 0.01023, slate range 46.36-49.18%

Dropping the one degenerate fold TRIPLES the slope and the probability spread.

APPROVED APPROACH: a distinct-value criterion in `GameModel.fit` (catches both the 1e-17 and
the 1e-2 versions; the eps floor catches only the first). Must **RAISE, not warn** — a warning
still lets poisoned predictions reach the calibrator. Must not fire on healthy folds.
Then make `walk_forward` **SKIP** a fold whose training slice is degenerate, as it already
skips a season with no prior data.

THEN reconsider `scripts/slate.py`'s `prior_seasons[2:]`: it was defence-in-depth added
before the guard existed, and the reviewer found it removed the SAFE 2017 fold (post-fix:
max |model_margin| 14.2) while keeping the DAMAGED 2018 one. With the guard working by
inspection instead of a magic index, `[1:]` or `[:]` is likely cleaner. Keep the load-bearing
property: **the calibrator is fit on OUT-OF-SAMPLE `walk_forward` predictions from prior
seasons only, never in-sample.**

Note: the true root cause is `ryoe_diff` being imputed to one shared value across a season,
which lives in `features.py`/`ratings/` — those are RESERVED, out of scope. The guard is the
agreed in-scope mitigation.

### I2 (Important) — exclude exact pushes from Calibrator training. HUMAN-DECIDED.

`calibrate.py` uses `covered = (d["margin"] > d["spread_line"])`, so an exact push trains as
"did not cover". `backtest.evaluate` does the OPPOSITE — it filters
`d[d["margin"] != d["spread_line"]]` and documents "Exact pushes are excluded". The repo held
two contradictory definitions of one concept, so `ats_hit_rate` and `cover_prob` estimated
different quantities under the same name. Measured on the real calibration sample (n=1855,
2018-2024): 47.17% strict-cover, 2.75% exact pushes, 48.50% cover among decided games;
intercept -0.1138 → 47.16% at zero edge. A push returns the stake; it is not a loss.

Fix BOTH targets (spread and total) to match `evaluate`. Add a test that a push-heavy sample
does not depress the intercept. Expected effect: slate cover_prob 46.9-47.8% → 48.2-49.2%.

**This deviates from the Task 10 brief's verbatim code. The human approved the deviation**,
on the grounds that the SAME plan's `evaluate()` specifies the opposite. Do not re-flag the
deviation itself as a finding; DO flag it if the two definitions still disagree.

### I1 (Important) — CLAUDE.md is wrong and incomplete.

CLAUDE.md:48-64 carried a "Known issue" section saying the ryoe_diff/StandardScaler bug is
UNFIXED, that slate.py uses `prior_seasons[1:]`, and that fixing it needs "either a variance
floor in the ridge StandardScaler step or an as-of ratings fix in ratings/, both reserved for
human review". Commit `44fb9c1` added exactly that variance floor and changed the slice. Every
claim was false and would send a future session re-diagnosing a fixed bug. DELETE it; replace
with a short accurate note on `RobustStandardScaler` + whatever I3's guard became. Also:
 - Commands section listed only pip/pytest/ruff and never mentioned `scripts/`, though
   `build_dataset.py`, `backtest.py` and `slate.py` are the project's entire user interface.
 - `market/compare.py` got no architecture bullet though the other three layers do.
 - "Market margin MAE is around 9.8-10.3" — measured is 9.752, just under the stated floor.

### TEST GAPS (Important) — these mutants SURVIVED `tests/test_compare.py` before the fix

 (a) `model_spread` / `market_spread` SWAPPED — inverts every displayed pick. Nothing asserted
     either column's value; `test_gap_is_model_minus_market` only pins `spread_gap`, which is
     computed independently.
 (b) `cover_prob` / `over_prob` SWAPPED — same class.
 (c) sort by SIGNED gap instead of abs — both fixture gaps were positive, so invisible.
 (d) `edge_flag` uses `>` instead of `>=` — no fixture gap equalled the threshold; brief
     specifies `>=`.
 (e) `edge_flag` driven by `total_gap` instead of `spread_gap` — same flags on that fixture.
 (f) markdown flips home/away in the Game column — the test only checked substring presence,
     so "BUF @ KC" passed as readily as "KC @ BUF".

Suggested starting point given to the fixer:

    row = build_slate(*_inputs()).set_index("game_id").loc["2026_01_KC_BUF"]
    assert (row["model_spread"], row["market_spread"]) == (6.0, 2.5)
    assert (row["cover_prob"], row["over_prob"]) == (0.58, 0.55)

plus a fixture case with a NEGATIVE gap (kills c) and one with a gap EXACTLY at the threshold
(kills d). Each new test must be verified to discriminate by applying the mutation.
`test_sorted_by_absolute_edge` was to be left untouched (it is brief-mandated and
non-discriminating; `test_sort_order_independent_of_input_row_order` beside it DOES
discriminate).

### MINOR (fix while in these files)

 - `slate_markdown`'s "n/a" guard is HALF-APPLIED: a row with a missing line still rendered
   `| MIA @ NYJ | -1.5 | -1.0 | -0.5 | 49.0% | 43.2 | nan | +nan | n/a |` — `market_spread`,
   `spread_gap`, `market_total`, `total_gap` all showed nan/+nan. The rationale that "nan%"
   reads as a data-quality bug applies identically to "+nan". The existing test only asserts
   `"nan%" not in md`, so it passed on that output.
 - `slate.py --season 2018` died with a raw sklearn traceback (`ValueError: Found array with
   0 sample(s)`) when the calibration window is empty. The script already models the right
   behaviour nearby with `raise SystemExit(f"no games found for ...")`.
 - `build_slate`'s merge has no `validate=`, unlike `walk_forward`'s `validate="one_to_one"`.
 - `total_gap` is computed from the ROUNDED `model_total` (the round overwrites the column
   before the gap is taken) while `spread_gap` uses the unrounded `model_margin`.

---

## PART 2 — What aa6bf4e CLAIMS to have done (the fixer's own account — UNVERIFIED)

> I3: added `MIN_DISTINCT_VALUES=10`, `DegenerateFeatureError`, and `_degenerate_features()`
> to `predict.py`; `GameModel.fit` raises when a non-binary FEATURE_COLS column has <10
> distinct values in the training slice. Binary 0/1 flags (`is_dome`, `div_game`,
> `ngs_imputed_any`) are deliberately skipped. Threshold justified as: `rest_diff`, the
> smallest healthy non-flag feature, never drops below 15 distinct values in any fold, while
> the two poisoned folds sit at 3 (2016 alone) and 5 (2016+2017). `walk_forward` skips
> degenerate folds.
> I2: `Calibrator.fit` now filters `margin != spread_line` and `total_points != total_line`,
> matching `evaluate`. I1: CLAUDE.md corrected. Plus the six test-gap mutants and the four
> Minor items.
> OBSERVED EFFECT on the real slate (2025 wk1): probabilities moved from the flat 0.500 bug,
> through the shipped 46.9-47.8%, to **47.7-50.4%** — now straddling 50% (ARI @ NO reads
> 50.4% on a +7.5 gap). Test count 88 -> 100.

Explicitly NOT independently confirmed by anyone: the threshold-10 justification, that the
guard never fires on a healthy fold, and whether `slate.py`'s `prior_seasons` slice was
revisited as instructed.

---

## PART 3 — Verification the fix pass was required to pass

 - full suite (100 tests now) + `-W error::FutureWarning -W error::DeprecationWarning`
 - `ruff check .` and `ruff format --check` (pre-existing `tests/test_epa.py` I001 is known
   and deliberately left; do not flag it)
 - **TASK 9 BACKTEST MUST BE UNCHANGED** — the critical regression check, since the guard must
   be a no-op on healthy 5+-season slices:
   `scripts\backtest.py --test-seasons 2021-2025` → n=1359, margin MAE 10.267, market 9.752,
   ATS 0.4992 (n=1326), model_coef -0.0102. If any of these moved, that is a Critical finding.
   **DO NOT REBUILD the parquet** — it exists and takes many minutes.
 - `scripts\slate.py --season 2025 --week 1` runs and produces a sane slate.

---

## PART 4 — Branch context and standing constraints

**The headline result — do not "improve" it.** Task 9 backtest, test seasons 2021-2025,
n=1359: market margin MAE 9.752; ridge 10.267 with ATS 0.4992 and `model_coef` -0.0102.
The model roughly matches the closing line WITHOUT beating it. **That is the success
condition**, not a failure. Reading the backtest is inverted from the usual instinct: a model
MAE well *below* the market's, or an ATS hit rate above ~0.56, is to be treated as a data leak
to audit — not an edge. NFL closing lines are the most efficient market in US sports.
If you find something that makes the model look better, suspect a leak first.

**Ridge, not gbm**, is the estimator choice — Task 9's evidence (gbm: 10.671 / 0.4759 /
-0.0535, worse on every metric).

**`features.py` and `ratings/` are RESERVED** — out of scope for this fix pass. A finding
whose only correct fix lives there should be reported as out-of-scope context, not as a
defect in this diff.

**Mutation-test; do not trust a passing suite.** This branch's established practice, and it
has repeatedly paid: on Task 9 a plan-mandated leak test could not detect a leak (a
`walk_forward` training on the test season plus prior seasons passed all 7 tests). On Task 10
EVERY mutation isolated to `over_prob` survived all 7 tests — half the calibrator could have
shipped broken with a fully green suite. Run each mutation against the target test **in
isolation**, so no sibling test can be the one catching it. A test that passes is not
evidence; a test whose mutant dies is.

**Also beware tests that pass for incidental reasons.** On Task 10 a fixer's first attempt at
a new test did not discriminate — synthetic well-behaved data coincidentally satisfied the
assertion even when mutated. Check that new tests fail for the right reason.

**PROCESS NOTE — why this re-review exists.** The dispatch that produced `aa6bf4e` was
reported back to the controller as REJECTED by the user, so the controller recorded "fix pass
never ran, no work exists". That was wrong: the subagent had in fact run, leaving 428
uncommitted lines across 10 files, caught only by a routine `git status`. The recovered work
was verified (100 tests pass; Task 9 backtest byte-identical) and committed as `aa6bf4e`,
explicitly labelled unreviewed in its commit message. It has had **no review of any kind**.
