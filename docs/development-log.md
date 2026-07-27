# Development log

**This is a historical record, not documentation. `CLAUDE.md` is the only authoritative
description of how this project currently works.**

What follows is the verbatim working ledger kept while the model was built, across roughly
a dozen tasks, several pauses, three review passes and a final whole-branch review. It was
written as the work happened — decisions, measurements, dead ends, and corrections in the
order they occurred. It is preserved because the *reasoning* behind the design is not
recoverable from the code or from git history, and because the project's most useful
lessons are the mistakes it records rather than the code it produced.

## How to read it

**Treat every claim in here as evidence, not as authority.** The log is a contemporaneous
account, and parts of it were later proven wrong by the very reviews it describes. Two
claims in particular were explicitly falsified:

1. *"The 2021-2025 backtest window contains zero rows affected by the team-code bug."*
   False. That statement was scoped to the only team-code bug known at the time; the
   separate, then-unrecorded `LA`/`LAR` mismatch hit every Rams game in all ten seasons,
   85 of them inside the backtest window.
2. *"The blanket `fillna(0.0)` still silently zero-fills legitimate week-1 NGS gaps."*
   False on both halves. That line was a dead no-op (0 NaN of 2639 rows); the week-1 fills
   happen about sixty lines earlier, and all 159 of them carry `ngs_imputed_any == 1`, so
   they were never silent.

Both were caught only because someone re-derived the claim instead of trusting the record.
Anything here that reads like a settled fact — especially a "this is not affected" or "this
is already covered" — deserves the same treatment before you rely on it.

Sections are append-only and roughly chronological, so **later entries supersede earlier
ones**. A "STILL OPEN" or "NEXT:" list is only accurate as of the entry that contains it;
the deferred-Minors list that appears partway through was cleared in full on 2026-07-27,
and the entry at the end of the file records that.

## Provenance

Frozen from `.superpowers/sdd/progress.md` at commit `352b1a3` (2026-07-27), unedited
below this header. The file was gitignored for its whole life by `.superpowers/sdd/.gitignore`,
so it existed on a single machine and any `git clean -fdx` would have destroyed it.

Supporting material:

- **Per-task briefs and reports** — archived alongside this file in `docs/sdd-archive/`.
- **Review diffs** — deliberately *not* archived. They were plain `git diff <a>..<b>`
  output over commits that are all still reachable in this repository's history, so they
  are regenerable on demand; e.g. the final whole-branch review diff is
  `git diff c00f5df..130e7a3`. See `docs/sdd-archive/README.md` for the full list.

The headline result the log builds toward, and the reason "no edge" is the success case, is
summarised in `CLAUDE.md` under "Reading the backtest". Read that first.

---

Task 1: complete (commits c00f5df..bad39fd, review clean)
Task 2: complete (commits bad39fd..4dd8bf4, review clean; loaders verified vs live data)
Task 3: complete (commit 4dd8bf4..1d92962, review clean; sanity-checked (544,12), BAL/BUF/DET top offenses)
Task 4: complete (commits 1d92962..b3b5108, review clean after 1 fix loop; sign convention verified on 2024 data, MIN/PHI/DEN/GB/DET top defenses). Fix b3b5108 filters sample weights to the non-null-target mask (human approved fixing the plan-mandated latent bug).

Task 5: complete (commit b3b5108..4df1e98, review clean; leak guard verified mechanically — strict week< cutoff, double-enforced via zero-weight + row drop). Implementer removed one unused numpy import from the brief's test file (confirmed no-op).

Task 6: complete (commit 4df1e98..a6efaeb, review clean; coverage verified on 2024 — 544 rows, cpoe_imputed=0.009, ryoe_per_att_imputed=0.140 matching brief; filter-before-aggregate drops week-0/POST rows; flag-then-fill imputation correct).

Task 7: complete (commits a6efaeb..c1af473 [incl. 2bf6cba impl + c1af473 ruff-format fix], review clean; both central properties verified — trailing-NGS leak guard uses strictly-prior weeks, future/unplayed games kept with null targets + full features). Added test_target_cols_fixed. Line-length nit fixed via ruff format.

Task 8: complete (commit c1af473..f8bd876; executed INLINE, not via subagents — session
config disallowed spawning agents. Pure transcription of the brief's reference code,
verbatim. 9 new tests pass, 51 total. One ruff-format line-length fix applied to
tests/test_predict.py before commit, same nit as Task 7.)

RESOLVED (commit 5ddf154, pre-Task-9): predict.py ridge had NO feature scaling. FEATURE_COLS
mixes scales (temp_outdoor ~60, wind ~0-20, 0/1 flags, EPA rating diffs ~0.1), and ridge's L2
penalty is scale-sensitive, so alpha=1.0 shrank the small-scale rating features carrying the
signal while barely touching temperature — handicapping ridge against the scale-invariant GBM
and biasing Task 9's estimator choice. Synthetic tests missed it because _train() drew every
feature from N(0,1). Fixed by make_pipeline(StandardScaler(), Ridge(alpha)) in ESTIMATORS.
Two regression tests added: unit-invariance (rescaling temp_outdoor x100 must not move
predictions) and small-scale-signal recovery (MAE was 1.92 vs a 0.4 noise floor before the fix).
Human approved the deviation from the plan's reference code.

Task 9: complete (commits 5ddf154..814fdbd — 2748b80 impl + 814fdbd fix pass; review CLEAN
after 1 fix loop: spec compliance OK, code quality Approved, no Critical, no Important).
Executed subagent-driven per human choice: sonnet implementer, opus reviewer, sonnet fixer.
65 tests pass. Re-reviewer verified every fix by MUTATION rather than by report, running each
mutation against the target test in isolation so no sibling test could be the one catching it.

*** TASK 9 BACKTEST RESULT — the branch's headline finding ***
Test seasons 2021-2025, n=1359 games. Market margin MAE 9.752.
  ridge: margin MAE 10.267, ATS 0.4992 (n=1326), model_coef -0.0102  <- WINS
  gbm:   margin MAE 10.671, ATS 0.4759, model_coef -0.0535
This is the HONEST, NON-LEAKED outcome the plan was written to detect: the model roughly
matches but does not beat the closing line, ATS is a coin flip, and model_coef ~0 means the
model adds nothing the market doesn't already contain. Per the "reading the backtest is
inverted" rule this is SUCCESS (a working pipeline), not failure. RIDGE is the estimator
choice that feeds Task 11's default. Reviewer independently verified no leak: mutation-tested
the season boundary, confirmed the merge is 1:1, reproduced the numbers byte-for-byte.

--- Task 9 review findings, all FIXED in 814fdbd ---
I1 (correctness): evaluate() filtered only on margin.notna(), so (a) margin_mae averaged over
all rows while market_margin_mae skipped NaN — different denominators, violating the plan's
"market's MAE on the same games"; (b) the push filter kept NaN-line rows since NaN != NaN,
and those rows scored as ATS *hits*. Latent on current data (no NaN lines) but fires on
Task 11's upcoming slates. Fixed via one shared valid-row mask across evaluate,
market_comparison_regression, and ats_by_threshold.
I2 (test validity, PLAN CONFLICT — human chose to replace): the plan-mandated
test_walk_forward_never_trains_on_the_test_season could not detect a leak. A walk_forward
training on the test season PLUS prior seasons passed all 7 tests (honest 2.677, leaky 2.429,
baseline 2.268 — the leaky variant still satisfied mae_honest > mae_leaked), because the
baseline was fit on 100 rows vs the model's 300. Replaced with exact prediction-equality
against a GameModel fit only on prior seasons; mutation-verified (< -> <= now fails).
I3/I4 (test coverage): the ATS/O-U sign convention and the market MAE benchmark survived
mutation — flipping picks_home to <, or replacing market_total_mae with a literal 999.0,
left the suite green. Both now pinned by exact-value assertions on hand-built frames.

*** OPEN BUG for final review: OAK/SD team-abbreviation mismatch ***
This exists NOWHERE in the repo — no code comment, no xfail, no issue file. Reviewer warned
it will be lost if only the task-9 report records it, and that report is gitignored.
Play-by-play normalizes team codes to modern LV/LAC even for historical seasons; schedules
keep the historical OAK/SD. So the build_game_features join misses and features.py's blanket
fillna(0.0) silently zero-fills ALL five rating diffs for those games: 78 of 2639 rows
(2016:30, 2017:16, 2018:16, 2019:16 — OAK 64, SD 16). ZERO rows fall in the 2021-2025 test
window, so the reported backtest is not contaminated; impact is training-side only (78/1280 =
6.1% of the pre-2021 pool). Those rows carry a biased target (mean margin +3.19 vs +1.32),
so they nudge the intercept — but the effect is model-DEGRADING, never result-inflating,
making the "adds nothing over the line" conclusion if anything slightly pessimistic.
This is a concrete instance of the OPEN DESIGN QUESTION below.

--- Note on build_dataset.py (reviewer-approved deviation from the plan) ---
The plan's build_dataset.py cannot run as written: ratings_by_week(seasons=[2016..]) calls
build_ratings(asof_season=2016, asof_week=1), no game qualifies as prior, and build_ratings
raises by design. The implementer loads ONE extra warm-up season of pbp for rating history,
still filtering final output to the requested seasons. Reviewer verified airtight: 2015 enters
only through decay_weights (strictly-prior by construction), no 2015 rating rows are produced,
and 0 rows outside 2016-2025 survive in the parquet.

--- Task 9 Minor findings deferred to final whole-branch review ---
All are coverage nits confirmed by surviving mutants; none is a live bug.
(1) ats_by_threshold and market_comparison_regression have no NaN-exclusion test of their
own — both are correct today via the shared _valid_games helper, but reverting the mask in
either one individually still passes the suite; only evaluate's use is pinned.
(2) O/U push exclusion untested — deleting `ou = d[d["total_points"] != d["total_line"]]`
survives the suite. The ATS equivalent IS pinned. Matters at n=11 real pushes in the dataset.
(3) ats_by_threshold's inclusive boundary unpinned — the test uses thresholds (0,5) against
edges of 1 and 6, so no edge sits exactly on a threshold and `>= t` -> `> t` survives.
(4) walk_forward never forwards `estimator` under test — dropping `estimator=estimator`
survives the whole suite because every test uses the default ridge, yet
`scripts/backtest.py --estimator gbm` is a supported path that produced a reported result.
(5) validate="one_to_one" is unexercised (nothing constructs a duplicate game_id) — fine as
defensive hardening.
Reviewer observation, NOT a finding, no action needed: requiring total_line non-null to score
a *margin* metric is slightly conservative (a game with a posted spread but no posted total is
dropped from margin_mae too). Deliberate trade for one identical game set across all metrics,
documented at _REQUIRED_COLS; errs toward dropping rows rather than inventing comparisons.

--- (paused here 2026-07-24 before Task 10; resumed same day) ---

Task 10: complete (commits 814fdbd..7b62ab9 — 8630478 impl + 7b62ab9 fix pass; review CLEAN
after 1 fix loop: spec compliance OK, code quality Approved, NO remaining findings at any
severity). Sonnet implementer, sonnet reviewer, sonnet fixer. 79 tests pass.
Re-reviewer independently re-ran all 8 mutations in isolation; the 4 that previously survived
now die. Also verified the predict() NaN fix against non-contiguous / out-of-order DataFrame
indices (e.g. iloc[[10,40,71,3,90]]) — the mask assignment is positional via .to_numpy() so it
holds regardless of index labels. That index-order property is NOT covered by a test; it is
correct today but unpinned. Minor, noted for final review.

The reviewer found a stark asymmetry worth remembering: every mutation to cover_prob was
caught, but EVERY mutation isolated to over_prob survived all 7 tests (sign flip on
went_over, predict_proba[:,0] instead of [:,1], and over_prob hardcoded to 0.5). Half the
calibrator could have shipped broken with a fully green suite, because the brief pinned the
cover side with two directional tests and specified no analog for totals.

--- Task 10 review findings, all FIXED in 7b62ab9 ---
I1 (test coverage): over_prob had zero directional coverage — see above. Added the two
missing directional tests (bigger-edge and zero-edge analogs); all three mutants now die.
I2 (correctness): Calibrator.predict() hard-crashed on a NaN batch. The task report claimed
a NaN edge "just produces a NaN prob for that row" — the reviewer tested it and that was
FALSE: one NaN in model_margin/spread_line/model_total/total_line raises
sklearn ValueError "Input X contains NaN" for the ENTIRE batch. The report's "verified
manually" note had only exercised fit() with NaN and predict() on clean data. This is on
Task 11's path (upcoming slates have not-yet-posted lines). Fixed to degrade per row with
independent per-target masks: a row missing total_line still gets a real cover_prob.
I3 (minor, fixed anyway): fit() gated BOTH training sets on the union of all six required
columns, so a row missing only total_line also lost its cover-training signal. Now filtered
independently per target.
I4 (minor, fixed anyway): reliability_table's mean_pred/observed swap survived, since the
brief-mandated test_reliability_table_shape checks only set(table.columns). Added a separate
non-verbatim semantics test. NOTE: the fixer's FIRST attempt at this test did not
discriminate — synthetic calibrated data coincidentally satisfied the bin-edges assertion
even when mutated — and was discarded for a deliberately-miscalibrated direct-input test.
A cautionary example: a test written against well-behaved synthetic data can pass for
incidental reasons. test_reliability_table_shape itself was left byte-identical to the brief.

--- Task 10 findings deferred to final whole-branch review ---
(1) The _REQUIRED_COLS / valid-rows mask is now DUPLICATED between backtest.py (_valid_games)
and calibrate.py. Reviewer confirmed the implementer was right not to import from backtest.py
— backtest.py already imports nfl_game.model.predict, so calibrate.py -> backtest.py would be
a reverse dependency into model/. If a THIRD consumer appears, a shared nfl_game/_validation.py
pays for itself. Revisit at final review.
(2) Pushes are trained as "did not cover": covered = margin > spread_line is strict, so an
exact push counts as a loss-ATS outcome. Brief-mandated and low-materiality; noted only.
(3) The "fit the calibrator on walk-forward, never in-sample" requirement is DOCUMENTATION
ONLY and is not enforceable in code — Calibrator.fit cannot know where preds came from.
Reviewer confirmed this is not a defect, but Task 11 must honour it by construction.

Task 11: IMPLEMENTED + FIX PASS DONE, BUT **NOT RE-REVIEWED / NOT ACCEPTED**.
Commits 7b62ab9..aa6bf4e — a2c37ef impl, 44fb9c1 scaling fix, aa6bf4e review-fix pass.
Opus review of a2c37ef..44fb9c1 returned spec OK / code quality NEEDS WORK; the findings
are listed below and aa6bf4e addresses all of them.
100 tests pass, tree clean, ruff clean except the known test_epa.py I001.

*** PROCESS INCIDENT — read before trusting the ledger's "not started" notes ***
The fix-pass dispatch was reported to the controller as REJECTED by the user, so the
controller recorded "fix pass never dispatched, no fix work exists" and began writing the
pause state on that basis. That was WRONG: the subagent had in fact run, leaving 428
uncommitted lines across 10 files. It was caught only because a routine `git status` before
pausing showed 10 modified files instead of the expected clean tree. Lesson: ALWAYS run
`git status` before declaring a pause state, and never infer tree contents from what the
tool results say happened. Had the pause gone ahead, the work would have sat uncommitted
and been destroyed by any `git clean -fdx`.
The recovered work was verified before committing (100 tests pass; Task 9 backtest
byte-identical: n=1359, margin MAE 10.267, market 9.752, ATS 0.4992, model_coef -0.0102)
and committed as aa6bf4e, explicitly labelled unreviewed in its commit message.

*** RE-REVIEW OF 44fb9c1..aa6bf4e: DONE 2026-07-25 (opus). Verdict: APPROVED WITH FIXES. ***
Tree verified clean at aa6bf4e afterwards by the controller (not just asserted by the
reviewer), no stashes. 100 tests pass, output pristine under -W error::FutureWarning
-W error::DeprecationWarning.
Spec: COMPLIANT on every Part-1 item (I3, I2, I1, test gaps a-f, all four Minors). No scope
creep — features.py and ratings/ untouched. ruff format --check fails on 6 files but NONE are
in this diff (the 9 changed .py files format clean); pre-existing, not attributable here.
The three never-confirmed claims were checked against the real parquet:
 - CLAIM 2 CONFIRMED: _degenerate_features fires ONLY on folds 2017 (ryoe_diff=3 distinct)
   and 2018 (=5); returns [] for 2019-2025. Surviving folds' max |model_margin| 13.5-18.8.
   The guard does not fire on a healthy fold.
 - CLAIM 3 CONFIRMED: slate.py:34 now passes prior_seasons whole ([:]), magic index gone,
   plus an empty-corpus SystemExit. Load-bearing OOS property HOLDS — walk_forward for 2025
   yields folds [2019..2024], n=1599, (oos.season >= 2025).any() is False.
 - CLAIM 1 PARTIALLY REFUTED: predict.py:53's comment says rest_diff "never drops below 15
   distinct values in any fold". Measured, the 2017 fold has 13, not 15. CLAUDE.md:102-104
   says 13 correctly, so the repo now carries two contradictory accounts of the one fact
   justifying the only tuning constant. Threshold itself still safe (13 > 10 > 5) and the
   2017 fold is independently rejected on ryoe_diff. Comment defect, not behavior defect.
I2 push exclusion measured on the real corpus: excludes 42 spread + 17 total pushes of 1599;
zero-edge cover_prob moved to 0.4852, inside the brief's predicted 48.2-49.2%.
16 of 19 mutations DIED against their intended test IN ISOLATION, including all six brief-named
test-gap mutants (a-f) and every I3 guard mutant. MIN_DISTINCT_VALUES is pinned from BOTH
directions (10->4 and 10->14 each kill a test).

--- Re-review findings on aa6bf4e (3 mutations SURVIVED the full 100-test suite) ---
CRITICAL: none.
IMPORTANT #1 (calibrate.py:66) — the TOTAL/over push exclusion has ZERO test coverage.
Deleting that line leaves all 100 tests passing. The brief mandated fixing BOTH targets; the
code does, but only the spread half is verified. tests/test_calibrate.py:159-215 is spread-only
by construction (its comments say so and it deliberately puts non-push noise on total_points).
THIS IS THE TASK-10 over_prob BLIND SPOT REPRODUCED VERBATIM ONE TASK LATER — same file, same
asymmetry, one task after it was called out as the lesson. Fix: mirror
test_pushes_do_not_depress_the_intercept with a total-side push-heavy sample.
MINOR #2 (compare.py:49-50) total_gap-from-unrounded fix UNTESTED — swapping the lines back
survives all 100. test_edge_flag_is_driven_by_spread_gap_not_total_gap:225 uses model_total
exactly 48.0, so the round is a no-op there. A fixture like 44.567 would discriminate.
MINOR #3 (compare.py:85-86) the market_total/total_gap half of the "n/a" fix UNTESTED —
reverting to raw format() survives all 100; the existing test only nulls spread_line.
MINOR #4 (predict.py:53) the wrong "15" comment above — make it say what CLAUDE.md:102 says.
MINOR #5 (compare.py:89-90) model_spread/model_total still raw-formatted while every other
numeric cell routes through _fmt; a NaN there renders literal nan/+nan. Defensible per _fmt's
docstring (they are not derived from a possibly-missing line) but the guard is half-applied
along a different axis than the brief named.
MINOR #6 (predict.py:81-82) `if values.empty: continue` silently EXEMPTS an all-NaN feature
column from the guard; it then reaches Ridge.fit and raises a raw sklearn error instead of the
clean DegenerateFeatureError. The most degenerate case possible is the one case exempted.
MINOR #7 (backtest.py:47-53) a degenerate fold is dropped with NO signal. Consistent with the
pre-existing silent train.empty skip, so not a regression, but scripts/backtest.py is the
project's acceptance test and can now quietly lose folds. Consider warnings.warn or a skip list.

*** HUMAN DECISION NEEDED (reviewer flagged, not a fix) ***
The guard now drops the 2017 fold that the PRIOR review measured as SAFE (max |model_margin|
14.2). The I3 brief complained that prior_seasons[2:] "removed the SAFE 2017 fold while keeping
the DAMAGED 2018 one" — post-fix the guard rejects 2017 too, on its 2016-only training slice
(ryoe_diff = 3 distinct). Net corpus is folds 2019-2024, n=1599 — exactly the configuration the
prior reviewer's own measurement preferred (cover coef 0.01023; measured 0.00980 after the push
exclusion). Outcome is right and defensible, but the "2017 was safe and got dropped" concern is
NOT resolved — it is now dropped by inspection instead of by magic index.

--- Re-review fix pass: commit 3f398c6, done INLINE by the controller 2026-07-25 ---
(Inline, not via subagent, because the session config restricts spawning agents and the user
said "resume" rather than approving another dispatch. Precedent: Task 8 was also inline.)
Fixed: IMPORTANT #1 (added test_total_pushes_do_not_depress_the_intercept — the over/under
mirror; mutation-verified IN ISOLATION, removing the filter drives zero-edge over_prob to
0.259 vs expected 0.5); MINOR #3 (added
test_markdown_handles_missing_total_line_without_rendering_nan_anywhere; mutation-verified in
isolation against a raw-format() revert); MINOR #4 (predict.py:53 comment now matches
CLAUDE.md's accurate account — 15 from the 2019 fold on, 13 in the 2017 fold which is already
rejected on ryoe_diff).
MINOR #2 DELIBERATELY NOT FIXED AND NOT TESTED — and this is a finding about the finding.
The pre- and post-fix total_gap orderings are BIT-IDENTICAL across 2,000,000 random cases for
any total_line ending in .0 or .5; they diverge only when the line carries >2 decimal places,
which no betting line does. The mutation survives because the reordering has NO BEHAVIORAL
SURFACE — not because coverage is missing. A test could only pin it by asserting on impossible
data, which is the "passes for incidental reasons" trap this branch has hit twice. Code left
in the current (more consistent) ordering. Do not re-open this as a coverage gap.
MINORS #5, #6, #7 deferred to the final whole-branch review (added to the deferred list below).
VERIFIED: 102 tests pass under -W error::FutureWarning -W error::DeprecationWarning; ruff check
and format clean on all 5 changed files; TASK 9 BACKTEST RE-RUN AND BYTE-IDENTICAL (n=1359,
margin MAE 10.267, market 9.752, ATS 0.4992 n=1326, model_coef -0.0102). Tree clean after
commit, verified by the controller.
STATUS: Task 11 is NOT yet accepted — 3f398c6 has not been re-reviewed.

NEXT ACTION: re-review 44fb9c1..3f398c6 (or aa6bf4e..3f398c6 for just the fix pass), then the
final whole-branch review. BUT the human has redirected first — see below.

*** USER-REQUESTED WORK, QUEUED 2026-07-25: fix the OAK/SD abbreviation bug for real ***
The user asked for all games to be mapped correctly, promoting the OAK/SD item from
"triage at final review" to actual work. THIS IS A BEHAVIOR CHANGE, unlike every fix so far.
CRITICAL CONSEQUENCE: the Task 9 backtest WILL MOVE. It has been the invariant safety net for
every fix on this branch ("if any move, STOP"). That net does not apply here — 78 previously
zero-filled training rows (2016-2019) gain real rating features, walk_forward trains the
2021-2025 test folds on those seasons, so predictions and every metric shift. Expected small.
Record the NEW baseline once it lands and re-point the invariant at it.
READ THE INVERTED-BACKTEST RULE BEFORE JUDGING THE RESULT: if the fix makes the model look
markedly BETTER (margin MAE well below ~9.75, or ATS above ~0.56), treat it as a leak to audit,
not as a win. Expected direction is a small improvement at most — the bug is model-degrading.
Scope note: the user said ALL games, so audit EVERY team-code mismatch, not just OAK and SD
(STL->LA/LAR in 2016 and WAS/WSH are the obvious other candidates), before choosing a fix site.

*** TEAM-CODE FIX: DONE, commit 130e7a3 (inline, TDD). NOT YET REVIEWED. ***
The audit found a THIRD mismatch nobody had recorded, and it is the biggest one:
  OAK -> LV   (schedules vs pbp/NGS)  78 games, 2016-2019   [previously known]
  SD  -> LAC  (schedules vs pbp/NGS)  included in the 78    [previously known]
  LA  -> LAR  (schedules AND pbp vs NGS)  165 games -- EVERY RAMS GAME, ALL TEN SEASONS,
              85 OF THEM INSIDE THE 2021-2025 BACKTEST WINDOW.   [NEVER RECORDED]
*** THIS FALSIFIES A STANDING LEDGER CLAIM. *** The Task 9 entry above says the OAK/SD bug
puts "ZERO rows in the 2021-2025 test window, so the reported backtest is not contaminated."
That is true of OAK/SD and FALSE of the dataset as a whole: 85 of the 1359 backtest games ran
with all three NGS features zeroed. The test window was never clean. Do not re-cite the old
"uncontaminated" claim; it was scoped to the only bug then known.
Discriminating evidence it was a failed join and not imputation: all-zero NGS diffs outside
week 1 were 0 of 2252 among unaffected teams, and 155 of 155 for the Rams.
Direction: FEATURE-DESTROYING, never target-leaking. Targets and lines are bit-identical
before/after. So it could only ever have made the model look worse -- the "no edge" finding
was safe, if slightly pessimistic.
FIX SITE: new `src/nfl_game/data/teams.py` (CANONICAL_TEAMS, TEAM_CODE_MAP,
normalize_team_codes) applied by all three loaders in `data/nfl.py` at ingestion. Deliberately
OUTSIDE the reserved features.py/ratings/. It RAISES on an unrecognised code rather than
passing it through -- silent pass-through is exactly how LA/LAR survived. 9 new tests, TDD
(verified red before green). CLAUDE.md architecture section updated.
EFFECT ON THE DATA: all-zero rating features 78 -> 0. All-zero NGS diffs 387 -> 159, and all
159 remaining are week-1 rows, which legitimately have no prior data (non-week-1: 0).
Row count, game_id set, targets and betting lines all unchanged.

*** NEW BACKTEST BASELINE -- REPLACES THE OLD INVARIANT ***
Every prior fix on this branch was checked against "backtest must be UNCHANGED (10.267 /
9.752 / 0.4992 / -0.0102); if any move, STOP". That invariant is now RETIRED -- this commit
is the first deliberate behaviour change. New values, test seasons 2021-2025, n=1359:
    margin MAE 10.274   (was 10.267)    market margin MAE 9.752  (unchanged, as it must be)
    total  MAE 10.684   (was 10.691)    market total  MAE 10.309 (unchanged)
    ATS 0.4977 (n=1326) (was 0.4992)    O/U 0.5022 (unchanged)
    model_coef -0.0218  (was -0.0102)   r2 0.2083
USE THESE for the next regression check.
INTERPRETATION, and it matters: repairing a real corruption across 243 games moved the model
ESSENTIALLY NOT AT ALL -- a hair worse on margin, a hair better on totals, all noise-level.
Nothing got suspiciously better, so per the inverted-reading rule there is NO leak signal.
The honest reading is that the NGS features carry so little margin signal that fixing them
changes nothing, which REINFORCES the "adds nothing over the closing line" conclusion rather
than threatening it. Do not treat the tiny MAE regression as a bug to chase.
slate.py --season 2025 --week 1 re-run clean; LAR/LV/LAC now appear in the output;
cover_prob range 48.0-49.7%, everything near 50% with few edges (honest for a market-tying
model). 111 tests pass; ruff clean on all changed files.
PARTIALLY CLOSES the OPEN DESIGN QUESTION below: the team-code INSTANCE of the blanket
fillna(0.0) masking is fixed, but the blanket fillna itself REMAINS and still silently
zero-fills legitimate week-1 NGS gaps. The final review must still decide on a NaN-guard.
A pre-fix copy of the dataset is kept at
`data/processed/game_features_PRE_teamcode_fix.parquet` (data/ is gitignored but persists;
it is NOT in the session scratchpad, which gets cleaned) for before/after comparison.
NOTE ON REBUILDING: scripts/build_dataset.py re-DOWNLOADS ~200MB of pbp via nflreadpy; the
loaders never read data/raw back. All raw parquets ARE cached in data/raw, so a rebuild that
skips the download is ~40 lines: read the cached parquets, apply normalize_team_codes with the
same column lists as data/nfl.py, then team_game_epa -> ratings_by_week -> team_week_ngs ->
build_game_features, filtered to 2016-2025. That is how the current dataset was produced.

--- PAUSED 2026-07-25 (third pause; user asked to exit) at HEAD 130e7a3 ---
Tree CLEAN, no stashes, branch feat/game-model, 20 commits ahead of master, NOTHING PUSHED.
111 tests pass. Verified directly with git status/stash at pause time, not inferred.
NEXT ACTION: nothing on this branch has been reviewed since aa6bf4e. TWO UNREVIEWED COMMITS
now stand: 3f398c6 (Task 11 fix pass) and 130e7a3 (team-code fix). The second is a
data-correctness change on the ingestion path for every downstream consumer -- do not merge it
unreviewed. Options: one review of aa6bf4e..HEAD covering both, or fold them into the final
whole-branch review. Then superpowers:finishing-a-development-branch.
Task 11 is STILL NOT FORMALLY ACCEPTED (its fix pass 3f398c6 is unreviewed), though the
aa6bf4e re-review returned Approved-with-fixes and those fixes are what 3f398c6 contains.
The final whole-branch review must still triage every deferred Minor in this ledger, the
still-open blanket fillna(0.0) design question, and re-review Minors #5/#6/#7 from the Task 11
re-review. Session config disallows spawning subagents unless the user asks -- confirm on resume.

--- What aa6bf4e did (UNVERIFIED — this is the fix subagent's own account) ---
I3: added MIN_DISTINCT_VALUES=10, DegenerateFeatureError, and _degenerate_features() to
predict.py; GameModel.fit raises when a non-binary FEATURE_COLS column has <10 distinct
values in the training slice. Binary 0/1 flags (is_dome, div_game, ngs_imputed_any) are
deliberately skipped. Threshold justified as: rest_diff, the smallest healthy non-flag
feature, never drops below 15 distinct values in any fold, while the two poisoned folds sit
at 3 (2016 alone) and 5 (2016+2017). walk_forward skips degenerate folds.
I2: Calibrator.fit now filters `margin != spread_line` and `total_points != total_line`,
matching evaluate. I1: CLAUDE.md corrected. Plus the six test-gap mutants and the four
Minor items.
OBSERVED EFFECT on the real slate (2025 wk1): probabilities moved from the flat 0.500 bug,
through the shipped 46.9-47.8%, to **47.7-50.4%** — now straddling 50% (ARI @ NO reads
50.4% on a +7.5 gap). Still near 50% with few edges, which is correct for a model that ties
the market. Test count 88 -> 100.
NOT independently confirmed: the threshold-10 justification, that the guard never fires on
a healthy fold, and whether slate.py's prior_seasons slice was revisited as instructed.

*** A CRITICAL BUG WAS FOUND AND PARTLY FIXED DURING TASK 11 — read this first ***
The first real slate run returned EVERY cover_prob and over_prob at exactly 0.500 with
zero variance. Root cause, independently reproduced: `ryoe_diff` is near-constant in an
early-season training slice (sparse NGS rushing data → nearly every row gets the same
imputed value). In the 2016-only slice its std is 1.17e-17 around a mean near zero.
StandardScaler's constant-feature guard (_is_constant_feature) tests variance RELATIVE TO
THE MEAN, so a near-zero-mean feature slips through with scale_=1.17e-17 instead of being
floored to 1.0. (ngs_imputed_any, std exactly 0.0, IS caught correctly.) Dividing by that
produced 2017 predictions of ±1.2e15, which dragged Calibrator's LogisticRegression
coefficients to ~0 → flat 0.5 everywhere.
This was a REGRESSION FROM COMMIT 5ddf154 (the StandardScaler that fixed ridge's
scale-sensitivity). Do NOT revert 5ddf154 — the scale fix is still needed and correct.
Fixed in 44fb9c1 by RobustStandardScaler(StandardScaler), which re-applies sklearn's own
absolute `scale < 10*eps` fallback after every fit — a fallback _handle_zeros_in_scale
contains but StandardScaler never reaches, because it always passes a mean-relative
constant_mask. Reviewer verified the subclass across fit/partial_fit/sample_weight/
with_std=False/sparse/clone/pickle, and confirmed no legitimate feature is clobbered
(nothing sits between 1e-14 and 6.7e-2 in any fold; EPA diffs live at ~0.1).
Task 9's backtest numbers are UNCHANGED by this fix, verified twice.

--- TASK 11 review findings: the fix pass IS DONE (committed as aa6bf4e), NOT re-reviewed ---
STALE HEADER CORRECTED 2026-07-25: this section used to read "NOT STARTED". It is not. The
findings below were dispatched to a fix subagent whose work IS committed as aa6bf4e (see the
PROCESS INCIDENT note above for why the ledger briefly claimed otherwise). This list is now
the RE-REVIEW SPEC, not a to-do: verify each item against the code rather than the fixer's
claim. BASE for the review package = 44fb9c1.
Copied verbatim (plus the fixer's account and branch constraints) into
`.superpowers/sdd/task-11-fix-brief.md`, which is what the re-reviewer reads.
Re-review of 44fb9c1..aa6bf4e dispatched 2026-07-25 to an opus subagent, human-approved mode.

I3 (Important) — HUMAN DECIDED: add a degenerate-feature guard to GameModel.fit.
The RobustStandardScaler floor only engages below 10*eps ~ 2.22e-15, so it is a machine-
epsilon guard, not a degeneracy guard. The milder case is still live: the 2018 calibration
fold (trained on 2016+2017) has ryoe_diff std 1.05e-2 — far above the floor — but only
5 DISTINCT VALUES across 512 rows. Its predictions run -209.8..+183.1 where every other
fold maxes at |18.8|. Those 256 rows are 13.8% of the 1855-row calibration sample and
measurably flatten the calibrator:
    folds 2018-2024 (as shipped): n=1855, cover coef 0.00342, slate range 46.86-47.80%
    folds 2019-2024:              n=1599, cover coef 0.01023, slate range 46.36-49.18%
Dropping the one degenerate fold TRIPLES the slope and the probability spread.
APPROVED APPROACH: a distinct-value criterion in GameModel.fit (catches both the 1e-17 and
the 1e-2 versions; the eps floor catches only the first). Must RAISE, not warn — a warning
still lets poisoned predictions reach the calibrator. Must not fire on healthy folds.
Then make walk_forward SKIP a fold whose training slice is degenerate, as it already skips
a season with no prior data.
THEN reconsider scripts/slate.py's `prior_seasons[2:]`: it was defence-in-depth added
before the guard existed, and the reviewer found it removed the SAFE 2017 fold (post-fix:
max |model_margin| 14.2) while keeping the DAMAGED 2018 one. With the guard working by
inspection instead of a magic index, [1:] or [:] is likely cleaner. Keep the load-bearing
property: the calibrator is fit on OUT-OF-SAMPLE walk_forward predictions from prior
seasons only, never in-sample.
The reviewer's own view: the true root cause is ryoe_diff being imputed to one shared
value across a season, which lives in features.py/ratings (reserved), and the scaler floor
is a numerical band-aid over it. The guard is the agreed in-scope mitigation.

I2 (Important) — HUMAN DECIDED: exclude exact pushes from Calibrator training.
calibrate.py uses `covered = (d["margin"] > d["spread_line"])`, so an exact push trains as
"did not cover". backtest.evaluate does the OPPOSITE — it filters
`d[d["margin"] != d["spread_line"]]` and documents "Exact pushes are excluded". The repo
holds two contradictory definitions of one concept, so ats_hit_rate and cover_prob estimate
different quantities under the same name. Measured on the real calibration sample (n=1855,
2018-2024): 47.17% strict-cover, 2.75% exact pushes, 48.50% cover among decided games;
intercept -0.1138 → 47.16% at zero edge. A push returns the stake; it is not a loss.
Fix both targets (spread and total) to match evaluate. Add a test that a push-heavy sample
does not depress the intercept. Expected effect: slate cover_prob 46.9-47.8% → 48.2-49.2%.
NOTE this deviates from the Task 10 brief's verbatim code — the human approved it, on the
grounds that the SAME plan's evaluate() specifies the opposite.

I1 (Important) — CLAUDE.md is wrong and incomplete.
CLAUDE.md:48-64 carries a "Known issue" section saying the ryoe_diff/StandardScaler bug is
UNFIXED, that slate.py uses prior_seasons[1:], and that fixing it needs "either a variance
floor in the ridge StandardScaler step or an as-of ratings fix in ratings/, both reserved
for human review". Commit 44fb9c1 added exactly that variance floor and changed the slice.
Every claim is now false and would send a future session re-diagnosing a fixed bug. It is
also the one thing in the deliverable the brief did not ask for. DELETE it; replace with a
short accurate note on RobustStandardScaler + whatever I3's guard becomes. Also:
 - Commands section lists only pip/pytest/ruff and never mentions scripts/, though
   build_dataset.py, backtest.py and slate.py are the project's entire user interface.
 - market/compare.py gets no architecture bullet though the other three layers do.
 - "Market margin MAE is around 9.8-10.3" — measured is 9.752, just under the stated floor.

TEST GAPS (Important) — these mutants SURVIVE tests/test_compare.py today:
 (a) model_spread / market_spread SWAPPED — inverts every displayed pick. Nothing asserts
     either column's value; test_gap_is_model_minus_market only pins spread_gap, which is
     computed independently. This is the exact failure the brief's context paragraph exists
     to prevent.
 (b) cover_prob / over_prob SWAPPED — same class.
 (c) sort by SIGNED gap instead of abs — both fixture gaps are positive, so invisible.
 (d) edge_flag uses > instead of >= — no fixture gap equals the threshold; brief specifies >=.
 (e) edge_flag driven by total_gap instead of spread_gap — same flags on this fixture.
 (f) markdown flips home/away in the Game column — the test only checks substring presence,
     so "BUF @ KC" passes as readily as "KC @ BUF".
Reviewer's suggested starting point:
    row = build_slate(*_inputs()).set_index("game_id").loc["2026_01_KC_BUF"]
    assert (row["model_spread"], row["market_spread"]) == (6.0, 2.5)
    assert (row["cover_prob"], row["over_prob"]) == (0.58, 0.55)
plus a fixture case with a NEGATIVE gap (kills c) and one with a gap EXACTLY at the
threshold (kills d). Verify each new test discriminates by applying the mutation.
Leave the brief-mandated test_sorted_by_absolute_edge untouched (it is non-discriminating —
its expected-first game is already first in input order — but the implementer correctly
added test_sort_order_independent_of_input_row_order beside it, which DOES discriminate).

MINOR (fix while in these files):
 - slate_markdown's "n/a" guard is HALF-APPLIED: a row with a missing line still renders
   `| MIA @ NYJ | -1.5 | -1.0 | -0.5 | 49.0% | 43.2 | nan | +nan | n/a |` — market_spread,
   spread_gap, market_total, total_gap all show nan/+nan. The rationale that "nan%" reads
   as a data-quality bug applies identically to "+nan". The existing test only asserts
   `"nan%" not in md`, so it passes on that output.
 - `slate.py --season 2018` dies with a raw sklearn traceback (ValueError: Found array with
   0 sample(s)) when the calibration window is empty. The script already models the right
   behaviour nearby with `raise SystemExit(f"no games found for ...")`.
 - build_slate's merge has no validate=, unlike walk_forward's validate="one_to_one".
 - total_gap is computed from the ROUNDED model_total (the round overwrites the column
   before the gap is taken) while spread_gap uses the unrounded model_margin.

VERIFICATION the fix pass must pass:
 - full suite (88 currently) + `-W error::FutureWarning -W error::DeprecationWarning`
 - ruff check . and format --check (pre-existing test_epa.py I001 is known; leave it)
 - TASK 9 BACKTEST MUST BE UNCHANGED — the critical regression check, since the guard must
   be a no-op on healthy 5+-season slices:
   `scripts\backtest.py --test-seasons 2021-2025` → n=1359, margin MAE 10.267, market 9.752,
   ATS 0.4992 (n=1326), model_coef -0.0102. If any move, STOP and report.
   DO NOT REBUILD the parquet — it exists and takes many minutes.
 - re-run `scripts\slate.py --season 2025 --week 1`; expect the spread to widen from the
   shipped 46.9-47.8% and the push fix to lift the level ~1.3pp. Still expect everything
   fairly near 50% with few edges — that is honest for a model that ties the market.

--- PAUSED 2026-07-24 (second pause; user asked mid-Task-11) at HEAD aa6bf4e ---
Tree clean, 100 tests passing, nothing pushed. Tasks 1-10 complete and reviewed clean;
Task 11 implemented AND its fix pass committed, but the fix pass is NOT re-reviewed and
Task 11 is NOT accepted. Resume by dispatching the re-review on 44fb9c1..aa6bf4e.
After the Task 11 fix pass + re-review: the FINAL WHOLE-BRANCH REVIEW (most capable model,
per the skill; give it scripts/review-package MERGE_BASE HEAD), which must triage every
deferred Minor finding in this ledger plus the OPEN DESIGN QUESTION and the OAK/SD bug.
Then superpowers:finishing-a-development-branch.

Old note, superseded by the above — extract remaining briefs with
`bash "C:/Users/sbuer/.claude/plugins/cache/superpowers-dev/superpowers/6.1.1/skills/subagent-driven-development/scripts/task-brief" docs/superpowers/plans/2026-07-23-nfl-game-model.md N`
(run via the Bash tool, not PowerShell — the script is bash and silently no-ops under PS).
Same for scripts/review-package BASE HEAD.
TASK 11 MUST DEFAULT TO RIDGE, not gbm — that is Task 9's evidence-based verdict.
After Task 11: final whole-branch review (most capable model, per the skill), which must
triage every deferred Minor finding above plus the OPEN DESIGN QUESTION and the OAK/SD bug.
Then superpowers:finishing-a-development-branch.

--- Task 8 deferred Minor findings (final whole-branch review) ---
Task 8 (predict.py): (1) ESTIMATORS["gbm"] factory takes `alpha` and silently ignores it,
so GameModel(estimator="gbm", alpha=5.0) is a no-op with no warning; (2) n_train_total_ is
set but no test asserts it (only n_train_margin_ is covered); (3) predict() does not
validate FEATURE_COLS presence → opaque KeyError on a malformed frame (same pattern already
logged for Tasks 4 and 7). All non-blocking.

*** OPEN DESIGN QUESTION for final whole-branch review (Important, plan-mandated) ***
[Task 9 UPDATE: this is no longer hypothetical — the OAK/SD bug above is a live instance,
78 games silently zero-filled. Also 387/2639 rows (14.7%) have all-zero NGS diffs,
concentrated at each season's week 1, which IS as designed. Decide both together.]
features.py: `out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)` blanket-fills features
for EVERY row, so a played game with a missing rating/NGS join (team-code mismatch, OR
genuinely no prior-data early-season week) gets silently zero-filled features rather than
an error — can corrupt training data with no signal. Verbatim from the brief. No current
test triggers it. Likely interacts with Task 8 backtest (early-season weeks with no prior
ratings). DECIDE at final review: add a NaN-guard/warning before the final fillna, or
accept as designed. Raise with the human.

--- Minor findings deferred to final whole-branch review ---
Task 3 (epa.py): (1) redundant duplicate `if len(out)>0` blocks ~epa.py:77-89;
(2) double int-cast of n_pass/n_rush (per-group + column); (3) no test for a game
with zero pass or zero rush attempts (NaN in epa_pass/epa_rush); (4) groupby
dropna=True silently drops rows with null defteam but populated posteam — undocumented
assumption. All non-blocking style/coverage nits.
Task 4 (epa.py fit_ratings): (1) no validation that team/opponent columns exist/non-null
(malformed frame → opaque KeyError); (2) all 7 tests hardcode alpha=0.01, default alpha=1.0
never exercised; (3) test_higher_def_rating_means_better_defense only asserts D>A, not the
full 4-team ordering (asymmetric vs the offense test). All non-blocking.
Task 5 (build.py): (1) weeks_back hardcodes 18-week season length in cross-season decay
(approximation only, not leak-relevant; would be a week off for historical 17-week seasons);
(2) column test checks set() not order; (3) three-way outer merge leaves NaN rating for a
team missing a target metric (e.g. no rush plays) — untested path. All non-blocking.
Task 6 (ngs.py): (1) no test for partial-NaN-within-a-present-group weighted averaging;
(2) no test for all-NaN-weight/zero-total-weight path or fully-empty input frames;
(3) _weighted_team_week uses a Python for-loop over groupby (verbatim from brief) — least
idiomatic/efficient; fine at current volume. All non-blocking.
Task 7 (features.py): (1) outdoor games with missing temp get fixed 60.0 placeholder
(deliberate, documented); (2) no input-schema validation → opaque KeyError on malformed
frames. NOTE: the fillna-masking risk is tracked separately above as an OPEN DESIGN QUESTION.
Pre-existing: tests/test_epa.py has a ruff I001 (unsorted imports) from Task 3 — sweep at final review.
Execution mode: subagent-driven-development, one implementer + one reviewer
per task, sonnet for data-manipulation tasks, haiku for pure transcription.
Plan: docs/superpowers/plans/2026-07-23-nfl-game-model.md
Branch: feat/game-model (master has only the two docs commits).

=== FINAL WHOLE-BRANCH REVIEW c00f5df..130e7a3: DONE 2026-07-25 (opus subagent) ===
VERDICT: READY TO MERGE **WITH FIXES**. CRITICAL: none. IMPORTANT: 3 (I1/I2/I3 below).
Human approved both the scope (fold the 2 unreviewed commits into one whole-branch review)
and the subagent dispatch, on resume.
Reviewer independently: ran 111 tests (pass, also clean under -W error), reproduced the
130e7a3 baseline EXACTLY (n=1359, margin 10.274, market 9.752, total 10.684, ATS 0.4977
n=1326, model_coef -0.0218, r2 0.2083), rebuilt the whole feature dataset from cached raw
parquets and confirmed the committed code reproduces game_features.parquet BYTE-FOR-BYTE,
and ran **52 mutations, each against its target test IN ISOLATION**, in a throwaway worktree.
44 died, 8 distinct root causes survived.

*** PROCESS TRAP FOR ANY FUTURE MUTATION TESTING IN THIS REPO -- READ THIS ***
The venv's editable install (__editable__.nfl_game-0.1.0.pth) points at the MAIN repo's src/,
so mutation testing inside a git worktree silently tests the UNMUTATED source. The reviewer's
first run reported "everything survived" for this reason. Set PYTHONPATH=<worktree>/src
(verified to take precedence) or the results are meaningless-but-plausible -- exactly the
failure mode that lets someone claim "mutation-verified" while proving nothing.

--- IMPORTANT findings (fix before merge) ---
I1 (data/nfl.py:19,35,47,62) *** THE ONE TO BLOCK ON *** The team-code normalisation WIRING
has ZERO coverage. Four mutations each leave 111/111 GREEN: deleting normalize_team_codes from
load_schedules, from load_pbp, from load_ngs, and narrowing PBP_TEAM_COLS to drop
posteam/defteam. Cause: tests/test_data_nfl.py:18,33 fixtures are {"game_id","epa"} with NO
team columns, so normalisation is a structural no-op in every loader test. teams.py is PROVEN
to work; NOTHING proves it is CALLED. A refactor can delete the ingestion-site application
wholesale, silently reinstating the LA/LAR bug (243 games re-zero-filled, backtest moves
~0.007 MAE -- nobody would notice). The PBP_TEAM_COLS mutant is the sharpest: it breaks the
EPA ratings join while leaving the schedule join intact = partial corruption.
FIX: 3 tests, one per loader, fixtures returning historical codes in the columns THAT loader
normalises (e.g. load_pbp with posteam=OAK, defteam=LA, home_team=SD -> LV/LAR/LAC).
I2 (features.py:84-103) Feature arithmetic only ~40% pinned. `cpoe_diff = away - home`
(sign flip) and net_rating_diff silently DROPPING its defensive term both survive 9/9
test_features.py. 8 of 14 FEATURE_COLS have no value assertion anywhere. Uniform sign flips
are ~harmless to ridge; the dangerous cases are content changes (net_rating_diff losing half
its definition) and ASYMMETRIC errors. THIS IS THE "one side pinned, mirror unguarded" PATTERN
FOR THE THIRD TIME, IN A THIRD FILE (after Task 10 over_prob and Task 11 total pushes).
I3 (CLAUDE.md:82-87,121-122) The current baseline is NOWHERE IN GIT. CLAUDE.md gives the
market's 9.752 but no model-side numbers, and :121-122 still cites "Task 9's backtest numbers"
(10.267/0.4992/-0.0102), superseded by 130e7a3. **This branch's entire regression discipline
("backtest must be unchanged; if any move, STOP") lives ONLY in this gitignored ledger and
DISAPPEARS ON MERGE.** Record the baseline in CLAUDE.md.

*** (a) THE fillna(0.0) DESIGN QUESTION -- THE LEDGER WAS WRONG, CLAIM RETRACTED ***
Standing claim, repeated in the memory file and the pause notes: "the blanket fillna REMAINS
and still SILENTLY zero-fills legitimate week-1 NGS gaps." **FALSE on both halves.**
Reviewer rebuilt features with line 120 removed: NaN in FEATURE_COLS = 0 of 2639 rows.
features.py:120 is a DEAD NO-OP on current data. The 159 week-1 zero-NGS rows are filled 60
lines earlier by _trailing_ngs's OWN fillna (features.py:57-58), and they are NOT silent --
features.py:47 sets trail_imputed_any=1 for exactly those rows.
CONTROLLER VERIFIED THIS DIRECTLY on the parquet (did not take it on trust):
  all-zero NGS rows 159; week==1: 159/159; ngs_imputed_any==1: 159/159; all-zero RATING rows 0.
So the legitimate week-1 gap is already flagged to the model BY DESIGN. This is the SECOND
falsified ledger claim on this branch (after the "test window contains zero affected rows" one).
REVIEWER'S RECOMMENDATION: ADD THE GUARD ANYWAY -- it is now FREE (zero behaviour change,
zero backtest risk) and it is the exact detector that would have caught LA/LAR in 2016 instead
of 2026, because the RATING columns have NO imputation flag, making a missed ratings join the
one silent-corruption path left. Scope it to the 5 rating-derived columns: RAISE if any is NaN
for a row whose (season,week) IS present in the ratings frame -- that condition is precisely
"the join should have hit and didn't" and cannot fire on a legitimately-unrated week. Leave
NGS on the existing flag-and-fill path. Fix M1 (predict.py:84-85) in the same pass, since the
guard makes an all-NaN column reachable.

--- MINOR findings (reviewer's triage) ---
M1 predict.py:84-85 FIX (was deferred Minor #6). Confirmed untested AND a real gap: an all-NaN
  column skips the guard, reaches Ridge.fit, raises raw sklearn ValueError -- and walk_forward
  catches only DegenerateFeatureError (backtest.py:50), so THE WHOLE BACKTEST CRASHES instead
  of skipping the fold. Unreachable today (0 NaN); becomes reachable if (a) is adopted.
  Two-line fix: `if values.empty: bad.append(col); continue`.
M2 backtest.py:48-54 FIX (was deferred Minor #7). Confirmed REAL, not hypothetical: slate.py's
  calibration corpus silently drops folds 2017+2018 (1855 -> 1599 rows) with nothing printed.
  One warnings.warn.
M3 backtest.py:90,129 ats_by_threshold `>= t`->`> t`, its own push filter deleted, and
  evaluate's O/U push filter deleted all survive. evaluate's ATS push filter IS pinned; its
  O/U mirror is NOT -- the same asymmetry again. Confirms deferred Task-9 Minors (2)(3) + one
  they missed.
M4 tests/test_epa.py:43 test_pass_rush_split_uses_indicator_columns promises a property its
  fixture cannot test -- survives `df["pass"]==1` -> `df["play_type"]=="pass"` because there is
  no scramble/sack row. The plan's Global Constraints name this exact trap. Add one scramble row.
M5 ratings/build.py the row-drop half of the double leak guard is unpinned (keeping zero-weight
  rows survives all 7 tests). Low materiality -- weights still enforce it. Note or pin cheaply.
M6 compare.py:89-90 DROP (was deferred Minor #5). Untested but model_margin/model_total CANNOT
  be NaN given the inner merge at compare.py:38. Would require asserting on unreachable data --
  the same trap correctly identified for the total_gap ordering.
M7 normalize_team_codes silently changes dtype (StringDtype/Categorical -> object). Harmless
  through parquet; note in the docstring.
M8 ngs_imputed_any is 1 on 87.5% of rows -- trail_imputed_any is a cumulative max over ALL prior
  weeks, so by midseason it is near-constant and carries almost no information. NOT a bug, an
  observation; a decayed FRACTION imputed would be the informative version. Backlog.
CONFIRMED NOT RE-OPENED: the total_gap ordering mutant (correctly closed), and nothing was
filed about the model lacking an edge.

--- (b) ~25 deferred Task 3-10 Minors: reviewer's triage ---
FIX (4, all cheap): Task 8(1) ESTIMATORS["gbm"] silently IGNORES alpha, yet --alpha is a
  documented CLI flag on both scripts -- a user can tune a knob that does nothing;
  Task 9(4) walk_forward never forwards `estimator` under test though --estimator gbm produced
  a REPORTED result; Task 9(2)+(3) = M3 above; and the predict() schema-validation item logged
  separately for Tasks 4/7/8 -- pick ONE site (GameModel.predict), not all five modules.
DROP (do not carry past merge): Task 3(1)(2)(3)(4), 4(1)(2)(3), 5(1)(2)(3), 6(1)(2)(3), 7(1),
  9(1)(5), 10(2)(3). Style nits and coverage-of-defensive-hardening. Task 5(1)'s hardcoded
  18-week season confirmed NON-leak-relevant. Task 10(3) confirmed not-a-defect and Task 11
  honours it by construction.
RE-JUDGED -- DROP the shared _validation.py item: the premise is FALSE. backtest.py:19-32 uses
  one 6-col UNION mask (conservative: identical game set across metrics); calibrate.py:23-24
  uses two INDEPENDENT 3-col per-target masks (permissive: a row missing total_line still
  trains the cover model). Two DIFFERENT policies that merely look alike, both correct, both
  documented in place. Extracting a shared module would invite someone to unify them and
  silently break one. The REAL DRY candidate is the push predicate, now repeated in FIVE places
  (backtest.py:80,90,129; calibrate.py:57,66) -- worth a two-line helper.
--- (d) ruff: SWEEP the test_epa.py I001 (only ruff check error in the repo). Also `ruff format
  --check` fails on 4 test files + the plan doc, all pre-existing/verbatim-from-plan, none in
  the unreviewed commits. Do as a SEPARATE style-only commit so the substantive diff stays
  reviewable, or exclude the plan doc.

REVIEWER'S OWN READING OF THE HEADLINE FINDING (unprompted, reinforces it): repairing a real
corruption across 243 games -- 85 inside the test window -- moved margin MAE by 0.007 and ATS
by 0.0015. "Features that carry real signal do not behave like that." The no-edge conclusion is
well-supported and the pipeline that produced it is sound. That is the success case.
NEXT: fix I1 + I3 at minimum (reviewer would block only on I1), then re-run the backtest and
confirm 10.274/9.752/0.4977/-0.0218 at n=1359, then superpowers:finishing-a-development-branch.

*** REVIEW FIX PASS: DONE, commit 9864b0c (INLINE by the controller, 2026-07-25) ***
Human chose scope "I1 + I2 + I3 + guard" from the four options offered. Inline, not via
subagent -- the dispatch approval covered the REVIEW only, and the session config restricts
spawning. Precedent: Task 8 and the 3f398c6 fix pass were also inline.

FIXED -- all three Important findings plus the recommended guard and 2 Minors:
I1: 3 new loader tests in tests/test_data_nfl.py (one per loader) over historical codes.
    ALL FOUR of the reviewer's surviving mutants now DIE in isolation (M10/M11/M12/M13).
I2: 2 new tests in tests/test_features.py pinning all four rating edges, net_rating_diff,
    the three NGS diffs, and ngs_imputed_any, on BOTH fixture games (opposite-signed, so a
    uniform sign flip cannot satisfy both). *** FIXTURE DEFECT FOUND WHILE DOING THIS: ***
    ryoe_per_att (0.1) and separation (2.8) were CONSTANT ACROSS ALL FOUR TEAMS, so every
    ryoe_diff/separation_diff was identically 0.0 and a sign flip on either was undetectable
    BY CONSTRUCTION -- a test asserting 0.0 would have "passed" while proving nothing. The
    reviewer's suggestion that "the fixture already has distinct per-team values" was true
    only of cpoe. Fixture now varies all three. Also nudged MIA def_rating -0.05 -> -0.04 so
    NYJ's home pass edge does not land on exactly 0.0, where too many mutations satisfy it.
    7 mutations (S5-S11) all die, including the two the reviewer found surviving.
I3: CLAUDE.md now carries the regression baseline as a TABLE (n=1359, margin 10.274/market
    9.752, total 10.684/10.309, ATS 0.4977 n=1326, O/U 0.5022 n=1348, model_coef -0.0218,
    r2 0.2083) plus an explicit "ties the market = SUCCESS, do not improve it into an edge"
    line, and a new "Failed joins vs missing data" section. The invariant is now IN GIT and
    no longer dies with this gitignored ledger. Stale "Task 9's backtest numbers" ref fixed.
GUARD (the (a) design question, finally CLOSED): new MissingRatingJoinError +
    _check_rating_joins in features.py, called on `out` immediately BEFORE the blanket
    fillna. Raises when a game has null rating features IN A WEEK WHERE OTHER TEAMS WERE
    RATED -- precisely "the join should have hit and didn't". A week with nobody rated
    (week 1, no strictly-prior games) is legitimate and ignored, which is what makes raising
    safe. NGS deliberately LEFT on flag-and-fill: ngs_imputed_any already makes those
    visible. Scoped to the 5 rating cols exactly as the reviewer recommended.
M1: predict.py `if values.empty` now APPENDS to bad instead of `continue`. Was a real gap --
    an all-NaN col reached Ridge.fit and raised an uncaught sklearn error, and since
    walk_forward catches only DegenerateFeatureError the WHOLE BACKTEST crashed rather than
    skipping one fold. Newly reachable now that the guard exists.
M2: walk_forward emits a RuntimeWarning naming the season + offending feature when it drops
    a fold. Two tests: one that it warns, one that a healthy run stays QUIET (or the warning
    becomes noise).

VERIFICATION (all run, not asserted):
 - 122 tests pass (111 -> 122), also clean under -W error::FutureWarning/-W error::DeprecationWarning.
 - 15 mutations, EACH against its target test IN ISOLATION: all 15 die (M10-M13, S5-S11,
   G6-G9). G9 is the one that matters most for the guard -- dropping the rated_weeks filter
   (so the guard fires on legitimately-unrated week 1 too) correctly kills the "must stay
   quiet" test, proving the guard is narrow and not just loud.
 - BACKTEST BYTE-IDENTICAL, run twice (before and after the ruff reformat): n=1359, margin
   10.274, market 9.752, total 10.684/10.309, ATS 0.4977 n=1326, O/U 0.5022, coef -0.0218,
   r2 0.2083. NO RuntimeWarning fired => no fold is dropped in the real backtest.
 - slate.py --season 2025 --week 1 clean, cover_prob 48.0-49.7%, no warnings.
 - ruff check clean on everything EXCEPT the known pre-existing tests/test_epa.py I001
   (deliberately out of the chosen scope). ruff format clean on all 7 changed .py files.
 - Tree clean after commit, verified with git status.

*** MUTATION-TESTING NOTE, confirmed first-hand ***
Mutating in the MAIN repo (not a worktree) is what works here, because of the editable-install
trap the reviewer documented. Always `git checkout --` the mutated file afterwards rather than
trusting a cp restore: the cp restores content but leaves LF endings where git expects CRLF.

STILL OPEN (deliberately NOT done -- outside the chosen scope, all recorded above):
 - the 4 cheap Task 3-10 Minors the reviewer said FIX: gbm silently ignoring --alpha;
   walk_forward not forwarding `estimator` under test; M3's push/threshold coverage
   (ats_by_threshold's inclusive boundary + its own push filter + evaluate's O/U push filter);
   one predict() schema check.
 - M4 (test_epa.py needs a scramble row -- the test promises a property its fixture cannot
   test), M5, M7, M8.
 - the ruff sweep (test_epa.py I001 + 4 test files failing format --check + the plan doc).
NEXT: superpowers:finishing-a-development-branch.

--- MERGED 2026-07-25: branch complete ---
Merged feat/game-model -> master as 13f34bb (--no-ff, 21 commits). Branch deleted.
122 tests pass on merged master; backtest on master byte-identical (n=1359, margin 10.274,
market 9.752, ATS 0.4977, coef -0.0218). Tree clean. NOTHING PUSHED -- repo has NO remote.
The regression baseline now lives in CLAUDE.md, so it no longer dies with this gitignored
file. Still open (non-blocking, listed in the fix-pass section above): 4 cheap Minors, the
test_epa.py scramble-row gap, and the ruff sweep.

--- DEFERRED-MINORS PASS 2026-07-27: the STILL OPEN list above is now EMPTY ---
Branch chore/deferred-minors off master @ 13f34bb. Two commits: 5a67989 (substantive) and
a7f55c8 (style-only ruff sweep, kept separate as the reviewer asked). 122 -> 133 tests.
Everything the reviewer triaged as FIX, plus M4/M5/M7, is done. M6 and the ~20 DROP items
stay dropped. M8 (ngs_imputed_any near-constant by midseason) stays a backlog observation,
not a defect -- unchanged.

WHAT WAS DONE, and the one judgement call worth recording:
 - gbm/--alpha: GameModel now WARNS when a non-default alpha meets an estimator that
   ignores it (new DEFAULT_ALPHA + ALPHA_IS_RIDGE_ONLY in predict.py; both scripts' --alpha
   help says so). Deliberately NOT wired to HistGradientBoostingRegressor.l2_regularization:
   that is a penalty on leaf values, not coefficients, and adopting it would move gbm off
   l2_regularization=0.0 and silently invalidate the recorded ridge-vs-gbm comparison that
   made ridge the default. Making a knob real is not a free "improvement" when a recorded
   measurement depends on its current value.
 - walk_forward estimator forwarding, M3 (threshold boundary + both push filters),
   predict() schema check at GameModel.predict ONLY, M4 scramble row, M5 row-drop pin,
   M7 dtype docstring note.

VERIFICATION (all run, not asserted):
 - 133 tests pass. ruff check AND ruff format --check both clean across src/tests/scripts.
 - 15 mutants (N1-N15), each against its target test IN ISOLATION, in the main repo:
   all 15 killed. slate.py --season 2025 --week 1 clean, cover_prob 48.0-49.7%.
 - BACKTEST BYTE-IDENTICAL, confirmed on the committed tree: n=1359, margin 10.274,
   market 9.752, total 10.684/10.309, ATS 0.4977 n=1326, O/U 0.5022 n=1348, coef -0.0218,
   r2 0.2083. No RuntimeWarning fired.

*** TWO PROCESS TRAPS HIT THIS PASS, BOTH WORTH KEEPING ***
(1) THE `git checkout --` RESTORE RECIPE DESTROYS UNCOMMITTED WORK. The mutation-testing
    note above is right that `git checkout -- <file>` beats a cp backup (cp leaves LF where
    git wants CRLF) -- but it restores HEAD, so running the harness against UNCOMMITTED
    edits silently reverted every change to predict.py mid-run. It presented as "the
    pattern no longer matches", not as data loss. COMMIT FIRST, THEN MUTATE. The harness
    now refuses to run when a mutation target has unstaged changes.
(2) AN ASSERTION SATISFIED BY THE ERROR MESSAGE'S OWN PROSE. The predict() schema mutant
    "check only FEATURE_COLS, not game_id" SURVIVED, because the test asserted
    `"game_id" in str(exc.value)` and the message's guidance sentence -- "Expected game_id
    plus every FEATURE_COLS entry" -- contains that literal regardless of what the check
    found. Same shape as the constant-fixture trap: the assertion could not fail. Fixed by
    asserting on columns that appear nowhere in the static prose, plus a separate test that
    drops game_id alone. When asserting a name appears in an error, check the name is not
    already in the boilerplate.

MERGED 2026-07-27: chore/deferred-minors -> master as 352b1a3 (--no-ff, 2 commits), branch
deleted, pushed to origin (confirmed with git ls-remote, not inferred). 133 tests and both
ruff gates pass on merged master; backtest on merged master byte-identical to the baseline.

--- ARCHIVED 2026-07-27: this file is no longer the only copy ---
This ledger and the 22 task briefs/reports beside it were gitignored for their whole life by
.superpowers/sdd/.gitignore (a bare `*`), so they existed on one machine and any
`git clean -fdx` would have destroyed them. They are now committed:
  - this file       -> docs/development-log.md (verbatim, under an explanatory header)
  - task-*.md       -> docs/sdd-archive/
  - review-*.diff   -> NOT archived. Plain `git diff <a>..<b>` over ranges still reachable
                       in history, so they are regenerable; the recipe for all 15 is in
                       docs/sdd-archive/README.md.
CLAUDE.md now points at the archive and states the precedence rule: CLAUDE.md is
authoritative, this log is evidence and not authority. THE ARCHIVE IS A FROZEN COPY -- if
anything is appended here afterwards, re-run the copy or the two will diverge.
NEXT: nothing outstanding. The deferred list is empty and the model is complete.
