# Web app development log

**This is a historical record, not documentation. `CLAUDE.md` and the "Web dashboard
operations" section of `README.md` are the authoritative descriptions of how the deployed
dashboard actually works.**

What follows is the verbatim working ledger kept while the FastAPI slate dashboard was built
on top of the finished game model, across six planned tasks, several review-and-fix rounds, a
final whole-branch review that ended BLOCKED, and the residual fix that unblocked it. It is
preserved for the same reason as `development-log.md`: the reasoning is not recoverable from
the code or from git history, and the useful part is the mistakes.

## How to read it

**Treat every claim here as evidence, not as authority** — the same rule the model's log
carries, and for the same reason. This ledger records one claim of its own being corrected:
the spec-era warning that an unset `ACCESS_CODE` would leave the app wide open (true of the
sibling fantasy app) is **false for this app**, which fails closed and exits non-zero instead.

The single most transferable lesson: the branch was blocked for its last stretch by a
one-line Dockerfile regression, in which a hardening pass that added a hashed dependency lock
and a digest-pinned base image *also* changed `pip install -e .` to an ordinary install. That
silently moved the package under `site-packages`, where `paths.py`'s
`Path(__file__).parents[2]` no longer resolves to `/app`, so the packaged parquet could not be
found and the container could not start. Nothing in the test suite could see it: the failure
exists only in an installed layout, and the deployed sibling app survives the identical
`paths.py` purely because its Dockerfile kept the editable install. **Install mode was
behavior, not packaging preference.**

## Note on this copy

This is a **frozen copy** made when the branch was integrated. The original lived at
`.superpowers/sdd/2026-07-27-nfl-game-webapp/progress.md`, gitignored by a bare `*` in
`.superpowers/sdd/.gitignore` — meaning it existed on exactly one machine and any
`git clean -fdx` would have destroyed it. If anything is appended to the original after this
copy was taken, the two diverge.

---

# SDD ledger — plan: docs/superpowers/plans/2026-07-27-nfl-game-webapp.md
Pre-flight ruling: configuration artifacts use the approved TDD exception; behavioral Docker and Render verification replaces source-text tests.
Pre-flight ruling: post-integration Task 7 runs only after the reviewed feature is integrated into master.
Task 1 ruling: because the brief orders validation green before later missing-method reds, add the full behavior test set before production code and implement the complete service in one green phase; retain both red checkpoints.
Task 1: complete (commits aa8c4de..7c0fddd, re-review clean: spec PASS, quality APPROVED).
Task 2: complete (commits 7c0fddd..fab8acc, two fix rounds, final re-review clean: spec PASS, quality APPROVED).
Task 3: fix round 1/5 (6 addressed, 1 open � undeclared system Node test prerequisite; commits b592876..30930d1).
Task 3: fix round 2/5 (1 addressed, 0 open � declared QuickJS runtime; commits 30930d1..107800b).
Task 3: complete (commits fab8acc..107800b, review clean).
Task 4: minor (deferred): add launcher success-path wiring test for packaged path and Uvicorn host/port.
Task 4: minor (deferred): add explicit PORT boundary and hostname/IPv6-wildcard rejection tests.
Task 4: observation for Task 5: feature worktree lacks packaged parquet; deployment packaging must supply it.
Task 4: complete (commits 107800b..00660d5, review clean).
Task 5: fix round 1/5 (0 addressed, 1 open � Docker parent-negation leak; commits 45d632b..0b84d50).
Task 5: fix round 2/5 (1 addressed, 0 open � corrected Docker data re-exclusions; commits 0b84d50..ad99baf).
Task 5: deferred release gate � Docker/Render build exact accepted commit; missing ACCESS_CODE exits nonzero; protected non-default PORT health 200 and root 303 /login; Blueprint validation/build succeeds.
Task 5: complete (commits 00660d5..ad99baf, review clean).
Task 6: fix round 1/5 (4 addressed, 0 open � DNS rollback, proxy boundary, health wording, baseline labeling; commits 36e5a9c..b69ba20).
Task 6: complete (commits ad99baf..b69ba20, review clean).
Pre-final runtime test hardening: complete (commits b69ba20..11da306, spec PASS, quality APPROVED).
Final review: fixes required � startup parquet schema validation; stale slate request sequencing/CSV state; reproducible base image + hashed production lock; PowerShell login smoke quoting; login network recovery.
Final review: minor (deferred): commit artifact provenance manifest before the next packaged dataset refresh.
Task 4 deferred tests: closed by commit 11da306; packaged-parquet observation resolved by Task 5.
Final fix wave: 4 addressed, 1 open (commit 11da306..bf0ef4f) � installed-layout artifact path regression from non-editable Docker project install.
Final review: BLOCKED � Docker cannot start because PROCESSED_DIR resolves under site-packages after ordinary install; needs one authorized residual fix + scoped re-review before integration.

--- RESIDUAL FIX: DONE 2026-07-29, commit 8140697 (human authorized option A of three) ---
ROOT CAUSE CONFIRMED INDEPENDENTLY, not taken from this ledger: reproduced with a real
non-editable install (PROJECT_ROOT -> <venv>/Lib, PROCESSED_DIR -> <venv>/Lib/data/processed,
exists=False). git log on the Dockerfile shows Task 5 shipped `pip install -e .` (reviewed
clean at 45d632b) and bf0ef4f replaced it with an ordinary install while adding the hashed
lock + pinned base image. The install-mode switch was COLLATERAL in that hardening pass, not
a deliberate choice -- that is the regression.
REFERENCE IMPLEMENTATION: the deployed fantasy app has a BYTE-IDENTICAL paths.py and works in
production solely because its Dockerfile uses `-e .`, with a comment stating that exact
reason. Broken-vs-working differs in install mode, NOT in paths.py.
FIX: one line, `pip install --no-cache-dir -e . --no-deps --no-build-isolation`, plus the
rationale as a comment so a refactor cannot silently reintroduce it. paths.py, test_smoke.py
and render.yaml deliberately UNTOUCHED -- paths.py is on the model path (imported by
build_dataset/backtest/slate) and the spec puts the model out of scope. Two rejected
alternatives are recorded: an env-override in paths.py (touches shared model code + breaks
the existing test_smoke.py contract) and resolving in game_app.py (adds a deploy env var to
forget).
*** DOCKER IS NOT INSTALLED ON THIS MACHINE, so the ledger's "behavioral Docker verification"
could NOT be run, and the prod lock is manylinux-x86_64-only so it cannot install on Windows
either. Verified instead against a faithful replica of the container /app layout (only the
Dockerfile's COPY targets present) using the Dockerfile's EXACT project install command.
THE REAL DOCKER/RENDER BUILD GATE REMAINS UNMET LOCALLY and must be satisfied on Render's
builder at deploy, exactly as the fantasy app builds on Fly's remote builder. ***
VERIFIED (all run, not asserted):
 - NEGATIVE TEST FIRST, so the check is known to be able to fail: ordinary install -> exit 2,
   "packaged dataset not found: ...\appvenv\Lib\data\processed\game_features.parquet".
 - editable install -> PROJECT_ROOT == /app-replica, parquet found, uvicorn startup complete.
 - fail-closed checks from the plan's Task 7 Step 3: root 303 -> /login; /api/options 401 with
   no cookie; wrong code 401; correct code 200; /api/options 200 with the session cookie.
 - /api/slate?season=2025&week=1 returns real data (cover_prob 48.0-49.7%, matching the
   recorded honest range); /api/slate.csv 200 text/csv.
 - packaged parquet md5 6ae3c75eff7052a8d368226844c73f63 == the model's real dataset, so the
   artifact is not a trimmed copy.
 - 273 tests pass under -W error::FutureWarning/-W error::DeprecationWarning. NOTE: the main
   .venv CANNOT run the web suite (no fastapi -- it predates the web work), and .pytest_cache
   in this worktree is not writable by the sbuer account (the worktree .git is owned by
   ScottDell/CodexSandboxOffline). Ran with a scratch venv + PYTHONPATH=<worktree>/src to
   dodge the documented editable-install trap, and -p no:cacheprovider.
 - ruff check clean. BACKTEST BYTE-IDENTICAL: n=1359, margin 10.274, market 9.752, total
   10.684/10.309, ATS 0.4977 n=1326, O/U 0.5022 n=1348, model_coef -0.0218, r2 0.2083. No
   RuntimeWarning, so no fold was dropped.

*** NEW FINDING, PRE-EXISTING TO THIS FIX, NOT CAUSED BY IT -- triage before merge ***
`ruff format --check src tests scripts` FAILS ON 7 FILES, all added by this branch:
  src/nfl_game/web/__init__.py, service.py, session.py, throttle.py,
  tests/test_web_runtime.py, test_web_service.py, tests/test_webapp.py
MASTER IS CLEAN on the same scope (31 files), and the model project's merge standard was
"ruff check AND ruff format --check both clean across src/tests/scripts", so this branch
would regress a gate master currently upholds. Both checkouts run the SAME ruff 0.16.0, so
this is not a version artifact. Deliberately NOT folded into 8140697: the repo's convention
is a separate style-only commit so the substantive diff stays reviewable. Note ruff 0.16
also reformats python blocks inside markdown, which is why a repo-wide run reports more
files (5 archived docs/sdd-archive briefs) -- those are frozen historical records; leave them.
NEXT: scoped re-review of bf0ef4f..8140697, plus a decision on the 7-file format gate, then
superpowers:finishing-a-development-branch, then post-integration Task 7 (Render).
