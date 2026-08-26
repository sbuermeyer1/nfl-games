# Publication Lead Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the pick publication lock from 24 hours to 4 days before kickoff, floored by a features-artifact vintage check so no game is ever published on a stale prediction.

**Architecture:** Two changes to the publication gate. `PUBLISH_BEFORE` becomes 4 days. A new required `first_publishable_week` parameter on `advance_live_ledger` blocks new-record creation for any week whose features were not built from a complete prior week. The floor is derived from the features artifact itself — `refresh_2026` appends only `active_prediction_weeks` (the first two unplayed weeks), so the minimum week present for the tracked season is exactly the week whose predecessors were all final at build time.

**Tech Stack:** Python 3.12, pandas, pytest, ruff. Run everything with `.venv/Scripts/python.exe` from the repo root.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-26-publication-lead-time-design.md`. Read it before starting.
- The floor parameter **must not have a permissive default**. A default meaning "no floor" reintroduces the unguarded path silently. `None` is allowed and means *publish nothing* — that is restrictive, not permissive.
- The floor applies **only to new-record creation**. Records that already exist must always advance, so they can still capture closing lines and finalize.
- Predictions stay **frozen at publication**. Do not add re-pricing.
- Do not modify `refresh_2026.py`, its cadence, or the feature build.
- `ruff check .` must be clean. `ruff format --check .` has a standing pre-existing 11-file gap — do not fix it, and do not let it grow.
- Commit messages: write to a file and use `git commit -F <file>`. PowerShell 5.1 shatters `-m` messages containing double quotes.
- Branch: `research/line-value` (current). The spec is committed there at `da246aa`.

---

### Task 1: Add the vintage floor to `advance_live_ledger`

`PUBLISH_BEFORE` stays at 24 hours in this task. Only the floor is introduced, so the two behaviour changes stay independently reviewable.

**Files:**
- Modify: `src/nfl_game/tracking/live.py:208-236`
- Modify: `tests/test_live_tracking.py` (30 call sites, plus new tests)
- Test: `tests/test_live_tracking.py`

**Interfaces:**
- Produces: `advance_live_ledger(existing_live, schedule, predictions, now, *, first_publishable_week, model_version=HISTORICAL_MODEL_VERSION)`. `first_publishable_week` is `int | None` and is **keyword-only and required**. `model_version` becomes keyword-only (safe: its only caller already passes it by keyword, at `tests/test_live_tracking.py:274`).

- [ ] **Step 1: Add a `week` parameter to the schedule fixture**

`schedule_fixture` currently hardcodes `"week": 1`. Task 1 needs a week-2 game. In `tests/test_live_tracking.py`, change the signature and the dict entry:

```python
def schedule_fixture(
    kickoff,
    *,
    spread=3.0,
    total=44.0,
    result=np.nan,
    actual_total=np.nan,
    game_id=GAME_ID,
    week=1,
):
```

and inside the returned record, replace `"week": 1,` with `"week": week,`.

- [ ] **Step 2: Add a test wrapper and repoint the existing call sites**

Add this helper immediately after `empty_live_ledger()` in `tests/test_live_tracking.py`:

```python
def advance(existing, schedule, predictions, now, *, first_publishable_week=1, **kwargs):
    """Existing tests all use week-1 fixtures, so the floor defaults to 1 here.

    The production signature has no default on purpose; this default lives in the test
    file only, so the floor tests below must pass `first_publishable_week` explicitly.
    """
    return advance_live_ledger(
        existing,
        schedule,
        predictions,
        now,
        first_publishable_week=first_publishable_week,
        **kwargs,
    )
```

Then repoint every existing call. From the repo root:

```bash
sed -i 's/= advance_live_ledger(/= advance(/g; s/return advance_live_ledger(/return advance(/g' tests/test_live_tracking.py
```

Verify the wrapper itself was not rewritten — it must still call `advance_live_ledger`:

```bash
grep -n "advance_live_ledger" tests/test_live_tracking.py
```

Expected: the `from nfl_game.tracking.live import ...` line and the one call inside `advance()`. If `sed` rewrote the wrapper's own body, restore that line by hand.

- [ ] **Step 3: Write the failing tests**

Add to `tests/test_live_tracking.py`:

```python
def test_floor_publishes_the_first_active_week():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    assert len(advanced) == 1
    assert advanced.loc[0, "week"] == 3


def test_floor_blocks_a_week_whose_features_predate_the_prior_week():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        NOW,
        first_publishable_week=2,
    )

    assert advanced.empty


def test_floor_blocks_the_week_after_the_first_active_week():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=4),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    assert advanced.empty


def test_floor_of_none_publishes_nothing():
    kickoff = NOW + pd.Timedelta(hours=12)

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=1),
        predictions_fixture(),
        NOW,
        first_publishable_week=None,
    )

    assert advanced.empty


def test_floor_never_blocks_an_existing_record_from_advancing():
    kickoff = NOW + pd.Timedelta(hours=12)
    published = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        NOW,
        first_publishable_week=3,
    )

    # The floor has since moved on; the existing record must still advance.
    advanced = advance_live_ledger(
        published,
        schedule_fixture(kickoff, week=3, result=7.0, actual_total=45.0),
        predictions_fixture(),
        kickoff + pd.Timedelta(hours=7),
        first_publishable_week=4,
    )

    assert len(advanced) == 1
    assert advanced.loc[0, "actual_margin"] == 7.0
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_live_tracking.py -k floor -v
```

Expected: FAIL — `TypeError: advance_live_ledger() got an unexpected keyword argument 'first_publishable_week'`.

- [ ] **Step 5: Implement the guard**

In `src/nfl_game/tracking/live.py`, change the signature:

```python
def advance_live_ledger(
    existing_live,
    schedule,
    predictions,
    now,
    *,
    first_publishable_week,
    model_version=HISTORICAL_MODEL_VERSION,
):
    """Advance live facts without mutating any caller-owned frame.

    `first_publishable_week` is the earliest week whose features were built from a
    complete prior week -- the minimum week present in the features artifact for the
    tracked season. A new record is created only for that week; None publishes nothing.
    Records that already exist always advance, regardless of the floor, so they can
    still capture closing lines and finalize.
    """
```

Then in the loop, extend the new-record gate (currently `live.py:234-236`):

```python
        record = records.pop(game_id, None)
        if record is None:
            if now < kickoff - PUBLISH_BEFORE:
                continue
            if first_publishable_week is None:
                continue
            if int(game["week"]) != int(first_publishable_week):
                continue
            prediction = prediction_rows.get(game_id)
```

Leave the rest of the loop untouched.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_live_tracking.py -v
```

Expected: PASS, all tests in the file, including the 30 repointed ones.

- [ ] **Step 7: Pin the invariant the floor depends on**

The floor is only correct because `active_prediction_weeks` returns the first two *unplayed* weeks. Add to `tests/test_schedule.py`:

```python
def test_first_active_week_has_every_prior_week_final():
    """The vintage floor in advance_live_ledger depends on this property.

    If active_prediction_weeks ever stops returning a prefix of the unplayed weeks,
    the minimum week in the features artifact stops implying that its predecessors
    were complete at build time, and the floor silently admits stale predictions.
    """
    now = pd.Timestamp("2026-10-01T12:00:00Z")
    rows = []
    for week in range(1, 6):
        kickoff = pd.Timestamp("2026-09-06T17:00:00Z") + pd.Timedelta(weeks=week - 1)
        played = kickoff + pd.Timedelta(hours=6) <= now
        rows.append(
            {
                "game_id": f"2026_{week:02d}_AAA_BBB",
                "season": 2026,
                "week": week,
                "away_team": "AAA",
                "home_team": "BBB",
                "kickoff_at": kickoff,
                "result": 3.0 if played else np.nan,
                "total": 44.0 if played else np.nan,
            }
        )
    schedule = pd.DataFrame(rows)

    weeks = active_prediction_weeks(schedule, now)
    first = min(weeks)
    prior = schedule.loc[schedule["week"] < first]

    assert not prior.empty, "fixture must contain at least one completed week"
    assert all(is_final_game(row, now) for _, row in prior.iterrows())
```

Add whatever of `numpy as np`, `pandas as pd`, `active_prediction_weeks`, `is_final_game` the file does not already import. Check first:

```bash
grep -n "^import\|^from" tests/test_schedule.py
```

- [ ] **Step 8: Run the new invariant test**

```bash
.venv/Scripts/python.exe -m pytest tests/test_schedule.py -v
```

Expected: PASS.

- [ ] **Step 9: Mutation-test the guard**

Confirm each new conjunct is load-bearing. Commit first — a `git checkout --` restore reverts uncommitted edits and presents as "pattern not found".

```bash
printf 'wip: floor guard before mutation testing\n' > /tmp/msgwip.txt
git add -A && git commit -F /tmp/msgwip.txt
```

Mutation A — delete the `None` branch:

```bash
sed -i '0,/if first_publishable_week is None:/{/if first_publishable_week is None:/,+1d}' src/nfl_game/tracking/live.py
.venv/Scripts/python.exe -m pytest tests/test_live_tracking.py -k floor -v
git checkout -- src/nfl_game/tracking/live.py
```

Expected: `test_floor_of_none_publishes_nothing` FAILS.

Mutation B — invert the week comparison to `==`:

```bash
sed -i 's/if int(game\["week"\]) != int(first_publishable_week):/if int(game["week"]) == int(first_publishable_week):/' src/nfl_game/tracking/live.py
.venv/Scripts/python.exe -m pytest tests/test_live_tracking.py -k floor -v
git checkout -- src/nfl_game/tracking/live.py
```

Expected: `test_floor_publishes_the_first_active_week` FAILS.

If either mutation survives, the corresponding test is not discriminating — fix the test before continuing.

- [ ] **Step 10: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/tracking/live.py tests/test_live_tracking.py tests/test_schedule.py
printf '%s\n' "feat: gate publication on features-artifact vintage" "" "advance_live_ledger now requires first_publishable_week and creates a new" "record only for that week. refresh_2026 appends only the first two unplayed" "weeks, so the minimum week present for a season is the one whose predecessors" "were all final at build time; the next week was built without the current" "week's results and must not be published from." "" "Existing records always advance regardless of the floor." > /tmp/msg1.txt
git commit -F /tmp/msg1.txt
```

---

### Task 2: Wire the floor through `update_live_tracker`

**Files:**
- Modify: `scripts/update_live_tracker.py:206-211` (`_select_schedule`), `:296-325` (`main`)
- Test: `tests/test_update_live_tracker.py`

**Interfaces:**
- Consumes: `advance_live_ledger(..., first_publishable_week=...)` from Task 1.
- Produces: `_first_publishable_week(features: pd.DataFrame, season: int) -> int | None`; `_select_schedule(schedule, live, now, first_publishable_week)` gains a fourth positional parameter.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_update_live_tracker.py`:

```python
def test_first_publishable_week_is_the_minimum_week_for_the_season():
    features = pd.DataFrame(
        [
            {"game_id": "2025_18_AAA_BBB", "season": 2025, "week": 18},
            {"game_id": "2026_03_AAA_BBB", "season": 2026, "week": 3},
            {"game_id": "2026_04_AAA_BBB", "season": 2026, "week": 4},
        ]
    )

    assert update_live_tracker._first_publishable_week(features, 2026) == 3


def test_first_publishable_week_is_none_when_the_season_has_no_rows():
    features = pd.DataFrame(
        [{"game_id": "2025_18_AAA_BBB", "season": 2025, "week": 18}]
    )

    assert update_live_tracker._first_publishable_week(features, 2026) is None


def test_select_schedule_excludes_weeks_above_the_floor():
    now = pd.Timestamp("2026-09-20T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_03_AAA_BBB",
                "week": 3,
                "kickoff_at": now + pd.Timedelta(hours=12),
            },
            {
                "game_id": "2026_04_CCC_DDD",
                "week": 4,
                "kickoff_at": now + pd.Timedelta(hours=13),
            },
        ]
    )
    live = pd.DataFrame({"game_id": pd.Series(dtype=str)})

    selected = update_live_tracker._select_schedule(schedule, live, now, 3)

    assert selected["game_id"].tolist() == ["2026_03_AAA_BBB"]


def test_select_schedule_keeps_existing_records_above_the_floor():
    now = pd.Timestamp("2026-09-20T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_04_CCC_DDD",
                "week": 4,
                "kickoff_at": now + pd.Timedelta(hours=13),
            }
        ]
    )
    live = pd.DataFrame({"game_id": ["2026_04_CCC_DDD"]})

    selected = update_live_tracker._select_schedule(schedule, live, now, 3)

    assert selected["game_id"].tolist() == ["2026_04_CCC_DDD"]


def test_select_schedule_publishes_nothing_when_the_floor_is_none():
    now = pd.Timestamp("2026-09-20T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_03_AAA_BBB",
                "week": 3,
                "kickoff_at": now + pd.Timedelta(hours=12),
            }
        ]
    )
    live = pd.DataFrame({"game_id": pd.Series(dtype=str)})

    selected = update_live_tracker._select_schedule(schedule, live, now, None)

    assert selected.empty
```

Check how the file already imports the script and reuse that. If it imports symbols directly rather than the module, adapt the `update_live_tracker.` prefix accordingly:

```bash
grep -n "^import\|^from" tests/test_update_live_tracker.py
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_update_live_tracker.py -k "floor or publishable or select_schedule" -v
```

Expected: FAIL — `AttributeError: module ... has no attribute '_first_publishable_week'`, and `_select_schedule() takes 3 positional arguments but 4 were given`.

- [ ] **Step 3: Implement**

Add to `scripts/update_live_tracker.py`, above `_select_schedule`:

```python
def _first_publishable_week(features: pd.DataFrame, season: int) -> int | None:
    """The earliest week whose features were built from a complete prior week.

    refresh_2026 appends only `active_prediction_weeks` -- the first two unplayed
    weeks -- so the minimum week present for the season is the one whose predecessors
    were all final at build time. The week after it was built without the current
    week's results, so publishing from it would freeze a stale prediction.
    """
    weeks = features.loc[features["season"].eq(season), "week"]
    if weeks.empty:
        return None
    return int(weeks.min())
```

Replace `_select_schedule` with:

```python
def _select_schedule(
    schedule: pd.DataFrame,
    live: pd.DataFrame,
    now: pd.Timestamp,
    first_publishable_week: int | None,
) -> pd.DataFrame:
    existing_ids = set(live["game_id"].astype(str))
    if first_publishable_week is None:
        eligible = pd.Series(False, index=schedule.index)
    else:
        eligible = schedule["kickoff_at"].le(now + PUBLISH_BEFORE) & schedule["week"].astype(
            int
        ).eq(int(first_publishable_week))
    existing = schedule["game_id"].astype(str).isin(existing_ids)
    return schedule.loc[eligible | existing].copy()
```

In `main`, after `service = SlateService(features)` add:

```python
    first_publishable_week = _first_publishable_week(features, args.season)
```

Change the `_select_schedule` call to pass it:

```python
    selected_schedule = _select_schedule(schedule, existing_live, current, first_publishable_week)
```

And the `advance_live_ledger` call:

```python
    advanced_live = advance_live_ledger(
        existing_live,
        selected_schedule,
        predictions,
        current,
        first_publishable_week=first_publishable_week,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_update_live_tracker.py -v
```

Expected: PASS, whole file.

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check .
git add scripts/update_live_tracker.py tests/test_update_live_tracker.py
printf '%s\n' "feat: derive the publication floor from the features artifact" "" "update_live_tracker computes the first publishable week from the features" "frame it already loads and passes it to both _select_schedule and" "advance_live_ledger. Narrowing _select_schedule avoids computing predictions" "for games the floor will refuse to publish; existing records still pass" "through so they can capture closing lines and finalize." > /tmp/msg2.txt
git commit -F /tmp/msg2.txt
```

---

### Task 3: Flip `PUBLISH_BEFORE` to 4 days

**Files:**
- Modify: `src/nfl_game/tracking/live.py:16`
- Modify: `tests/test_live_tracking.py:64-83` (the 24-hour boundary test)
- Modify: `README.md`, `CLAUDE.md`
- Test: `tests/test_live_tracking.py`

**Interfaces:**
- Consumes: the floor from Tasks 1 and 2. No signature changes.

- [ ] **Step 1: Update the boundary test to 4 days**

In `tests/test_live_tracking.py`, rename `test_publication_starts_at_exactly_24_hours_but_not_before` to `test_publication_starts_at_exactly_four_days_but_not_before` and change its first line from `kickoff = NOW + pd.Timedelta(hours=24)` to:

```python
    kickoff = NOW + pd.Timedelta(days=4)
```

Leave the body otherwise unchanged — it already tests both directions via `too_soon` and `boundary`.

Also rename `test_first_run_inside_24_hours_freezes_prediction_and_available_markets` to `test_first_run_inside_the_window_freezes_prediction_and_available_markets`. Its body uses `NOW` and a 2-hour offset, so it needs no other change.

- [ ] **Step 2: Add the slot-behaviour tests**

These pin the cases from the spec's lead-time table that the constant alone does not express. Add to `tests/test_live_tracking.py`:

```python
def test_thursday_game_is_held_by_the_floor_not_the_four_day_mark():
    """A week-3 Thursday kickoff sits 4 days after the week-2 Sunday slate.

    The 4-day mark alone would publish it while week 2 was still being played, on
    week-1 features. The floor is what holds it until the week-3 refresh.
    """
    kickoff = pd.Timestamp("2026-09-24T00:15:00Z")  # Thu 8:15pm ET
    four_days_out = kickoff - pd.Timedelta(days=4)  # Sun, week 2 still in progress

    held = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        four_days_out,
        first_publishable_week=2,
    )
    released = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        four_days_out,
        first_publishable_week=3,
    )

    assert held.empty
    assert len(released) == 1


def test_sunday_game_publishes_at_the_full_four_days():
    kickoff = pd.Timestamp("2026-09-27T17:00:00Z")  # Sun 1:00pm ET

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        kickoff - pd.Timedelta(days=4),
        first_publishable_week=3,
    )

    assert len(advanced) == 1


def test_friday_game_is_held_by_the_floor():
    """The one Friday afternoon game on the 2025 calendar is floor-bound at ~3.4 days.

    Its 4-day mark lands on the prior Monday, before the Tuesday refresh that folds in
    the previous week. Kept as a distinct case because Friday football is rare enough
    that a Thursday-only test would not cover it.
    """
    kickoff = pd.Timestamp("2026-09-25T19:00:00Z")  # Fri 3:00pm ET
    four_days_out = kickoff - pd.Timedelta(days=4)  # Mon, week 2 not yet finalized

    held = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        four_days_out,
        first_publishable_week=2,
    )
    released = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=3),
        predictions_fixture(),
        four_days_out,
        first_publishable_week=3,
    )

    assert held.empty
    assert len(released) == 1


def test_week_one_publishes_at_four_days_with_a_vacuous_floor():
    """Week 1 has no prior week, so the floor is satisfied by first_publishable_week=1."""
    kickoff = pd.Timestamp("2026-09-13T17:00:00Z")

    advanced = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(kickoff, week=1),
        predictions_fixture(),
        kickoff - pd.Timedelta(days=4),
        first_publishable_week=1,
    )

    assert len(advanced) == 1
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_live_tracking.py -v
```

Expected: FAIL. `test_publication_starts_at_exactly_four_days_but_not_before` fails because 4 days out is still outside a 24-hour window, and the three new slot tests fail for the same reason.

- [ ] **Step 4: Flip the constant**

In `src/nfl_game/tracking/live.py:16`:

```python
PUBLISH_BEFORE = pd.Timedelta(days=4)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_live_tracking.py tests/test_update_live_tracker.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the full suite**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: PASS. The baseline before this plan is 133 tests in the main repo; the count should rise by the tests added here and nothing should fail. If a test outside these files fails, it is coupled to the 24-hour window — read it before changing it, and report it rather than silently adjusting an assertion.

- [ ] **Step 7: Update the docs**

In both `README.md` and `CLAUDE.md`, find the text describing the 24-hour publication window:

```bash
grep -n "24 hours\|24-hour\|hours=24" README.md CLAUDE.md
```

Replace each with a description of the new rule. Use this wording:

> Picks lock **4 days before kickoff**, floored so that no game is published until the
> features artifact has been rebuilt from a complete prior week. In practice the Sunday
> and Monday slate gets the full 4 days; Thursday games and Thanksgiving are held by the
> floor at roughly 2.3–2.6 days.

If neither file mentions the window, add the paragraph to the section describing the live tracker rather than skipping this step.

- [ ] **Step 8: Verify the lint gap did not grow**

```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check . | tail -3
```

Expected: `ruff check` clean. `ruff format --check` reports the standing 11 files and no more.

- [ ] **Step 9: Commit**

```bash
git add src/nfl_game/tracking/live.py tests/test_live_tracking.py README.md CLAUDE.md
printf '%s\n' "feat: lock picks 4 days before kickoff instead of 24 hours" "" "Measured CLV at the 4-day number is +0.267 pts on spreads at edge >= 2" "(z = 3.68, positive in all five seasons), worth roughly 55% of the 0.48" "points needed to clear the vig. The 24-hour lock sat after most of that" "movement had already happened." "" "The vintage floor from the preceding commits is what makes 4 days safe:" "measured on the 2025 schedule, a flat 4-day rule would have published" "21 of 272 games on stale features, and a 6-day rule 236 of 272." > /tmp/msg3.txt
git commit -F /tmp/msg3.txt
```

---

## Verification

- [ ] `.venv/Scripts/python.exe -m pytest -q` — full suite green
- [ ] `.venv/Scripts/python.exe -m ruff check .` — clean
- [ ] `.venv/Scripts/python.exe -m ruff format --check .` — still exactly the standing 11 files
- [ ] `git status` — tree clean apart from the pre-existing untracked `task11-test.tmp.py`
- [ ] Dry run against the real artifact:

```bash
.venv/Scripts/python.exe scripts/update_live_tracker.py --dry-run
```

Expected: exits 0. The 2026 season has not started (opener 2026-09-09), so the current artifact holds weeks `[1, 2]` and the floor is 1. Week 1 kickoffs are more than 4 days out as of 2026-08-26, so the run should report no new records. A non-zero count before 2026-09-05 means the floor or the window is wrong — investigate rather than accepting it.

## Out of scope

- Re-pricing or updating a prediction after publication.
- Any change to `refresh_2026.py`, its cadence, or the feature build.
- Line shopping guidance.
- The ridge-v2 selection defect and the public-record correction.
- Pushing the branch — the repo is public and pushing is the user's call.
