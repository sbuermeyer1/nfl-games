# Publication lead time: move the pick lock from 24 hours to 4 days

**Date:** 2026-08-26
**Status:** implemented, then **SUPERSEDED on 2026-08-27 (the lock is now 5 days)**, and its
CLV evidence **WITHDRAWN on 2026-08-31 — every lead figure below was measured at the wrong lead.**
**Branch at time of writing:** `research/line-value`

> ## Evidence withdrawn 2026-08-31: every lead figure below is mislabelled
>
> The d04/d05/d06 caches were built by `snapshot_timestamps`, which anchors ONE snapshot per
> week to **that week's first kickoff**. `PUBLISH_BEFORE` is measured from **each game's own
> kickoff**. Measured over 1,359 games, the `_d05` cache has a mean lead of **7.51 days**, not
> 5 — only 98 of 1,359 games are within 0.1 days of a true 5-day lead.
>
> So "0.2431 at 4 days" and "0.4717 at 5 days" describe roughly 6.5- and 7.5-day leads. The
> **direction** of the finding survives (more lead gave more CLV across these arms), but the
> **labels do not**, and the lock the code implements is shorter than any arm measured here.
>
> Re-measured at true per-game leads: **5d +0.1280, 7d +0.4123, 9d +0.5103, 11d +0.5445**
> (edge >= 2, fixed 778-game common set). And publishing earlier does not help in practice —
> those longer-lead arms were using week N-1 results that do not exist that far out. With
> correctly lagged ratings, 9d scores **+0.0032 (z 0.03)** against 5d's +0.1409.
>
> **The 5-day lock stands. Its justification does not.** Do not cite any number below as a
> lead-specific result. See CLAUDE.md, "It also beats the EARLY line".

> ## Superseded: the lock is 5 days, not 4
>
> Everything below shipped as written. The **4-day number** did not survive its own open
> question. This spec closed with the caveat "lead time is calibrated at 4 days, not
> validated at 4 days" — the 4-day choice rested on a **14-game** availability sample
> (14/14 priced at 4 days, 12/14 at 6). Once the d04/d05/d06 backfills completed, CLV was
> re-measured on the **990 games priced at all three leads**:
>
> | lead | edge>=2 CLV | z | vs 0.48 break-even | edge>=3 CLV |
> | --- | --- | --- | --- | --- |
> | 4 days (this spec) | 0.2431 | 3.15 | 51% | 0.2605 |
> | **5 days (shipped)** | **0.4717** | **5.20** | **98%** | **0.5815** |
> | 6 days | 0.4747 | 5.23 | 99% | 0.5997 |
>
> Almost all the value sits between day 4 and day 5; day 6 adds +0.003 at edge>=2 and turns
> **negative** at edge>=1. Availability also favours 5 days: 1,224/1,360 games priced (90.0%)
> vs 88.8% at 4 days and 84.6% at 6.
>
> **The design below is unchanged and still correct** — only the constant moved. The vintage
> floor, the staleness analysis, and the safety argument all carry over, and the floor is
> what makes the wider lock safe. Re-measured under the 5-day lock: **245/272 games get the
> full 5 days, 27 are floor-bound, the minimum lead anywhere is still 2.31 days, and zero
> games are published on stale features.** No game receives *less* lead than it did at 4
> days — 251 improve and 21 are unchanged.
>
> The lead table in this spec is now reproducible: `scripts/lead_time_distribution.py`
> regenerates it, and its `--assert-baseline` flag re-derives this spec's published 4-day
> figures (272 / 251 / 21 / 2.31) as a check on the method.

## Problem

`PUBLISH_BEFORE = pd.Timedelta(hours=24)` (`src/nfl_game/tracking/live.py:16`) locks each
pick 24 hours before its kickoff. The closing-line-value research established that most of
the line movement the model correctly anticipates has already happened by then.

Measured over 1,178 joined and priced games (`scripts/analyze_line_value.py`):

- Spreads, edge >= 1: **CLV +0.254 pts, z = 4.30**, beat-close 57.5%.
- Spreads, edge >= 2: **+0.267 pts, z = 3.68**, positive in **all five seasons**.
- Totals are stronger and strengthen with conviction: edge >= 1 at z = 5.80.

Converted on the 1,359-game corpus: **1 spread point = 4.93% win probability**, and
break-even at -110 needs 2.38%, i.e. **~0.48 spread points**. The measured +0.267 is
roughly **55% of the way to break-even** — real, and currently being discarded by
publishing too late.

The lead time actually calibrated for line availability is **4 days**: 14/14 games priced at
4 days out, 12/14 at 6 days.

## Two findings that shaped the design

### 1. The recorded "Thursday caveat" does not exist

The prior ledger warned that a Thursday-night result could update two teams' ratings
mid-week, so publishing earlier might use staler inputs. That is not how the model works.

`src/nfl_game/ratings/build.py:36` cuts on **strictly prior weeks**:

```python
is_past = (season < asof_season) | ((season == asof_season) & (week < asof_week))
```

A Thursday game in week N carries `week == asof_week` for every other week-N game, so a TNF
result **can never enter a same-week prediction**. Nor is the prediction computed live:
`SlateService.model_predictions` is a pure lookup on `(season, week)` into a prebuilt feature
parquet, so `now` is not an input to it at all. Publishing Tuesday versus Saturday yields a
bit-identical prediction.

### 2. A real and larger version of the problem replaces it

`PUBLISH_BEFORE` is measured from **each game's own kickoff**. Naively setting it to 4 days
publishes week-N Thursday games on the **previous Sunday afternoon**, while week N-1 is still
being played. Those games' features are then built from data through week **N-2**, and
`_new_record` **freezes the prediction permanently** — subsequent runs only call
`_apply_schedule_change`, which touches kickoff and nothing else.

Games published before the prior week finished, measured on the real 2025 schedule:

| lead from kickoff | stale-feature games |
| --- | --- |
| 3 days | 20 / 272 (7.4%) |
| 4 days | 21 / 272 (7.7%) |
| 5 days | 27 / 272 (9.9%) |
| 6 days | **236 / 272 (86.8%)** |

6 days is disqualified — not for the line-availability reason previously recorded, but
because 87% of the slate would run a week stale.

### Why a time-based floor cannot work

The two workflows run on very different cadences:

- `.github/workflows/refresh-2026-model.yml` — **once daily**, `cron: "30 10 * * *"`.
- `.github/workflows/update-2026-tracker.yml` — **every 15 minutes**,
  `cron: "0,15,30,45 * * 8-12,1-2 *"`.

`update_live_tracker.py` loads its **schedule live** from `nflreadpy` but takes its
**features from the artifact**. The two are decoupled: a live schedule showing week N-1 as
final does **not** imply the features incorporate week N-1. Any floor computed from schedule
timestamps would open the gate hours before the refresh that makes it true.

## Design

Two changes, both to the publication gate.

1. **`PUBLISH_BEFORE`: `pd.Timedelta(hours=24)` -> `pd.Timedelta(days=4)`.**
2. **A vintage floor:** a game publishes only when its week is the **first active week** in
   the features artifact.

The floor is deliberately not time arithmetic. It reads the vintage out of the same artifact
the prediction comes from, so it self-synchronizes with the daily refresh instead of racing it.

### The floor is derivable with no new artifact

`active_prediction_weeks` (`src/nfl_game/data/schedule.py:67`) returns `sorted(unplayed)[:2]`,
so the artifact always holds `{first_unplayed, first_unplayed + 1}`.

- "every week `< W` was final when these features were built" <=> `W <= first_unplayed`
- "`W` is present in the artifact" <=> `W >= first_unplayed`

So the safe condition is exactly:

```
W == min(weeks present in the features artifact for the season being tracked)
```

The floor is computed **per season**, from the rows of the features frame matching the season
under advancement — not hardcoded to 2026, even though `refresh_2026.py` is currently the only
producer. Confirmed against the current artifact: 2026 weeks `[1, 2]`, 32 rows.

Week N+1 is present in the artifact but was built **without** week N's results — publishing it
early is precisely the stale case, and `W == min` excludes it. Week 1 is handled naturally: it
has no prior week, `min` is 1, and the floor is vacuous.

### Resulting lead times

Computed from the real 2025 schedule under the full rule (4-day lead, floored by the first
10:30 UTC refresh at or after the prior week finalizes, `FINALIZATION_DELAY` = 6h):

| slot | games | lead (min-max, days) | floor-bound |
| --- | --- | --- | --- |
| Sun 1:00pm | 131 | 4.00 | 0 |
| Sun 4:25pm | 41 | 4.00 | 0 |
| Sun 4:05pm | 26 | 4.00 | 0 |
| Sun 8:20pm | 18 | 4.00 | 0 |
| Thu 8:15pm | 15 | 2.57-2.61 | 15 |
| Mon 8:15pm | 15 | 4.00 | 0 |
| Sun 9:30am (intl) | 6 | 4.00 | 0 |
| Mon 7:00pm | 2 | 4.00 | 0 |
| Thu 1:00pm (Thanksgiving) | 2 | 2.31 | 2 |
| Mon 10:00pm | 2 | 4.00 | 0 |
| Mon 7:15pm | 2 | 4.00 | 0 |
| Thu 4:30pm (Thanksgiving) | 2 | 2.46 | 2 |
| Thu 8:20pm | 2 | 2.62-4.00 | 1 |
| Sat 8:00pm | 2 | 4.00 | 0 |
| Sat 4:30pm | 2 | 4.00 | 0 |
| Fri 8:00pm | 1 | 4.00 | 0 |
| Fri 3:00pm | 1 | 3.40 | 1 |
| Sat 8:20pm | 1 | 4.00 | 0 |
| Sat 5:00pm | 1 | 4.00 | 0 |

**251 / 272 games receive the full 4 days. 21 are floor-bound. The minimum lead anywhere is
2.31 days. Stale-feature games: 0.**

Today every one of these is 1.0 day.

Notes on the table:

- `Thu 8:20pm` spans 2.62-4.00 because its two games sit in different regimes: the Week 1
  opener has no prior week, so the floor is vacuous and it takes the full 4 days. Per-slot
  **min-max is reported rather than mean** precisely so a mean does not average two regimes
  into one misleading number.
- Saturday football (6 games, 4 slots) sits far enough past the Tuesday refresh that the floor
  never binds.
- `Fri 3:00pm` is floor-bound at 3.40 days — an edge case worth keeping in the test set.

### Where the gate lives

`live.py` is the pure lifecycle module and owns `PUBLISH_BEFORE`, so **it must take the floor
as an explicit parameter** rather than inferring it or trusting its caller.

Filtering only in `update_live_tracker._select_schedule` would leave `advance_live_ledger`
publishing stale records whenever it is called directly — which is how every test calls it.
The parameter must not have a permissive default; a default meaning "no floor" reintroduces
the unguarded path silently.

`update_live_tracker` computes the floor from the features frame it already holds via
`SlateService` and passes it in.

### Failure direction

If the daily refresh fails, the floor stays put and picks **stop publishing** — no picks rather
than bad picks. A prolonged outage surfaces on its own through the existing
`publication_window_missed` path at `LINE_DEADLINE`, so no new alerting is required.

## Testing

Both directions of the new predicate, since testing only the firing case leaves "always fire"
invisible:

1. A week-N game at 4 days out with a current artifact **publishes**.
2. The same game with a stale artifact (`min_active_week == N-1`) **does not**.
3. A week-N+1 game 4 days out **does not** publish, even though its week is in the artifact.
4. Boundary: exactly `kickoff - 4d` publishes; one second earlier does not.
5. The TNF floor actually binds — a Thursday game whose 4-day mark falls on the prior Sunday
   publishes at the floor, not at the 4-day mark.
6. `Fri 3:00pm` style case: a Friday game that is floor-bound at ~3.4 days.
7. Week 1: the floor is vacuous and the full 4 days applies.

The existing `LINE_DEADLINE` / `publication_window_missed` behaviour must be shown unchanged.

## Out of scope

- Re-pricing or updating a prediction after publication. Predictions stay frozen at publication
  by design.
- Any change to `refresh_2026.py`, its cadence, or the feature build.
- Line shopping guidance (separately measured as worth more than the entire model edge).
- The ridge-v2 selection defect and the public-record correction — a different thread.

## Risks and open items

- **The floor couples to `active_prediction_weeks`' `[:2]` slice.** If that slice changes, the
  `W == min` invariant breaks. Implementation should pin this with a test that fails if the
  artifact ever holds a week whose predecessors were not final.
- **Lead time is calibrated at 4 days, not validated at 4 days.** The 14/14-priced figure comes
  from the d04 backfill. Live CLV should be tracked from the ledger's existing `published_*` vs
  `closing_*` columns, which need no schema change.
  **RESOLVED 2026-08-27, and this caveat was right:** measured on 990 games priced at every
  lead, 5 days nearly doubles CLV over 4. The lock is now 5 days. See the banner at the top.
- The 2026 opener is **2026-09-09**; this must land before then to apply for the full season.
