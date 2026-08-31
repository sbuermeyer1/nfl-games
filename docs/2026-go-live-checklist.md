# 2026 tracker go-live checklist

**Hard deadline: 2026-09-05 00:20 UTC** (Fri Sep 4, 8:20pm ET).

That is the moment the 5-day lock opens on `2026_01_NE_SEA`, the season's first game
(kickoff 2026-09-10 00:20 UTC). It is the deadline for capturing the opener at the **full
5-day lead**. Publish later and the record is frozen against a line that has already moved
toward the close.

**Corrected 2026-08-31:** this used to say the 5-day lead reaches "98% of break-even" (CLV
0.2431 at 4 days vs 0.4717 at 5). Those came from a line-history cache anchored to each week's
FIRST kickoff, so the file called `_d05` held a mean 7.51-day lead, not 5. At a true per-game
5-day lead the figure is **+0.1444 at edge>=2 (z=2.41)**, about **30%** of the ~0.483 needed.
Publishing earlier does not recover it -- a 9-day arm with the features it would actually have
scores +0.003 (z=0.03). The lock length is still right; the level is much lower. See CLAUDE.md.

**Correction (2026-08-31): a late flip does NOT lose the opener.** This file previously said
"the opener is never recorded ... a late flip cannot backfill it". That is wrong. The gate at
`tracking/live.py:244` is `if now < kickoff - PUBLISH_BEFORE: continue` — a one-sided *lower*
bound. Nothing closes the window; the only other constraint is `week == first_publishable_week`,
and the floor holds at week 1 until every week-1 game is final, because
`active_prediction_weeks` returns the first two weeks containing any unplayed game.

Measured, by simulating a flip on 2026-09-08 against the committed ledger and current
artifacts: the opener **is** recorded, `published_at` 2026-09-08 12:00 UTC, both markets
`published`, `void_reason` null — at a lead of **1.51 days instead of 5.00**. So the cost of
being late is lead time and line quality, not the record. Hit the deadline anyway; just do not
treat a miss as unrecoverable.

Note the deadline moved: under the old 4-day lock it was 2026-09-06 00:20 UTC. Widening the
lock pulled it a day earlier.

**Scheduled runs are heavily delayed on this repo.** The tracker cron is every 15 minutes and
the refresh cron is 10:30 UTC daily, but GitHub defers them: the last four refresh commits
landed at 20:40, 21:15, 15:12 and 14:57 UTC (4.5-10.7 hours late), and observed tracker runs on
2026-08-30/31 were spaced 2.5-11.5 hours apart. Budget for the first run *inside* the window to
be hours after 00:20 UTC, and flip the variable with margin rather than at the deadline.

## What was verified on 2026-08-27

Simulated against real schedule and feature artifacts, `--dry-run --now <clock>`:

| clock (UTC) | new live records | floor blocked |
| --- | --- | --- |
| 2026-09-04 12:00 | 0 | 0 |
| 2026-09-05 12:00 | 1 | 0 |
| 2026-09-07 12:00 | 2 | 0 |
| 2026-09-09 12:00 | 15 | 0 |

All 16 week-1 games publish (the Monday nighter arrives 2026-09-10 00:15 UTC).
`floor_blocked_games` is 0 at every clock, so the vintage floor never sticks.

The record proposed for the opener, reviewed field by field:

```
game_id                  2026_01_NE_SEA      model_version   ridge-v1
week                     1                   estimator       ridge
published_at             2026-09-05 12:00Z   kickoff_at      2026-09-10 00:20Z
published_spread_line    3.5                 published_total_line    44.5
spread_publication_status published                 total_publication_status published
spread_pick              home                total_pick      over
spread_edge              0.42                total_edge      0.56
spread_grade             pending             void_reason     <NA>
```

Both markets publish with a frozen line and an observation timestamp; nothing is excluded.
Edges are below 1.0, so the opener is not a qualified pick — that is fine and expected.

## Re-verified on 2026-08-31 (artifacts had refreshed four times since 08-27)

All six gates pass on `541347b`.

| gate | evidence |
| --- | --- |
| 1 escape hatch merged | `allow_missing_pbp` at `refresh_2026.py:117,132`; `vars.ALLOW_MISSING_PBP` in the workflow |
| 2 suite green | 879 passed locally (3:19) and in CI (63s); `ruff check` clean |
| 3 week 1 present | dry run prints `"first_publishable_week": 1` |
| 4 dry run clean | exit 0, no traceback, locally and in tracker run #352 |
| 5 records reviewed | simulated write at 2026-09-05 12:00 UTC; fields unchanged from the 08-27 table above |
| 6 floor not stuck | `floor_blocked_games` 0 at every clock tested |

The 08-27 lead-time table reproduces exactly on the refreshed artifacts: 0 new live records at
2026-09-04 12:00 UTC, 1 at 09-05, 2 at 09-07, 15 at 09-09, 16 at 09-11, `floor_blocked_games` 0
throughout. The opener's simulated record is unchanged field for field from the 08-27 review —
lines frozen 3.5/44.5, both markets `published`, no exclusion reasons, `void_reason` null,
edges 0.42/0.56, grades `pending`.

Repository variables are currently **empty**, so `ENABLE_OFFICIAL_TRACKER` is unset and the
workflow's write and commit steps are correctly skipped. `ALLOW_MISSING_PBP` is also unset,
which is the right state until a game actually needs forgiving.

One CI annotation is outstanding and is cosmetic: `actions/checkout` targets Node.js 20, which
is deprecated and being forced onto Node.js 24.

## Go / no-go gates

Run these in order. Any FAIL is a no-go.

1. **Escape hatch is merged.** `grep -n "allow_missing_pbp" src/nfl_game/pipeline/refresh_2026.py`
   returns a hit, and `.github/workflows/refresh-2026-model.yml` reads `vars.ALLOW_MISSING_PBP`.
   Without this, one never-delivered pbp game halts the season with no operator response.
2. **Suite green on the commit you are shipping.** `python -m pytest -q` and `ruff check .`.
3. **Refresh has run recently and produced week 1.**
   `python scripts/update_live_tracker.py --dry-run` prints `"first_publishable_week": 1`.
   If it prints `null`, the features artifact has no 2026 rows — run the refresh first.
4. **Dry run is clean at the real clock.** Same command, exit 0, no traceback.
5. **Review the proposed records** at a clock just past the window opening, and confirm every
   record has a non-null `published_spread_line` / `published_total_line`, a `published_at`
   inside the expected lead, and `void_reason` null.
6. **`floor_blocked_games` is 0.** A nonzero value that persists across days means the floor
   is stuck — usually a game whose result never arrived. Do not go live until it clears.

## Go live

```bash
gh variable set ENABLE_OFFICIAL_TRACKER --body "true"
gh workflow run update-2026-tracker.yml     # do not wait for the 15-minute cron
gh run watch
```

Then confirm the first real write:

```bash
git pull
python -c "import pandas as pd; l=pd.read_parquet('data/processed/tracker_ledger.parquet'); \
  print(l[l.record_type.eq('live')][['game_id','published_at','published_spread_line','spread_pick']])"
```

## After the first write

- The ledger is append-only for published records. A wrong record cannot be edited, only
  voided with `--void-game GAME_ID=REASON` (both parts must be non-blank). Check the
  first one carefully.
- Watch `floor_blocked_games` daily for the first two weeks. Persistent nonzero means the
  floor is stuck, which is the failure mode the escape hatch exists for.
- If a pbp game never lands, follow "When play-by-play never lands for a game" in the README.
  Forgiving a game is not free: that week's ratings are built without it.

## Rollback

Setting `ENABLE_OFFICIAL_TRACKER` to anything other than `true` returns the workflow to
`--dry-run` immediately; it has no write or commit step in that mode. Records already
written stay written — revert the artifact commit if they must go.
