# 2026 tracker go-live checklist

**Hard deadline: 2026-09-05 00:20 UTC** (Fri Sep 4, 8:20pm ET).

That is the moment the 5-day lock opens on `2026_01_NE_SEA`, the season's first game
(kickoff 2026-09-10 00:20 UTC). If `ENABLE_OFFICIAL_TRACKER` is not `true` by then, the
opener is never recorded — the tracker publishes on the first run *inside* the window and
`_new_record` freezes what it finds, so a late flip cannot backfill it.

Note the deadline moved: under the old 4-day lock it was 2026-09-06 00:20 UTC. Widening the
lock pulled it a day earlier.

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
