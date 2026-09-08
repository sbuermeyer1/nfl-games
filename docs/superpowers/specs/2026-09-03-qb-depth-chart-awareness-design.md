# QB depth-chart awareness — design

**Date:** 2026-09-03
**Branch:** `feat/qb-depth-chart-awareness`
**Baseline:** master @ `2866751`

## Problem

The shipped model does not know who is playing quarterback. `FEATURE_COLS`
(`src/nfl_game/model/features.py:13`) is fourteen columns of team ratings, rest, weather,
divisional status and trailing NGS. `scripts/build_dataset.py` loads play-by-play,
schedules and NGS only — no depth charts, no player stats.

The team EPA ratings that carry most of the model's signal are built from games the
*previous* starter played. When a backup starts, those ratings describe a quarterback who
is not on the field, and the model's number moves by exactly zero. A Tyler Huntley start
for Lamar Jackson, or a rookie starting over an established veteran, is invisible.

## What already exists

The machinery for this was built during the Ridge-v2 research track and never adopted.

- `src/nfl_game/ratings/depth.py` normalizes the depth-chart feed, which arrives as two
  disjoint schemas: a pre-2025 era (332,174 rows, weekly charts, `depth_team` in
  {'1','2','3'}) and a 2025-era (554,215 rows, daily `dt` snapshots, `pos_rank` 1.0-12.0).
  The two share almost no columns. `CHANGE_WINDOW` is fixed at 7 days so a daily-snapshot
  diff is not compared against a weekly one.
- `src/nfl_game/ratings/qb.py` resolves the expected starter from the chart and emits
  `qb_epa_per_db`, `qb_cpoe`, `qb_sack_rate`, `qb_int_rate`, `qb_change_epa`,
  `qb_new_starter`, `qb_rookie`, `qb_uncertain`, each shrunk toward a league prior at
  `QB_PRIOR_DROPBACKS = 200`.
- `src/nfl_game/model/v2_features.py` lifts these to game level as candidate **C2**
  (`qb_epa_diff`, `qb_change_epa_diff`, `qb_new_starter_any`, …).

None of it reaches the shipped model. `src/nfl_game/pipeline/build_v2.py` is its only
consumer.

## Evidence on the table

Three measurements, and the reason none of them closes the question.

1. The nested Ridge-v2 selection chose **C1 for both targets in all five evaluation
   seasons**. C2 (C1 plus quarterback context) was never selected.
2. The single direct ablation of the QB block — `data/processed/ridge_v2_ablation.parquet`,
   2019 margin, inside C4 — shows removing C2's eight columns **improved** MAE by
   0.069874. One fold, so weak, but it is the only attribution that exists.
3. The block is not rare. Over the 1,359 games in 2021-2025, `qb_new_starter_any` fires on
   **302 games (22.2%)**, `qb_rookie_any` on 112 (8.2%), and `qb_uncertain_any` on **1**,
   so historical depth-chart coverage is effectively complete.

Both (1) and (2) are **pooled** MAE over all games. A block that is right on the 22% where
a starter changed and adds noise on the other 78% loses on pooled MAE while being exactly
the thing this project wants. The existing evidence does not distinguish those cases.

### A measurement defect in the existing block

`_targets_from_schedule` (`src/nfl_game/ratings/qb.py:99`) sets the depth-chart cutoff to
**kickoff time**. The tracker publishes at `PUBLISH_BEFORE` (5 days, floored). So C2 was
measured with starter knowledge no published pick could ever have — and still did not earn
selection. Any adoption must be re-measured at the publication cutoff.

### The level/delta split

Of C2's game-level columns, four are *level* features — `qb_epa_diff`, `qb_cpoe_diff`,
`qb_sack_rate_diff`, `qb_int_rate_diff` — non-zero on every game (`qb_epa_diff` std 0.1239,
median 0.0021). One is a *sparse delta*: `qb_change_epa_diff` has 25th and 75th percentiles
both exactly 0.0000, with a tail to 1.5701. It is non-zero only when the starter changed.

**Hypothesis, not a finding:** the level features largely re-describe information
`net_rating_diff` already carries, on 100% of games, which would be a mechanism for C2
losing on pooled MAE. The correlation between `qb_epa_diff` and `net_rating_diff` has not
been measured. Checking it is a task in Track B, not an assumption of this design.

The delta carries what the ratings cannot: the ratings are built from games the previous
starter played, and nothing in them knows a backup is starting. It is also zero on the 78%
of games where nothing changed, so it cannot degrade them the way a level feature can.

A weak prior from a sibling project: the fantasy model tested the same level-versus-delta
split and adopted the QB1-change delta for WR/TE/RB where level features were not adopted.
Different model and different target, so it is a prior and not evidence.

## Decisions taken

| Decision | Choice |
| --- | --- |
| What changes | Two tracks: ship a presentation-layer advisory now, measure conditionally in parallel |
| Research lead arm | The sparse `qb_change_epa` delta, with full C2 as a comparison arm |
| Which side of the lock | **Live view, advisory only.** The published prediction and ledger are never touched |
| Quantification | Raw EPA/dropback now; add a points estimate when Track B measures a margin coefficient |
| `edge_flag` | **Untouched.** A separate `qb_watch` marker sits alongside it |
| Injury feed | Deferred, recorded, not dropped |

### Why live-advisory rather than published

`PUBLISH_BEFORE = pd.Timedelta(days=5)` (`src/nfl_game/tracking/live.py:16`), measured from
each game's own kickoff. For a Sunday 1pm game that is Tuesday around 1pm — before the week's
first practice report (Wednesday) and well before the final status report (Friday). Official
team depth charts also routinely list an injured starter as QB1 all week. At the moment the
model publishes it may genuinely not know a backup is starting; by Sunday morning, when the
bet is actually placed, it does.

The lead is not uniform. A floor holds publication until the features artifact has been
rebuilt from a complete prior week, so Thursday and Thanksgiving games publish at roughly
2.3-2.6 days rather than 5. Gate 0 must therefore report visibility **by weekday cohort**, not
pooled: a short-lead Thursday game may see a starter change that a full-lead Sunday game does
not.

> **Doc discrepancy, unresolved and out of scope here.** `CLAUDE.md`'s "Immutable and
> market-blind facts" section states picks lock **4 days** before kickoff. The code says 5, and
> `CLAUDE.md`'s own "Reading the backtest" section also says 5 ("a true 5 days before each
> game's own kickoff", "the 5-day lock"). This spec follows the code constant. The stale line
> should be corrected separately rather than as a side effect of this work.

The tracker contract forbids rewriting a published prediction, so these are two different
products and must stay separate:

- **the published pick** — locked at `PUBLISH_BEFORE`, graded, in the ledger, limited starter
  knowledge;
- **the live view** — Sunday morning, full starter knowledge, advisory only, never written back.

### Why `edge_flag` is untouched

`edge_flag` (`src/nfl_game/market/compare.py:54`) and the tracker's qualification
(`src/nfl_game/tracking/summary.py:80`, `spread_edge >= QUALIFIED_EDGE`) are **separate code
paths**, so changing `edge_flag` would not disturb the 1,359-row ledger baseline. Suppression
was therefore available and was still rejected: it would make one flag mean two things and
would silently hide edges. An independent `qb_watch` marker keeps both readable — a game can
be flagged for edge, for QB, or both.

---

# Track A — live starter advisory (ships first)

## Architecture

Track A adds no model feature, so `data -> ratings -> model -> market -> tracking -> web`
is unchanged. Three touch points:

- `src/nfl_game/ratings/qb.py` — extended; resolution logic stays where it lives.
- `src/nfl_game/market/live_starters.py` — new; bounded-cache live depth-chart read.
- `src/nfl_game/market/compare.py` and `src/nfl_game/web/` — overlay and rendering.

**Placement note.** A depth chart is not market data. The live fetch goes in `market/`
regardless, because that package is defined as the layer that overlays independently
nullable data after prediction and owns the bounded five-minute live cache and
stale-snapshot behavior — which describes this component exactly. The alternative is a new
top-level package holding one module. `market/live_starters.py` mirrors `market/live.py`'s
`MarketSnapshot` dataclass, staleness flag and fail-soft contract rather than reinventing
them.

## Prerequisite: parameterize the depth-chart cutoff

`_targets_from_schedule` hardcodes `cutoff = kickoff_at`. Replace with an explicit cutoff
policy:

- `at_kickoff` — preserves current behavior exactly, so existing Ridge-v2 research output
  remains reproducible;
- an explicit timestamp — used by the live advisory and by Track B's publication-cutoff arm.

This single change is a prerequisite for both tracks.

## Data flow

    load_depth_charts (live fetch, bounded cache)
      -> normalize_depth_charts          [exists; handles both feed eras]
      -> chart_as_of(cutoff=now)         [exists; now cutoff-parameterized]
      -> expected starter per team
      -> join prior-starter EPA from player_stats
      -> qb_change_epa, qb_new_starter, qb_rookie, qb_uncertain
      -> overlay onto slate AFTER prediction
      -> render

## Slate contract

`build_slate` gains an optional starters frame. New columns:

| column | meaning |
| --- | --- |
| `home_qb`, `away_qb` | expected starter display name |
| `qb_change_epa_home`, `qb_change_epa_away` | expected starter EPA/dropback minus recent starter's |
| `qb_watch` | 1 when either side has a detected starter change; null when unknown |

Player display names require `load_players`. `edge_flag`, `model_margin` and `model_total`
are unchanged by construction.

## Error handling

Every failure is soft. The advisory is decoration on a slate that must still render.

- Fetch fails, or the snapshot is stale beyond its bound: advisory columns are **null** and
  `qb_watch` is **null, not 0**. A null means "we do not know"; a zero would assert "no
  change", which is a different and false claim.
- No chart for a team: the existing code falls back to last week's starter and sets
  `qb_uncertain = 1`. That fallback is retained, but the advisory must **label** the value as
  inferred rather than presenting it as read from a chart.
- Team codes are normalized at ingestion through `src/nfl_game/data/teams.py`, which raises on
  an unrecognized code. Every downstream join is a left merge ending in a blanket `fillna(0.0)`,
  which is how a `LA`/`LAR` mismatch once silently zeroed every Rams game across ten seasons.
  No per-join fixups.
- `web/` remains read-only. The advisory is never written to a packaged artifact.

## Testing

- **Cutoff parameterization is tested in both directions**: the same game resolving a
  *different* starter at a Tuesday cutoff than at a kickoff cutoff. Testing a single cutoff
  would leave "cutoff argument ignored" invisible.
- **Both feed eras**, with fixtures drawn from live data rather than invented. The pre-2025
  weekly and 2025-era daily schemas share almost no columns, and an invented schema matching
  neither era would let a block ship a constant.
- **Fail-soft**: the fetch raises, the slate renders, no exception escapes.
- **Staleness** surfaces on the snapshot and reaches the rendered page.
- **Overlay is inert**: `edge_flag`, `model_margin` and `model_total` are bit-identical with
  and without the advisory joined.
- **Regression baseline holds.** `scripts/backtest.py --test-seasons 2021-2025` must still
  produce margin MAE 10.274 and total MAE 10.684. Track A touches no feature; if that baseline
  moves, something is wrong.

---

# Track B — conditional measurement (separate phase, separate branch)

## Gate 0 (blocking, runs first)

For 2021-2025, resolve QB1 at kickoff and at each game's **own** `PUBLISH_BEFORE` cutoff
(5 days, floored), and count how many of the 302 `qb_new_starter_any` games are visible at
each.

Reported **by weekday cohort**, not pooled, because the floor gives Thursday and Thanksgiving
games a 2.3-2.6 day lead against a Sunday game's full 5. Pooling would average a short-lead
cohort that probably sees the change together with a full-lead cohort that probably does not,
and report a middling rate describing neither.

Anchor the cutoff to **each game's own kickoff**, never to the week's first kickoff. This gate
exists because this project has already been wrong about exactly that: a cache named for a
five-day lead turned out to have a 7.51-day mean, with only 98 of 1,359 games within 0.1 days
of a true 5-day lead, because it anchored to the week.

If the publication-cutoff chart sees almost none of the changes, there is no publishable
feature, only the live advisory — and that is known before a single arm is built.

## Arms

Three arms on a common fold set, reusing the `_reference_seasons` fix in
`src/nfl_game/experiments/v2_selection.py` so the unequal-fold defect is not repeated:

| arm | schema |
| --- | --- |
| A0 | C0 baseline |
| A1 | C0 plus `qb_change_epa_diff` alone (sparse delta) |
| A2 | C0 plus full C2 (eight columns) |

## Reporting

Reported **conditionally**, not only pooled:

- pooled MAE, margin and total;
- MAE restricted to the ~302 `qb_new_starter_any` games;
- CLV on that subset at the true per-game five-day lead, via `scripts/analyze_line_value.py`.

The adoption rule is **pre-registered in code before the run**, not asserted afterward. A
report-only table still shows what it shows, so the restriction has to be enforced
mechanically rather than by operator discipline.

## Additional check

Measure the correlation of `qb_epa_diff` with `net_rating_diff`. This tests the redundancy
hypothesis stated above, which is currently unmeasured and must not be relied on until it is.

---

# Deferred

`load_injuries` is available in `nflreadpy`. Practice participation and game status are the
signal that a starter is actually out; a depth chart is a weak proxy that routinely lists an
injured starter as QB1. Explicitly out of scope here, and worth revisiting once Gate 0 has
quantified how weak the chart alone is.
