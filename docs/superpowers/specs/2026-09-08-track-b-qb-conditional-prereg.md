# Pre-registered: does the QB block help on the games where it fires?

**Written and committed 2026-09-08, before any arm was fitted.** Nothing below was chosen
with knowledge of an outcome. The measurement script is written after this file is
committed, and this file is not edited afterwards — a correction goes in a new section,
dated, saying what changed and why.

## Why this exists

Ridge-v2 candidate **C2** (C1 plus quarterback context) was never adopted: the nested
selection chose C1 for both targets in all five evaluation seasons, and the single direct
ablation of the block (`ridge_v2_ablation.parquet`, 2019 margin, inside C4) says removing
its eight columns **improved** MAE by 0.069874.

Both are **pooled** MAE over all games. `qb_new_starter_any` fires on 302 of 1,359 games
(22.2%). A block that is right on the 22% and adds noise on the other 78% loses on pooled
MAE while being exactly the thing a bettor wants. Nothing measured so far distinguishes
those two cases, and that is the whole question here.

## Scope, and one thing this test cannot answer

This measures whether the signal has **value**. It does not measure whether a pick
published five days before kickoff could **act** on it.

That second question needs a depth chart timestamped before kickoff, and the feed only
carries timestamps for 2025-2026. Seasons 2021-2024 are week-labelled with no `dt`, and
`chart_as_of`'s labelled branch ignores the cutoff entirely — so for those seasons both
cutoffs return the identical chart by construction, and any "visible at publication"
figure would be an artefact of the feed's shape. Only **66 of the 302** change games
(2025) can answer it. That is a separate, smaller test and is deliberately not gating this
one: the two questions are independent, and gating the 302-game question behind the
66-game one spends the better data on the weaker branch.

## The arms

All three fit through the **production preprocessing** —
`make_pipeline(RobustStandardScaler(), Ridge(alpha=1.0))`, i.e. `ESTIMATORS["ridge"]` at
`src/nfl_game/model/predict.py:108` with `DEFAULT_ALPHA` — on
`data/processed/game_features_ridge_v2.parquet`.

| arm | schema |
| --- | --- |
| **A0** | `FEATURE_COLS` exactly (14 columns) — the shipped Ridge-v1 baseline |
| **A1** | A0 plus `qb_change_epa_diff` (15) — the sparse delta, zero on 78% of games |
| **A2** | A0 plus the full C2 block (22): `qb_epa_diff`, `qb_cpoe_diff`, `qb_sack_rate_diff`, `qb_int_rate_diff`, `qb_change_epa_diff`, `qb_new_starter_any`, `qb_rookie_any`, `qb_uncertain_any` |

A1 is the lead arm on a stated mechanism: the team EPA ratings are built from games the
*previous* starter played, so they cannot know a backup is starting. The four **level**
features in A2 are non-zero on every game and largely re-describe what `net_rating_diff`
already carries, which is a candidate mechanism for C2's pooled loss. `qb_change_epa_diff`
has 25th and 75th percentiles both exactly 0.0000.

## Protocol

- Walk-forward: for each test season S in **2021-2025**, train on all seasons strictly
  before S, predict S. This is `backtest.walk_forward`'s rule, reimplemented only because
  `GameModel` hardcodes `FEATURE_COLS`.
- **Common fold set.** If any arm's training slice for a season cannot be fitted, that
  season is dropped for **every** arm. The unequal-fold defect that produced a false
  Ridge-v2 conclusion came from exactly this: C0 alone was allowed to skip its two worst
  folds. No arm is scored on a set another arm did not face.
- Targets: `margin` and `total_points`. Margin is primary.
- Subset: `qb_new_starter_any == 1`, evaluated on the same predictions as the pooled figure.

## The adoption rule, fixed now

**Primary — A1 on margin, on the change subset.** Adopt A1 if, versus A0:

1. the paired per-season improvement in change-subset margin MAE has a **positive mean**, and
2. its **lower 90% bound is positive** (the `lower90` convention the Ridge-v2 gates already use), and
3. pooled margin MAE does **not** worsen by more than **0.010**.

Condition 3 is a guard, not a target: a feature that helps where it fires must not pay for
it everywhere else.

**A2 is reported on the identical basis** and adopted only if it clears the same three
conditions *and* beats A1 on change-subset margin MAE. Absent that, A1 is preferred on
parsimony — one column against eight.

**Anything else is a rejection**, including a result that is positive but fails the lower
bound. A pooled improvement with no change-subset improvement is also a rejection: it would
mean the block is doing something other than what this test is about.

## What would falsify the hypothesis

Change-subset margin MAE that is unchanged or worse for both arms. That would say the QB
signal carries nothing the team ratings do not already hold, even on the games where the
starter demonstrably changed, and it would close the question rather than defer it.

## Secondary, reported but not gating

- Totals, all three arms, pooled and on the subset.
- Correlation of `qb_epa_diff` with `net_rating_diff` — the stated redundancy mechanism for
  the level features, currently unmeasured and not to be relied on until it is.
- Change-subset MAE per season, so a single-season effect cannot hide inside a pooled mean.

## Adoption consequence

An adopted arm does **not** ship on its own. It would change `FEATURE_COLS`, the regression
baseline, and the model version, and the tracker forbids a model-version change rewriting
published live history. Adoption here authorises a *proposal*, evaluated against the full
backtest and the tracker contract, and taken to the owner as a decision. The live advisory
(`market/live_starters.py`) is unaffected either way: it is presentation and does not read
these columns.
