# NFL Game Model — Design

**Date:** 2026-07-23
**Status:** Approved, pending implementation plan

## Context

The existing NFL Fantasy Football Model (`~/Documents/NFL Fantasy Football Model`) projects
per-player fantasy output for draft and lineup decisions. This project applies the same data
sources and engineering conventions to a different question: **predicting NFL game outcomes
against the spread and the total**.

The model predicts each game's margin and combined points from team strength alone, without
seeing the betting line, then compares those predictions to the market in a separate layer.
The separation is deliberate — it makes model-vs-market disagreement explicit and measurable
rather than baking the line into the prediction.

## Data sources

All data comes from `nflreadpy`, already proven in the fantasy project. No new external
dependency, no API key, no scraping.

### Verified availability (checked 2026-07-23)

`nflreadpy.load_schedules()` — 7,548 games, seasons 1999–2026, 46 columns. Confirmed coverage
for regular-season games:

| Field | Coverage |
|---|---|
| `spread_line`, `total_line` | complete, 2000–2025 (2026 partial: 67 games posted) |
| `away_moneyline`, `home_moneyline` | complete 2010+, partial 2006–2009, absent before |
| `result` (home margin), `total` (actual points) | complete through 2025 |
| `away_rest`, `home_rest`, `div_game`, `roof`, `surface`, `temp`, `wind` | present |
| `away_qb_id`, `home_qb_id`, `away_qb_name`, `home_qb_name` | present |
| `away_coach`, `home_coach`, `referee`, `stadium_id` | present |

`nflreadpy.load_pbp()` — play-by-play with EPA, the backbone of team ratings.

`nflreadpy.load_nextgen_stats(seasons, stat_type)` — **2016+ only**, three types:

- `passing` — `completion_percentage_above_expectation` (CPOE), `avg_time_to_throw`,
  `avg_air_yards_to_sticks`, `aggressiveness`, `avg_air_yards_differential`
- `rushing` — `rush_yards_over_expected_per_att`, `percent_attempts_gte_eight_defenders`,
  `efficiency`, `avg_time_to_los`
- `receiving` — `avg_separation`, `avg_yac_above_expectation`, `avg_cushion`,
  `percent_share_of_intended_air_yards`

### NGS constraints (measured, not assumed)

1. **Offense only.** There is no defensive NGS table. The defensive half of every matchup must
   come from EPA. This is the single most important constraint on the design.
2. **2016+.** Restricts NGS-era training data to roughly 2,600 regular-season games.
3. **Qualifier gaps.** Measured on 2024 (544 actual team-games): passing covers 539 (99%,
   ~1 QB per team-week); rushing covers 468 (86%). Rushing gaps require explicit handling.
4. **Week 0 rows** are season aggregates and must be filtered out of weekly joins.

NGS earns its place because CPOE, RYOE/attempt, and separation stabilize faster than
box-score yardage — they identify real team quality earlier in a season than raw results do.
That early-season signal is the most plausible source of edge over a market whose priors move
slowly.

## Architecture

New standalone repository at `~/Documents/NFL Game Model`, package `src/nfl_game/`.

**Not** a dependency on the fantasy model. The two share only a handful of loader wrappers;
a cross-repo path dependency would couple deployment and make refactors in either project
break the other. Conventions are copied, code is not: `paths.py` structure, the nflreadpy →
pandas loader idiom with `save=True` parquet caching, pytest/ruff config, and the
one-directional data flow rule.

```
data → ratings → model → market
```

Reverse dependencies (e.g. `ratings` importing from `model`) are prohibited, mirroring the
rule in the fantasy project's CLAUDE.md.

```
src/nfl_game/
  paths.py            RAW_DIR / PROCESSED_DIR
  data/
    nfl.py            schedules, pbp, ngs, injuries loaders
  ratings/
    epa.py            opponent-adjusted offensive/defensive EPA per team-week
    ngs.py            team-level NGS offensive quality aggregation
    build.py          combined recency-weighted rating table
  model/
    features.py       game-level feature assembly
    margin.py         predicts home margin
    total.py          predicts combined points
    calibrate.py      point gap → cover / over probability
  market/
    compare.py        model vs closing line, edge flags
  backtest.py         walk-forward evaluation by season
scripts/
  build_ratings.py
  slate.py            weekly report
  backtest.py
tests/
```

## Ratings layer

### EPA backbone (both sides of the ball)

Per team, per season-week, derived from play-by-play:

- offensive EPA/play, split pass and rush
- defensive EPA/play allowed, split pass and rush
- success rate, offense and defense

**Opponent adjustment is the critical step.** Raw EPA largely measures schedule quality. A
ridge regression of play-level EPA on offense-team and defense-team indicator variables
isolates each team's own effect. Without this, ratings encode who a team played rather than
how good it is.

**Recency weighting.** Ratings are exponentially decayed over recent games, with the halflife
tuned in backtest rather than assumed.

**Preseason prior.** Each season opens with the prior season's final rating regressed toward
the league mean, so Week 1 predictions have a defensible prior instead of a cold start. The
shrinkage factor is tuned in backtest.

### NGS offensive layer (2016+)

Team-week aggregates, weighted by volume:

- passing, attempt-weighted: CPOE, `avg_time_to_throw`, `avg_air_yards_to_sticks`,
  `aggressiveness`
- rushing, attempt-weighted: `rush_yards_over_expected_per_att`,
  `percent_attempts_gte_eight_defenders`
- receiving, target-weighted: `avg_separation`, `avg_yac_above_expectation`

Same exponential recency weighting as the EPA ratings.

**Missingness handling.** Rushing's 86% coverage is handled explicitly: impute the league mean
and set an `is_imputed` flag the model can see. This follows the `qb1_is_unproven` pattern
already used in the fantasy project's `projections/teammate_context.py` — the model is told
when a value is a guess rather than being silently fed one.

## Game model

### Features

Assembled per game in `model/features.py`:

- rating differentials — each team's offense against the opposing defense, pass and rush
  computed separately rather than collapsed
- NGS differentials, same structure
- rest differential (`home_rest` − `away_rest`)
- home field indicator
- `div_game`
- `roof` (dome / outdoor / retractable)
- `temp` and `wind`, applied to outdoor games only
- QB-change flag, derived by comparing `home_qb_id` / `away_qb_id` against each team's recent
  starter

### Targets

Two separate models:

- **margin** — `result` (home score − away score)
- **total** — `total` (actual combined points)

### Estimators

Ridge / elastic-net as the honest baseline; gradient boosting as the challenger. Selection is
decided by backtest, not by preference. Both sit behind a single interface so the choice can
change without touching downstream code.

Training window: 2016+, the NGS era.

## Calibration and market comparison

A logistic regression maps `(model_margin − market_spread)` to historical cover outcomes, and
an equivalent fit maps total disagreement to over/under outcomes. This converts a raw point
gap into a probability with an empirical basis — "the model likes it by 4" becomes a hit rate
grounded in how 4-point disagreements have actually resolved.

The weekly slate emits per game:

| Field |
|---|
| `model_spread`, `market_spread`, `spread_gap` |
| `cover_prob` |
| `model_total`, `market_total`, `total_gap` |
| `over_prob` |
| `edge_flag` (configurable threshold) |

Written as CSV and markdown to `data/processed/`, following the fantasy project's
`cheatsheet.py` output pattern. The schema is designed so a future web app can render it
without rework — the CLI is the deliverable now, the UI is deferred, not precluded.

Bet sizing (Kelly staking) is deliberately **out of scope** for v1. Staking advice is only as
sound as the probability calibration underneath it; it can be added once calibration is
demonstrated in backtest.

## Validation

This section determines whether the model is real. The market is the benchmark, not a
strawman baseline.

**Protocol:** walk-forward by season. Train through season N−1, predict season N, never
peeking forward. Repeat across the NGS era.

**Metrics:**

1. Margin MAE and total MAE, reported **against the market's own MAE** on the same games.
2. ATS hit rate bucketed by edge threshold, with the 52.4% break-even line marked explicitly.
3. Calibration curve and Brier score for `cover_prob` and `over_prob`.
4. **The decisive test** — regress actual margin on both the market line and the model line.
   If the model's coefficient is indistinguishable from zero, the model adds nothing over the
   market. This result is reported plainly whatever it shows.

**Baselines for context:** market-only, home-field-only, prior-season record.

**Leak tests.** pytest assertions that no future information reaches any feature — that
ratings for week N use only data through week N−1, and that no target-derived column survives
into the feature matrix. This is the failure mode that makes a betting model look excellent
and be worthless, so it is tested directly rather than assumed.

## Expected outcome

NFL closing lines are the most efficient market in US sports. A realistic good result is
**matching** the market's MAE overall while finding genuine edges in a narrow subset — early
season, where NGS stabilizes faster than slow-moving market priors; weather games; and QB
changes. A model that appears to beat the line across the board is far more likely to have a
data leak than an edge, which is why the leak tests and the coefficient test above are part of
the design rather than an afterthought.

## Build order

1. Repo scaffold, packaging, and data layer
2. EPA ratings with opponent adjustment
3. NGS offensive layer
4. Margin and total models, plus the backtest harness
5. Calibration and market comparison
6. Weekly slate CLI

## Out of scope for v1

- Kelly staking / bankroll management
- Web app (schema accommodates it; implementation deferred)
- Moneyline and player props
- Live / in-game line movement
- Multi-book line shopping (uses the single consensus line in `load_schedules`)
