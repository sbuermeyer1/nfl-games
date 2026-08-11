# Ridge v2 Model Upgrade — Design

**Date:** 2026-08-11  
**Status:** Approved design, pending written-spec review and implementation plan

## Context

The production website currently uses the independent Ridge `ridge-v1` game model. It fits
separate regressions for home margin and total points, but both targets consume the same 14
features and use a fixed Ridge penalty. Its opponent-adjusted EPA and NGS foundation is
leak-free and reproducible, but it omits several plausible sources of predictive signal.

The frozen 2021–2025 Ridge-v1 baseline is:

| Metric | Ridge v1 | Closing market |
| --- | ---: | ---: |
| Games | 1,359 | — |
| Margin MAE | 10.274 | 9.752 |
| Total MAE | 10.684 | 10.309 |
| ATS hit rate | 0.4977 (n=1,326) | — |
| O/U hit rate | 0.5022 (n=1,348) | — |
| Margin model coefficient beside market | -0.0218 | Market coefficient 1.0755 |

Ridge v2 will try to improve this baseline with better data and feature construction, not by
changing to a more complex estimator or allowing the betting market to become a model input.
The work is an evidence-gated challenger experiment: Ridge v1 remains official until v2 passes
all promotion criteria.

## Decisions

- Ridge remains the only estimator.
- Betting spreads and totals remain outside the prediction model. They are benchmarks used
  only after independent predictions exist.
- Margin and total use separate feature schemas and separately tuned Ridge penalties.
- New data must be free, reproducible, and available through stable public sources.
- No paid injury-data provider will be introduced.
- Ridge v1 remains live while Ridge v2 is developed and shadow-tested.
- New signals enter through predefined feature blocks and must demonstrate incremental value.
- Promotion is evidence-first. A historical betting hit rate above break-even is not required,
  because selecting features directly for ATS/O/U profitability would invite overfitting.
- Ridge-v1 and Ridge-v2 records are never combined into one tracker win rate.

## Goals

1. Improve out-of-sample margin and total accuracy over Ridge v1.
2. Demonstrate positive information beyond the closing line for both targets.
3. Preserve or improve ATS, O/U, and probability-calibration performance without optimizing
   features directly against a small profitable-looking betting bucket.
4. Expand the data foundation while retaining strict as-of correctness and reproducibility.
5. Make it possible to identify which data blocks help, hurt, or add no value.
6. Support reliable 2026 production refreshes without a paid feed.

## Non-goals

- Gradient boosting, neural networks, ensembles, or other non-Ridge estimators
- Feeding a spread, total, moneyline, or market-derived probability into Ridge
- Paid injury data
- Scraping an unsupported injury source
- Bet sizing or bankroll management
- In-game predictions
- Optimizing against a single season or a small edge bucket
- Rewriting Ridge-v1 backtests or previously published live picks
- Making FTN charting a requirement for the initial Ridge-v2 release

## Architecture

Ridge v2 is a parallel challenger inside the existing one-directional architecture:

```text
public data sources
        ↓
source-specific normalized caches
        ↓
as-of team, quarterback, style, and personnel ratings
        ↓
game feature artifact
        ↓
margin Ridge + total Ridge
        ↓
independent predictions
        ↓
market comparison, calibration, tracker, and website
```

The production code continues to respect `data → ratings → model → market`. Source-specific
modules normalize team and player identities at ingestion. Rating and feature modules never
import market data.

### Model interface

`GameModel` will retain one public fit/predict interface while using two explicit schemas:

- `MARGIN_FEATURE_COLS`
- `TOTAL_FEATURE_COLS`

Both pipelines use the existing robust standardization guard followed by Ridge. Margin and
total each receive their own selected `alpha`; rating-recency settings are also selected only
from earlier data within each outer fold.

## Data sources

### Required existing sources

- nflverse schedules for game identity, kickoff, venue context, results, and later market
  comparison
- nflverse play-by-play for team strength, situational efficiency, style, turnovers, field
  position, and special teams
- NFL Next Gen Stats for passing, rushing, and receiving quality

### New free sources

- nflverse player and team weekly statistics
- weekly rosters
- depth charts, using their point-in-time timestamp from 2025 onward
- snap counts
- PFR advanced passing, rushing, receiving, and defensive summaries through nflverse

`nflreadpy` exposes loaders for these datasets. nflverse documents daily or post-game refreshes
for play-by-play, stats, rosters, depth charts, snap counts, NGS, and PFR data:

- <https://github.com/nflverse/nflreadpy>
- <https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html>

### Injury limitation

nflverse's injury source ended after the 2024 season and currently provides no dependable
2025+ production feed. Historical injuries may be used only in a clearly labeled research
analysis. They cannot enter a production-required feature, because that model could not be
rebuilt consistently for 2026.

Availability will instead be represented by observable free signals: expected starter,
depth-chart movement, roster movement, prior snap share, and uncertainty flags. These are
proxies, not claims that a complete injury report is available.

### Short-history FTN data

FTN charting begins in 2022. It includes motion, play action, RPO, screens, pressure structure,
drops, catchable passes, interception-worthy passes, and related charted fields. Its short
history gives too few independent seasons for the initial production candidate. It will be a
separate experiment and will not be required for Ridge-v2 promotion.

- <https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html>

## As-of data contract

Every source row used for game `G` must have been knowable before `G`'s prediction cutoff.
The implementation must distinguish:

- **Event time:** when a game, play, roster move, or depth-chart state occurred.
- **Availability time:** when the source made that information available.
- **Prediction cutoff:** the instant after which an official prediction cannot change.

Historical feature generation simulates the same rule used in production. Post-game
performance includes only previously completed games. Point-in-time depth-chart rows must
have timestamps no later than the prediction cutoff. When a historical source lacks a true
availability timestamp, its documented update cadence supplies a conservative availability
rule rather than assuming immediate availability.

Official live predictions retain the existing publication policy: model values become
immutable when published 24 hours before kickoff.

## Feature blocks

Feature blocks are declared before the outer backtest. Each block has its own formulas,
coverage report, missingness flags, and isolated/cumulative evaluation.

### Block 1: improved team ratings

This block rebuilds the EPA foundation without changing its opponent-adjusted, as-of nature.

- Overall, passing, and rushing EPA
- Offensive and defensive success rate
- Early-down EPA
- Neutral-game-state EPA, using a fixed predeclared score/win-probability filter
- Explosive pass and rush creation/prevention rates
- Sack/pressure proxies available in play-by-play
- Short-term and long-term rating windows
- Opponent adjustment
- Preseason regression toward a prior-season rating and league mean

Short- and long-term ratings remain separate inputs. Ridge can weight recent form against a
more stable prior rather than forcing one global half-life to represent both.

### Block 2: quarterback context

- Rolling QB EPA per dropback
- Rolling CPOE
- Sack avoidance
- Interception rate with small-sample regression
- Expected starter identity
- Difference between expected-starter quality and the team's recent starter quality
- Home-minus-away quarterback quality for margin
- Combined quarterback quality for total
- Rookie, backup, new-starter, and uncertain-starter flags

A starter change is represented by estimated quality loss/gain as well as a flag. If the
expected starter cannot be established from the free sources, the feature uses a documented
neutral estimate and exposes uncertainty to Ridge.

### Block 3: game style and hidden yardage

- Neutral-situation pass tendency
- Pace and expected play volume
- Explosive-play tendency
- Field-position contribution
- Special-teams EPA
- Penalty tendency when coverage and definitions are stable
- Interception and fumble outcomes separated from their more repeatable underlying rates

Turnover features must regress small samples toward league expectation. Raw turnover margin
cannot be treated as fully repeatable team quality.

### Block 4: personnel continuity

- Returning offensive and defensive snap share
- Position-group continuity
- Depth-chart movement
- Roster churn
- Recent snap concentration
- Explicit data-availability and uncertainty flags

For an upcoming week, all snap-derived values stop at the most recently available completed
game. Preseason and Week 1 features combine prior-season snaps with current roster/depth-chart
identity; they do not use future Week 1 participation.

### Block 5: PFR advanced data

PFR weekly advanced data is available from 2018 onward. Candidate aggregates include:

- Pressure faced and created
- Time in pocket
- On-target, bad-throw, throwaway, and drop rates
- Play-action and RPO efficiency
- Rushing yards before/after contact and broken-tackle signals where available
- Receiving drop and yards-after-catch signals
- Defensive pressure and disruption

This block is optional until its team-week coverage and 2026 refresh reliability pass the same
checks as the core blocks. Source licensing and attribution requirements must be documented
before production use.

## Target-specific schemas

The exact columns are finalized before the outer evaluation, but their roles are fixed here.

### Margin model

The margin schema emphasizes differences between the two teams:

- Home-minus-away overall, pass, and rush strength
- Matchup edges: each offense against the opposing defense
- Home-minus-away quarterback quality and starter-change effect
- Rest differential and home-field context
- Personnel-continuity differential
- Special-teams and field-position differential
- Style-matchup terms that can move one team relative to the other

### Total model

The total schema emphasizes combined scoring conditions:

- Sum of both offensive matchup strengths against the opposing defenses
- Combined quarterback quality and starter uncertainty
- Expected pace and play volume
- Combined neutral pass tendency and explosive-play potential
- Combined pressure/sack environment
- Dome, temperature, and wind
- Combined special-teams scoring/field-position contribution
- Total-specific PFR aggregates

The two schemas may share underlying ratings, but they are not required to contain identical
columns or choose identical penalties.

## Candidate ladder

The initial experiment matrix is cumulative, with companion ablations:

| Candidate | Included data |
| --- | --- |
| C0 | Exact Ridge-v1 baseline |
| C1 | C0 plus improved team ratings and target-specific schemas |
| C2 | C1 plus quarterback context |
| C3 | C2 plus game style, turnovers, field position, and special teams |
| C4 | C3 plus personnel continuity |
| C5 | C4 plus PFR advanced data |
| E1 | Best core candidate plus FTN charting; research only |

Each cumulative candidate is compared with a remove-one-block ablation where the training
history supports it. This makes attribution visible without testing arbitrary combinations of
individual features.

## Nested walk-forward training

The outer evaluation seasons remain 2021–2025. For each outer season `S`:

1. Construct all eligible features using only information available before each game.
2. Restrict model training and configuration evidence to seasons before `S`.
3. Run time-ordered inner validation within those earlier seasons.
4. Choose the eligible candidate block set, margin alpha, total alpha, and predefined rating
   settings using only inner validation.
5. Refit on all eligible data before `S`.
6. Predict every eligible game in `S` exactly once.

The selected configuration may evolve as additional seasons become available. That is part of
the declared algorithm, not post-hoc tuning. The report records the configuration selected for
every outer season.

The implementation plan must define a small finite grid before results are generated. It may
cover separate Ridge penalties, short/long rating half-lives, and preseason shrinkage. It must
not grow into an open-ended hyperparameter search.

## Block eligibility

A data block may enter a production candidate only if:

- It can be rebuilt for every required 2021–2025 outer prediction without future information.
- Aggregated team-week numeric coverage is at least 90% in every included season before
  imputation. A lower-coverage block remains experimental.
- Missingness is represented explicitly and uses a documented neutral estimate.
- Team/player joins are unique and identity coverage is reported.
- A source outage cannot silently alter the model schema.
- The same source and transformation can run for 2026.

The 90% threshold applies to the block's final team-week aggregates, not to every player row.
Qualifier-based player datasets may legitimately omit non-qualifiers before aggregation.

## Source freshness and failure behavior

Every cache write records source, seasons, retrieval time, schema fingerprint, row count, and
coverage. A refresh is rejected if it introduces duplicate identities, impossible timestamps,
non-finite model inputs, an unexpected schema change, or a material unexplained coverage drop.

- Rosters and depth charts used for a new prediction must have a valid snapshot no more than
  72 hours old.
- Trailing performance feeds must include every completed game whose result has been available
  for at least 48 hours before the prediction cutoff.
- A last-known-good snapshot may be reused only while it satisfies those rules.
- If a required v2 source is stale, the refresh fails and does not replace the last valid
  artifact. It does not silently remove the block, substitute Ridge v1, or label stale output
  as newly refreshed Ridge v2.

## Evaluation report

Every candidate is scored on the identical valid game set used by Ridge v1. Reports include
overall and per-season results.

### Accuracy

- Margin MAE and RMSE
- Total MAE and RMSE
- Paired per-game absolute-error difference versus Ridge v1
- Closing-market MAE on the same rows

### Incremental market value

Two joint regressions are reported:

- Actual margin on closing spread and independent model margin
- Actual total on closing total and independent model total

The model coefficient measures whether Ridge v2 contributes information after the line is
already present. Standard errors and uncertainty are clustered/resampled by season-week.

### Betting and calibration

- ATS and O/U hit rates, excluding pushes
- All predictions and cumulative 2+, 5+, 10+, and 15+ edge cohorts
- Wins, losses, pushes, non-push sample size, and uncertainty interval for every cohort
- Cover and over Brier scores
- Reliability tables/curves
- Calibration slope and intercept

Edge-cohort results are diagnostics, not feature-selection targets.

## Promotion gates

Ridge v2 is promoted only if all gates pass on the complete nested outer prediction corpus:

1. Margin MAE is strictly lower than Ridge v1's 10.274 on the same games.
2. Total MAE is strictly lower than Ridge v1's 10.684 on the same games.
3. Margin MAE improves in at least three of the five outer seasons.
4. Total MAE improves in at least three of the five outer seasons.
5. For both targets, the mean paired absolute-error improvement has a positive lower bound in
   a one-sided 90% season-week block-bootstrap interval using 10,000 resamples and a fixed seed.
6. The Ridge-v2 coefficient in each joint market regression is positive. Its one-sided 90%
   season-week block-bootstrap lower bound must also be positive.
7. Overall ATS hit rate is no more than one percentage point below Ridge v1's 0.497738.
8. Overall O/U hit rate is no more than one percentage point below Ridge v1's 0.502226.
9. Cover and over Brier scores are each no worse than their Ridge-v1 value on the same
   out-of-sample rows.
10. All correctness, availability, determinism, and source-reliability checks pass.
11. A shadow-production rebuild succeeds without changing Ridge-v1 artifacts or official
    tracker records.

No edge cohort is required to exceed the 52.4% break-even rate. Small cohorts are reported but
cannot override the primary gates.

If any gate fails, Ridge v1 remains official. The candidate results and block-level findings
remain useful research, but no production label changes.

## Calibration

Calibration remains downstream of the point models. For every outer season, the calibrator is
fit only on earlier out-of-sample predictions, never on in-sample fitted values or on that
outer season's outcomes. Margin and total calibration are evaluated separately.

After a candidate passes all gates, the 2026 production calibrator is fit on the complete
eligible 2021–2025 out-of-sample Ridge-v2 prediction corpus.

## Testing

### Data and feature tests

- Loader schema and conversion tests for each new nflreadpy source
- Team and player identity normalization tests
- One-row-per-key and join-cardinality tests
- Coverage and missingness reports with threshold tests
- Formula tests for every derived feature
- Timezone and availability-time tests
- Source-staleness and last-known-good tests

### Leak tests

- Adding or changing a future game cannot alter an earlier game's features.
- Adding a future roster or depth-chart snapshot cannot alter an earlier prediction.
- A week `N` performance feature uses no game from week `N` or later.
- A selected starter uses only a snapshot available by the prediction cutoff.
- Inner tuning never reads the outer season.
- Calibration never reads the season it is calibrating.
- Market columns cannot enter either Ridge feature matrix.

### Model tests

- Ridge-v1 exact regression baseline before any v2 comparison
- Independent margin and total schemas
- Independent alpha selection
- Robust scaler and degenerate-feature behavior
- Deterministic selection and predictions under a fixed seed
- Candidate/ablation accounting
- Metric calculations and block-bootstrap reproducibility

### End-to-end tests

- Full historical rebuild from cached sources
- Nested 2021–2025 backtest
- 2026 feature refresh
- Shadow website predictions
- Artifact schema/digest reporting
- Proof that Ridge-v1 features and tracker ledger remain unchanged

## Artifacts and versioning

Ridge-v1 artifacts remain frozen and retain their current names and exact acceptance metrics.
Ridge-v2 writes separate challenger artifacts during research. The implementation plan will
choose final filenames, but the logical separation is mandatory:

- Ridge-v2 features/configuration manifest
- Ridge-v2 outer-fold predictions
- Ridge-v2 evaluation and ablation report
- Ridge-v2 calibration parameters
- Ridge-v2 historical tracker ledger, created only after promotion approval

Every artifact records a model version, feature-schema version, source manifest digest, and
build timestamp.

## Website and tracker rollout

Before promotion, Ridge v2 is not shown as official on the public tracker. Shadow results may
be reviewed locally or in an operator-only report.

After all gates pass and the user explicitly approves promotion:

- The dashboard defaults to `ridge-v2` while retaining an unambiguous version label.
- Ridge-v1 historical results remain accessible and unchanged.
- Ridge-v2 receives its own historical tracker selection.
- Tracker summaries always filter by model version before aggregating season or overall data.
- Live rows record the exact model version that generated them.
- Previously published live picks are never rewritten under a new version.

If Ridge v2 is approved before the first 2026 official publication, live tracking may start
with v2. If Ridge v1 has already published live picks, the switch creates a dated version
boundary and the records remain separate.

## Delivery stages

1. Reproduce Ridge-v1 and build the nested evaluation harness.
2. Add source loaders, manifests, coverage checks, and as-of contracts.
3. Implement and evaluate improved ratings and target-specific schemas.
4. Add and evaluate quarterback context.
5. Add and evaluate style, turnovers, field position, and special teams.
6. Add and evaluate personnel continuity.
7. Add and evaluate PFR advanced data.
8. Lock the Ridge-v2 candidate matrix and run the complete nested backtest.
9. Produce the promotion report and request a promotion decision.
10. If approved, run shadow production, create v2 artifacts, update the website/tracker, and
    perform the normal staged deployment verification.
11. Evaluate FTN charting separately; it cannot delay or silently alter the core v2 decision.

Each stage preserves a working Ridge-v1 system and produces a reviewable report before the next
block is accepted.

## Success outcome

The ideal outcome is a Ridge-v2 model that passes every promotion gate and begins 2026 live
tracking with better independent forecasts. A valid alternative outcome is that one or more
new data blocks improve a target but the complete candidate does not clear the evidence bar.
In that case Ridge v1 remains official, and the experiment still identifies which signals are
worth carrying into a later candidate. A failed promotion is preferable to deploying a model
whose apparent edge came from leakage, unstable free data, or a small historical betting
cohort.
