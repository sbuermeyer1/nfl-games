# NFL Performance Tracker — Design

**Date:** 2026-08-01  
**Status:** Approved

## Goal

Add a transparent performance tracker to the existing NFL Game Model website. The tracker
will report the official Ridge model's performance against the spread and over/under by
season and overall, distinguish qualified picks from all predictions, and expose the
game-level record behind every percentage.

Historical walk-forward backtests and genuinely published live picks are different kinds of
evidence. They must remain visibly and computationally separate. Historical records will be
available in this phase. The live interface and data contract will be ready for 2026, while
the workflow that freezes and later grades 2026 predictions belongs to the separate 2026-data
phase.

## Scope

### Included

- A packaged, immutable, game-level performance ledger generated offline.
- Historical Ridge walk-forward results for the validated 2021–2025 backtest window.
- Historical season and overall ATS and O/U records.
- Headline records for qualified picks with an absolute model edge of at least 2 points.
- Supporting records for every prediction with a non-zero model edge.
- Cumulative ATS records at absolute spread edges of 5+, 10+, and 15+ points.
- W–L–P, graded sample size, and win percentage for every displayed record.
- A separate live tracker view that initially states that live tracking begins in 2026.
- A ledger contract and interface placeholders for published-line records and closing-line
  metrics: average closing-line value, percentage beating the close, and W–L–P against the
  closing line.
- A game-level audit table for the selected historical season.
- Read-only tracker API routes protected by the existing website authentication boundary.

### Excluded

- Adding the 2026 schedule, lines, ratings, or predictions.
- Capturing or updating live published and closing lines.
- Scheduled jobs, a database, or server-side data mutation.
- Tracking GBM or allowing visitors to choose which estimator constitutes the official
  record.
- Recomputing backtests in response to a web request.
- Bet sizing, bankroll tracking, or wagering recommendations.
- Rebuilding historical opening-line snapshots that were never recorded.

## Chosen approach

Generate an immutable row-level ledger offline and package it with the deployment. The
website loads, validates, filters, and summarizes the reviewed ledger; it never retrains the
model to display tracker results.

This is preferable to storing summary tables alone because the row-level facts provide an
audit trail and support new groupings without rebuilding the model. It is preferable to
on-demand backtesting because it keeps requests fast and prevents past records from silently
changing after code or model updates.

## Model governance

The tracker represents one official model: Ridge with the established default alpha. GBM
remains available on the weekly slate but has no tracker record.

Every ledger row carries a stable `model_version` in addition to `estimator="ridge"`. The
initial historical artifact uses one version for the current validated model. A deliberate
future model change must use a new version; it must not rewrite already published live rows.
Historical backtests may be regenerated for research under a new version, but versions must
remain distinguishable and the website must not choose the best-performing version after the
fact. The first tracker release exposes only the approved current version.

## Architecture

Add focused modules without changing the existing one-way model dependency flow:

- `nfl_game.tracking.ledger` defines the ledger schema, builds and grades rows from
  predictions, and validates persisted rows.
- `nfl_game.tracking.summary` filters one record type and season selection, then calculates
  records, win percentages, edge groups, and closing-line metrics.
- `scripts/build_tracker.py` creates the historical ledger from the packaged game-feature
  dataset using `walk_forward(..., estimator="ridge")` for 2021–2025.
- `nfl_game.web.tracker_service` loads the ledger once, validates request selections, and
  returns tracker summaries and audit rows.
- `nfl_game.web.tracker_page` owns the framework-free tracker HTML, CSS, and JavaScript so the
  existing application page does not become responsible for two interfaces.
- `nfl_game.web.app` only wires the new page and API routes into the existing authenticated
  application.

The data flow is:

`game features -> offline Ridge walk-forward -> reviewed ledger artifact -> tracker service -> API -> browser`

Nothing under `tracking` imports from `nfl_game.web`. The web application cannot write to or
replace the artifact.

## Ledger contract

Use one schema for historical and future live rows. Required groups of fields are:

### Identity

- `record_type`: `backtest` or `live`
- `model_version`
- `estimator`, required to be `ridge` for the official ledger
- `game_id`, `season`, `week`, `away_team`, `home_team`

The tuple `(record_type, model_version, game_id)` is unique.

### Forecast

- `model_margin`
- `model_total`

Spread values use the repository's internal convention throughout: positive means the home
team is favored by that many points.

### Lines and timestamps

- `official_spread_line`, `official_total_line`
- `published_spread_line`, `published_total_line`
- `closing_spread_line`, `closing_total_line`
- `published_at`, `kickoff_at`

For a backtest row, the closing lines are also the official grading lines; published lines
and `published_at` are null because no contemporaneous snapshot exists. For a live row, the
official lines equal the frozen published lines and never change. Closing lines may be null
until captured after the market closes.

### Outcomes and derived audit fields

- `actual_margin`, `actual_total`
- frozen `spread_pick` (`home` or `away`) and `total_pick` (`over` or `under`)
- official spread and total edge sizes
- official spread and total grades (`win`, `loss`, `push`, `pending`, or `no_pick`)
- spread and total CLV when both published and closing lines exist
- spread and total grades against the closing line while retaining the frozen original pick

The builder writes the derived fields for easy inspection. The loader independently
recomputes and verifies them from the stored facts so stale or hand-edited grades fail
validation rather than reaching the website.

## Pick and grading rules

### Direction

- Spread pick is `home` when `model_margin > official_spread_line` and `away` when it is
  lower.
- Total pick is `over` when `model_total > official_total_line` and `under` when it is lower.
- Exact equality produces `no_pick`; it is not forced to one side.

### Official grade

- Backtests are graded against their closing/official line.
- Live picks are graded against the frozen published/official line.
- A result exactly equal to the relevant line is a push.
- A row without a required line or final outcome is pending and is not graded.

### Records and win percentage

Every summary returns wins, losses, pushes, `n_graded = wins + losses`, and
`win_rate = wins / n_graded`. Pushes are displayed but do not enter the win-rate
denominator. Empty records return a null win rate rather than zero percent.

### Qualified and all-prediction records

- A qualified spread pick has `abs(model_margin - official_spread_line) >= 2.0`.
- A qualified total pick has `abs(model_total - official_total_line) >= 2.0`.
- All-prediction records include every otherwise gradeable pick with a non-zero edge.
- A value exactly on a threshold is included.

### Cumulative spread-edge records

The tracker calculates independent, cumulative records at 5+, 10+, and 15+ points of
absolute spread edge. These are not exclusive bands: every 15+ pick is also included in the
10+ and 5+ records. The displayed sample size is mandatory for each threshold.

## Closing-line metrics for live picks

The frozen published-line record remains the official record. Closing-line measurements are
secondary and cannot alter it.

CLV is oriented so positive always means the published line was better for the frozen pick:

- Spread CLV is `(closing_spread - published_spread)` for a home pick and the negative of
  that difference for an away pick.
- Total CLV is `(closing_total - published_total)` for an over and the negative of that
  difference for an under.

For spread and total separately, the live section reports:

- average CLV in points over rows with both snapshots;
- percentage beating the close, defined as the share with CLV greater than zero among all
  rows with calculable CLV; a zero CLV does not beat the close;
- W–L–P for the original frozen pick graded at the closing line.

The historical backtest section does not report CLV because it has closing lines but no
honestly captured published lines. Missing live closing lines remain pending only for
closing-line metrics and do not affect the official record.

## Web interface

Add a visible navigation choice between the existing weekly slate and a new `/tracker`
page. Keep the current framework-free visual language and responsive behavior.

The tracker page contains:

1. `Historical backtest` and `Live record` tabs that never combine their totals.
2. A season selector with `Overall (2021–2025)` and each available historical season.
3. Headline ATS and O/U cards for qualified 2+ point picks, each showing W–L–P, win rate,
   and graded sample size.
4. Supporting all-prediction ATS and O/U records.
5. A cumulative spread-edge table with 5+, 10+, and 15+ records.
6. In the overall view, a season-by-season comparison table.
7. For a selected season, a game-level audit table with matchup, model value, line, pick,
   edge, final outcome, and grade.
8. Clear labels stating that the historical record is a walk-forward backtest against
   closing lines and the live record begins in 2026.

The live tab initially shows the 2026 start message instead of empty zero-percent cards. Its
future populated state also includes average CLV, percentage beating the close, and the
secondary closing-line record for both spreads and totals.

The page marks 52.4% as the standard -110 break-even reference but continues to describe the
output as model tracking, not betting advice. Missing values render as `n/a`.

## Read-only routes

- `GET /tracker` returns the tracker page.
- `GET /api/tracker/options` returns available record types, historical seasons, the current
  official model version, qualification threshold, cumulative edge thresholds, and live
  availability.
- `GET /api/tracker/summary?record_type=backtest&season=all|YYYY` returns the headline,
  supporting, edge-threshold, closing-line, and season-breakdown summaries applicable to the
  selection.
- `GET /api/tracker/games?record_type=backtest&season=YYYY` returns the audit rows for one
  concrete season. `all` is rejected on this endpoint to avoid returning the full history
  when the overall interface only needs the season summary table.

All routes except the existing public health and login routes remain behind the existing
authentication middleware. API errors use the existing client-safe JSON boundary.

## Artifact workflow

The historical build command reads `data/processed/game_features.parquet`, runs Ridge
walk-forward predictions for 2021–2025, grades them, validates the complete ledger, and writes
`data/processed/tracker_ledger.parquet`.

The tracker artifact is a deliberate checked-in deployment input, like the game-features
artifact. Updating it requires an explicit offline rebuild, tests, review of the artifact
diff/summary, commit, and redeployment. Website visitors cannot trigger a rebuild.

The build must reproduce the established all-prediction acceptance corpus before the
artifact is accepted: 1,359 games, ATS `n=1326` with hit rate `0.4977`, and O/U `n=1348` with
hit rate `0.5022`, subject only to the explicit `no_pick` rule if an exact zero model edge is
found. Any discrepancy is an investigation gate, not an automatic baseline update.

## Validation and error handling

- A missing, unreadable, empty, or schema-invalid tracker artifact stops startup with a clear
  server-side error.
- Duplicate identity keys, a non-Ridge official row, an unknown record type, invalid team or
  grade values, non-finite required numbers, or derived fields that disagree with recomputed
  values fail validation.
- Historical rows must use their closing lines as official lines and must not claim a
  published timestamp.
- Live rows must have frozen published lines equal to official lines before they can become
  official picks.
- Invalid record type or season selections return client-safe validation errors.
- A valid selection with no graded results returns zero counts and null percentages, not a
  fabricated 0% record.
- The live tab receives an explicit unavailable state until the future 2026 artifact contains
  live rows.
- Unexpected exceptions are logged server-side and returned without stack traces, local
  paths, or ledger internals.

## Testing

### Grading unit tests

- Home, away, over, and under wins and losses.
- Exact spread and total pushes.
- Exact zero model edges producing `no_pick`.
- Missing lines and unfinished games remaining pending.
- Exact 2, 5, 10, and 15 point boundary inclusion.
- Cumulative membership, including every 15+ pick in the 10+ and 5+ groups.
- Win-rate denominators excluding pushes and pending rows.
- Spread and total CLV signs for both pick directions.
- Closing-line grades retaining the original frozen pick.

### Ledger and summary tests

- Schema and value validation, uniqueness, and recomputation of derived fields.
- Historical and live records never sharing an aggregate.
- Overall historical results equal the aggregation of displayed seasons.
- Edge-group sample sizes are monotonic: 15+ cannot exceed 10+, which cannot exceed 5+.
- Empty and unavailable selections return their explicit states.
- The historical all-prediction summary matches the established backtest acceptance corpus.

### Web tests

- Authentication protects the tracker page and APIs under the existing policy.
- Options, valid summaries, concrete-season audit rows, and invalid inputs return the expected
  status and safe response shapes.
- `all` is rejected for the game-audit endpoint.
- Historical/live tabs and season changes discard stale requests using the page's existing
  request-invalidation pattern.
- Responsive rendering, missing-value formatting, labels, and the live-unavailable message
  remain present.
- Existing slate routes, CSV behavior, authentication, and statistical baselines remain
  unchanged.

## Release acceptance

The tracker phase is complete when:

1. The reviewed historical artifact covers 2021–2025 Ridge walk-forward predictions.
2. Historical overall and per-season ATS/O-U records, all-prediction context, and cumulative
   5+/10+/15+ spread records render from the artifact.
3. A selected season exposes the game-level audit rows behind the summaries.
4. The live tab clearly announces the 2026 start and contains no fabricated metrics.
5. The complete automated suite, style checks, and existing 2021–2025 backtest invariant
   pass.
6. The packaged deployment remains read-only and fail-closed under its existing authentication
   and artifact-validation policies.

