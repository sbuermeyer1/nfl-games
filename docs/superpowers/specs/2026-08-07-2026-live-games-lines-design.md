# 2026 Live Games, Betting Lines, and Official Tracker Design

**Date:** 2026-08-07
**Status:** Approved for implementation planning

## Objective

Add the 2026 regular-season schedule to the website, overlay current nflverse spread and
total lines, keep the Ridge model current after completed game days, and activate the 2026
official live tracker. The system must remain market-blind during prediction, reproducible
for audit, and available when the upstream line feed is slow or unavailable.

## Approved product decisions

- Use `nflreadpy.load_schedules()` as the live schedule and betting-line source.
- Cache live lines for five minutes.
- Refresh model features automatically after completed game days.
- Publish each official prediction 24 hours before its scheduled kickoff.
- Freeze the model prediction at publication; freeze spread and total independently.
- Retry a missing market until one hour before kickoff, then exclude only that market.
- Capture final closing lines and game results after the game is final.
- Persist generated features and tracker records as versioned repository artifacts through
  GitHub Actions; the deployed website remains read-only.
- Show model predictions for the current and next scheduled week. Keep the full 2026
  schedule separately browseable.
- Keep official tracking Ridge-only.
- Keep spread edge thresholds overlapping: absolute edge at least 5, 10, and 15 points.

## Existing constraints

The application currently loads immutable `game_features.parquet` and
`tracker_ledger.parquet` artifacts at startup. Historical tracking covers Ridge `ridge-v1`
for 2021-2025. The model feature matrix does not include market lines, and that separation
must remain intact.

The current feature builder derives target weeks from play-by-play. That is insufficient for
future 2026 games because a scheduled week can exist before any 2026 play-by-play exists.
The new pipeline must derive prediction targets from the regular-season schedule while still
using only completed games strictly before each target cutoff.

The nflverse schedule feed exposes future games and market fields including `spread_line`
and `total_line`. Its published update schedule states that game/schedule data refreshes
every five minutes during the season.

## Architecture

The design has three independent data layers.

### 1. Model data

GitHub Actions rebuilds the packaged game-feature artifact daily during the season and on
manual request. The rebuild includes:

- historical rows needed for training and the fixed 2021-2025 acceptance baseline;
- prediction-ready rows for the earliest unplayed regular-season week and the next distinct
  regular-season week;
- as-of team ratings built from completed play-by-play strictly before each target game;
- available trailing NGS values, with the existing explicit imputation behavior when current
  season data is unavailable.

Target week enumeration comes from the schedule, not from play-by-play. The feature builder
must accept explicit target `(season, week)` pairs so Week 1 can use prior-season ratings and
future weeks can be produced before they have play-by-play rows.

Market lines remain output metadata. They must not be added to `FEATURE_COLS`, estimator
training inputs, rating inputs, or calibration features beyond the existing post-prediction
market comparison and probability calibration contracts.

### 2. Live market data

The website uses a dedicated market provider boundary to fetch and normalize 2026 nflverse
schedule rows. It validates game identity, teams, kickoff time, and numeric market values
before returning data to the web service.

The provider maintains a process-wide, concurrency-safe cache with a five-minute TTL and a
single in-flight refresh. A refresh has a short bounded timeout. On a successful refresh, the
cache stores the normalized rows and an observation timestamp. On failure:

1. serve the most recent in-memory observation if one exists;
2. otherwise use the spread and total stored in the packaged feature artifact;
3. expose the fallback as stale rather than presenting it as current.

Spread and total are independent. A missing or invalid spread does not remove a valid total,
and vice versa. Zero is a valid numeric line; missing values remain null.

The market overlay joins to predictions only by validated `game_id` after confirming the
home and away teams match. It can change the displayed market lines, gaps, and market-derived
probabilities, but it cannot change the model margin or model total.

### 3. Official tracker data

GitHub Actions is the only writer of official live records. The website reads the resulting
ledger and cannot publish, revise, or grade official picks.

The existing live ledger key remains unique by:

`(record_type, model_version, game_id)`

The live workflow extends the fact schema with enough state to distinguish a pending market
from one excluded at the deadline. At minimum, each market needs a publication status,
source observation time, and optional exclusion reason. Closing capture also records its
source observation time. Derived picks, edges, grades, and CLV remain recomputable from fact
columns and must never be manually persisted with contradictory values.

## Website behavior

### Week selection

The service determines the earliest unplayed 2026 regular-season week from validated kickoff
timestamps and final-result state. That is the default prediction week. The next distinct
unplayed week is also available. It must not default to the maximum scheduled week.

The full schedule view may show every 2026 regular-season game, but only the current and next
week receive model predictions.

### API output

Each prediction response retains the existing model and comparison fields and adds market
metadata:

- market source (`nflverse` or packaged fallback);
- market observation timestamp;
- independent spread and total status (`live`, `stale`, or `missing`);
- current spread and total values, preserving nulls.

The HTML page shows the observation time, marks stale fallback data clearly, and renders
missing values as an em dash. It must never render `NaN`, infinity, or a fabricated zero.

### Runtime refresh behavior

Ordinary line refreshes do not rebuild model features, mutate official records, commit files,
or trigger a deployment. The next request after cache expiry performs one bounded refresh;
concurrent requests reuse the same refresh result.

## Tracker lifecycle

### Prediction publication

The tracker workflow runs every 15 minutes and selects games whose scheduled kickoff is at
most 24 hours away and still more than one hour away.

On the first eligible run it freezes:

- Ridge model version and estimator;
- model margin and model total;
- teams, season, week, and kickoff timestamp;
- `published_at`.

The frozen prediction cannot change on rerun, feature refresh, schedule-feed correction, or
deployment.

Each market is then handled independently:

- if a valid line is available, freeze it as both the published and official line;
- if it is missing, leave that market pending and retry on later runs;
- at one hour before kickoff, convert any still-missing market to excluded with reason
  `missing_line_at_deadline`;
- never convert an excluded market into an official pick after the deadline.

An excluded market does not enter win-rate, CLV, edge-threshold, or closing-record
denominators. The other market for the same game can still be official.

### Closing and grading

After nflverse marks a game final, the workflow captures:

- actual home margin and total points;
- final nflverse spread and total as closing lines when available;
- closing observation time.

If a final result or closing market is temporarily absent, the record remains pending and is
retried without changing any frozen publication fact. A record still incomplete seven days
after its scheduled kickoff causes the workflow to fail visibly for manual review.

Official ATS and over/under grades use the frozen published line. Pushes remain separate and
are excluded from win-rate denominators. Closing-line grades are stored as a separate view of
the same frozen pick.

CLV is signed so positive always means the published number was better for the selected side:

- home spread pick: `closing_spread - published_spread`;
- away spread pick: `published_spread - closing_spread`;
- over pick: `closing_total - published_total`;
- under pick: `published_total - closing_total`.

The existing closing summary continues to report average CLV, beat-close rate, sample size,
and record against closing lines.

### Edge summaries

Qualified spread and total records continue to require an absolute edge of at least two
points. Spread threshold summaries remain nested, using absolute spread edge at least 5, 10,
and 15 points. They are not mutually exclusive ranges.

### Schedule changes

- Before publication, a kickoff change moves the publication window normally.
- After publication, the frozen prediction and lines remain unchanged; the workflow records
  the revised kickoff separately for lifecycle timing without rewriting publication facts.
- Cancelled games and games that never become final are marked void after manual review and
  excluded from denominators.
- Postponed games retain the same ledger identity and resume lifecycle processing at the new
  kickoff.

## Automation

### Model refresh workflow

The model workflow runs daily during the NFL season and supports `workflow_dispatch`. It:

1. downloads schedule, completed play-by-play, and NGS data;
2. identifies the current and next unplayed regular-season week;
3. rebuilds prediction-ready features in a temporary path;
4. validates schema, uniqueness, finite features, team joins, and strict as-of cutoffs;
5. rebuilds the historical tracker in memory and verifies the fixed 2021-2025 acceptance
   baseline;
6. runs focused and full automated tests;
7. atomically replaces and commits the feature artifact only when its digest changes.

### Tracker workflow

The tracker workflow runs every 15 minutes during the season and supports
`workflow_dispatch`. It:

1. loads the packaged features and current validated schedule feed;
2. advances eligible live records through publication, deadline exclusion, closing capture,
   and grading;
3. combines live rows with the unchanged historical ledger;
4. validates uniqueness, fact/derived consistency, Ridge-only enforcement, and the historical
   acceptance baseline;
5. atomically replaces and commits the ledger only when its digest changes.

Both workflows use a repository concurrency group, least-privilege `contents: write`
permission, bounded execution time, and deterministic generation. They must not amend, force
push, or overwrite a remote change. A failed download, build, validation, or test leaves the
last committed artifacts untouched.

Generated-data commits may trigger deployment, but no-op workflow runs create no commit.

## Error handling and observability

- Runtime market errors are logged without credentials or response bodies that may contain
  sensitive data.
- Website responses expose freshness state, not internal exception details.
- Workflow failures remain visible in GitHub Actions and preserve the last valid deployment.
- Schema drift in nflverse fails validation explicitly rather than silently dropping fields.
- Duplicate games, team mismatches, invalid kickoff timestamps, and non-finite market values
  are hard failures for automation and rejected refreshes at runtime.
- Manual workflow dispatch provides recovery after an upstream outage or corrected source
  record.

## Testing strategy

Unit and integration tests use fixed nflverse-shaped fixtures and a controllable clock. They
must cover:

- schedule normalization, team-code mapping, and schema drift;
- target weeks obtained from schedules rather than play-by-play;
- strict exclusion of current and future results from every rating cutoff;
- current/next-week selection across byes and irregular game days;
- five-minute cache expiry, concurrent refresh coalescing, timeout, stale fallback, and cold
  failure;
- independent missing or invalid spread and total values;
- API and page freshness metadata and null rendering;
- publication at the 24-hour boundary and exclusion at the one-hour boundary;
- independently published markets, idempotent reruns, and immutable prediction facts;
- final grading, pushes, signed CLV, closing-line records, postponements, and void handling;
- exact historical acceptance metrics and nested 5/10/15 spread thresholds;
- workflow dry-run behavior that cannot commit or publish.

A separate smoke test calls the real nflverse feed, validates the current 2026 schema, and
reports data freshness. It does not replace deterministic fixture tests.

## Rollout

### Stage 1: read-only 2026 data

- Generate and review current/next-week 2026 feature rows.
- Deploy live line overlay and freshness UI.
- Run both workflows in dry-run mode.
- Complete a production Docker build and application smoke test.

### Stage 2: official automation

- Review dry-run publication and closing records against source data.
- Enable generated artifact commits.
- Confirm the first official live rows appear separately from historical backtests.
- Keep manual workflow dispatch and rollback to the preceding artifact revision documented.

## Acceptance criteria

The feature is ready when:

1. The website defaults to the earliest unplayed 2026 week and exposes only current/next-week
   model predictions.
2. Valid nflverse lines refresh no more than once every five minutes per app process.
3. Feed failures serve visibly stale packaged lines without changing model predictions.
4. Scheduled future weeks receive leak-free ratings even without same-season play-by-play.
5. An official Ridge prediction is frozen once per game/model version inside the 24-hour
   window.
6. Missing spread and total markets retry independently and become excluded at the one-hour
   deadline.
7. Final results, closing lines, official grades, closing grades, and CLV are reproducible
   from persisted facts.
8. Historical 2021-2025 tracker metrics remain exact.
9. Failed automation cannot replace or partially write a valid artifact.
10. Stage 1 passes full tests, production Docker build, and smoke checks before tracker writes
    are enabled.

## Primary upstream references

- <https://nflreadpy.nflverse.com/api/load_functions/#nflreadpy.load_schedules>
- <https://nflreadr.nflverse.com/reference/load_schedules.html>
- <https://nflreadr.nflverse.com/articles/dictionary_schedules.html>
- <https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html>
