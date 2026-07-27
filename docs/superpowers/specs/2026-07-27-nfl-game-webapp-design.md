# NFL Game Model Web App — Design

**Date:** 2026-07-27
**Status:** Approved

## Goal

Deploy the existing NFL Game Model as an access-code-protected web application at
`https://nfl.ashburn-capital.com`. The application will follow the same operational
structure as the NFL Fantasy Football Model's deployed draft app: FastAPI, a simple
shared-code login, Docker packaging, and a Render Blueprint.

Version one is a read-only weekly slate dashboard. It uses a packaged, prebuilt
`game_features.parquet` dataset and never rebuilds upstream data on the server.

## Scope

### Included

- Fail-closed shared access-code login with server-side sessions.
- Historical season and week selection from the packaged dataset.
- Estimator selection between the existing `ridge` and `gbm` implementations.
- Configurable spread-edge threshold, defaulting to `2.0`.
- Latest available season/week as the initial selection.
- The existing model-versus-market slate presented as a responsive HTML table.
- CSV download of the currently selected slate.
- Docker and Render deployment files.
- Documentation for connecting the Render service to `nfl.ashburn-capital.com`.

### Excluded

- Server-side data ingestion or dataset rebuilding.
- Scheduled refresh jobs.
- User accounts or per-user saved preferences.
- Editing market lines or model features through the browser.
- A separate JavaScript frontend or design framework.
- Changes to the statistical model, calibration logic, or established backtest baseline.

## Chosen Approach

Add the web application directly to the `nfl-games` repository and mirror the fantasy
project's proven structure. The two repositories remain independently deployable.

This intentionally duplicates a small amount of authentication and deployment code.
Extracting shared infrastructure would couple the repositories and create a broader
refactor with no user-facing benefit for this version. A separate frontend is also
unnecessary for the bare-bones dashboard.

## Architecture

Add an `nfl_game.web` package with focused modules:

- `app.py` creates the FastAPI application, registers routes, and serves the embedded
  HTML page.
- `auth.py` supplies access-code middleware, secure session-cookie handling, and the
  environment-variable contract.
- `login.py` supplies the login page and throttled login route.
- `sessions.py` stores opaque session tokens in memory.
- `service.py` owns dataset options, model/calibrator caching, slate generation, and CSV
  serialization.

Add `scripts/game_app.py` as the production entry point. It reads
`data/processed/game_features.parquet` once at startup, constructs the service and app,
adds authentication, and starts Uvicorn on the platform-provided `PORT`.

The modules have one-directional dependencies:

`existing model/market code -> web service -> FastAPI routes -> embedded browser page`

The existing model and market packages must not import from `nfl_game.web`.

## Model Lifecycle and Caching

Generating a slate requires:

1. Walk-forward predictions from prior seasons to fit the calibrator.
2. A model trained on every season before the selected season.
3. Predictions for the selected season/week.
4. Existing `build_slate` logic to join predictions, probabilities, and market lines.

The service caches fitted model/calibrator pairs by `(season, estimator)`. The ridge
penalty remains the existing validated default and is not exposed in version one.
Changing only the selected week or edge threshold therefore does not repeat historical
walk-forward fitting.

The edge threshold is applied when building the returned slate and does not alter the
trained model. Cache access must be safe under concurrent requests. All cached objects
are process-local and may be discarded on a Render restart without losing user data.

## Routes

- `GET /login` renders the shared-code login page.
- `POST /login` validates the submitted code and issues a secure session cookie.
- `GET /` renders the slate dashboard.
- `GET /api/options` returns available seasons, weeks for the active/default season,
  estimators, the default edge threshold, and the latest available selection.
- `GET /api/weeks?season=Y` returns weeks available for one valid season.
- `GET /api/slate?season=Y&week=W&estimator=E&edge_threshold=T` returns the selected
  slate as JSON.
- `GET /api/slate.csv?season=Y&week=W&estimator=E&edge_threshold=T` returns the same
  selected slate as a downloadable CSV.
- `GET /health` returns a minimal readiness response after the dataset has loaded.

JSON and CSV routes call the same service method so their rows, order, edge flags, and
missing-value handling cannot diverge.

## Authentication

The authentication UI and session behavior mirror the fantasy application, but the
startup policy is deliberately stricter:

- `ACCESS_CODE` set: all dashboard and API routes except login and health require a valid
  opaque session cookie.
- `ACCESS_CODE` unset: production/default startup fails with a clear configuration error.
- Unauthenticated local development requires the explicit `scripts/game_app.py --no-auth`
  flag. The flag binds to loopback by default and is never present in the Docker command.
- Successful login sets a secure, HTTP-only, same-site cookie.
- Login attempts are throttled using the fantasy app's established behavior.
- Session tokens and their expiry are held in memory; a restart signs visitors out.
- No access code or session token is written to logs or returned by an API.

The Render service receives `ACCESS_CODE` as a private, non-synchronized environment
variable. It may use the same human-entered code as the fantasy site, but the setting is
independent and can be rotated separately.

## Interface

The page remains deliberately small and framework-free. It contains:

- title and a brief model description;
- season selector;
- week selector;
- estimator selector (`ridge` or `gbm`);
- numeric edge-threshold control with a `2.0` default;
- **Run slate** button;
- **Download CSV** button;
- loading and error message area;
- one results table.

On load, the page selects the latest season/week in the packaged dataset and the ridge
estimator, then requests that slate. Changing the season refreshes the available weeks.
The run button is disabled while a request is active to prevent duplicate submissions.

The table uses the existing slate schema and sort order:

1. matchup (`away @ home`);
2. model spread;
3. market spread;
4. spread gap;
5. cover probability;
6. model total;
7. market total;
8. total gap;
9. over probability;
10. edge flag.

Missing market lines or probabilities display as `n/a`. The table scrolls horizontally
on narrow screens. A short note states that spreads are home-team margins and that an
edge flag is model/market disagreement, not betting advice.

The HTML, CSS, and JavaScript may be embedded in the Python package, matching the
fantasy application's structure and avoiding a frontend build system.

## Validation and Error Handling

- A missing `ACCESS_CODE` stops default/production startup with a clear configuration
  error; only the explicit local `--no-auth` mode bypasses the gate.
- A missing, unreadable, or schema-invalid packaged dataset stops application startup
  with a clear server-side error.
- Season and week must exist together in the dataset.
- Estimator must be one of the existing registry values.
- Edge threshold must be a finite, non-negative number.
- A season with no usable calibration corpus returns a specific client-safe error.
- `DegenerateFeatureError` becomes a client-safe model-training error.
- An empty selected slate returns a readable not-found response.
- Existing missing-line behavior is preserved as null JSON values and `n/a` in HTML.
- Unexpected exceptions are logged server-side and returned to the browser as a generic
  failure without stack traces or internal paths.

## Packaged Data and Refresh Workflow

`data/processed/` is ignored by default, so deployment must deliberately include exactly
`data/processed/game_features.parquet` as a tracked deployment artifact (using a narrow
gitignore exception or an explicit forced add). Other generated slates, backups, and raw
data remain ignored.

The operational refresh workflow is:

1. Run `scripts/build_dataset.py` locally for the desired season range.
2. Run the existing tests/backtest checks as appropriate.
3. Replace the packaged `game_features.parquet`.
4. Commit and push the updated artifact.
5. Redeploy the Render service.

No visitor can trigger this workflow from the website.

## Deployment

Add:

- FastAPI and Uvicorn runtime dependencies;
- a Dockerfile modeled on the fantasy repository;
- a `render.yaml` Blueprint defining one Docker web service, tentatively named
  `ashburn-nfl`, with `ACCESS_CODE` marked `sync: false`.

The Docker image includes package metadata, `src/`, the web entry point, and the single
packaged parquet artifact. It reads Render's injected `PORT` instead of hardcoding a
platform port.

After the service is live on its Render hostname:

1. From an external client with no cookies, verify `/` redirects to `/login` and
   `/api/options` returns `401`.
2. Verify an incorrect access code is rejected and the configured access code succeeds.
3. Restart/redeploy the service and repeat the unauthenticated checks. DNS cutover is
   blocked until these checks pass.
4. Add `nfl.ashburn-capital.com` as a custom domain in Render.
5. Add the CNAME record Render provides in the DNS manager for `ashburn-capital.com`.
6. Wait for Render's domain verification and managed TLS certificate.
7. Repeat the external authentication checks on the custom domain, then verify the latest
   slate and CSV download.

DNS and Render dashboard changes are operational steps; repository code cannot perform
them by itself.

## Testing

### Service tests

- Available seasons/weeks and latest defaults are derived correctly.
- Invalid season/week pairs, estimators, and thresholds are rejected.
- Model/calibrator cache keys distinguish season and estimator.
- Repeated weeks in one season reuse the same fitted model/calibrator.
- Generated slates match the existing CLI path for identical inputs.
- Missing market lines remain null and never render as `nan`.
- CSV rows and ordering match the JSON/HTML source slate.

### Web tests

- Default startup fails when `ACCESS_CODE` is absent.
- Explicit local `--no-auth` mode starts without login and binds to loopback.
- Protected routes redirect or reject unauthenticated requests when it is present.
- Correct and incorrect login attempts follow the expected session/throttle behavior.
- Options, weeks, slate JSON, CSV headers, download filename, and health endpoints are
  covered.
- Client-safe model/data errors do not expose internal exceptions.

### Completion checks

- Run the full existing pytest suite plus the new web tests.
- Run Ruff across `src`, `scripts`, and `tests`.
- Build the Docker image.
- Start the container locally with an access code and verify health, login, dashboard
  load, slate generation, and CSV download.
- Verify the explicit non-container local `--no-auth` mode separately.
- Confirm the existing statistical backtest baseline is unchanged; this feature must not
  modify model behavior.

## Success Criteria

- `nfl.ashburn-capital.com` presents the access-code login and then the weekly dashboard.
- A missing production access code prevents the service from starting rather than
  exposing the dashboard.
- Every packaged historical season/week can be selected.
- Both existing estimators work with the validated default ridge penalty.
- The edge threshold updates the displayed/downloaded edge flags.
- Displayed results preserve the existing model and market conventions.
- CSV downloads exactly match the selected table.
- Data cannot be rebuilt or mutated through the public application.
- The existing model tests and baseline remain unchanged.
