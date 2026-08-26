# NFL Game Model

Predicts NFL game margins and totals from EPA + Next Gen Stats team ratings, then compares
those predictions against the closing spread and total.

The model is **market-blind**: it never sees the betting line when predicting. A separate
layer compares model output to the market and reports calibrated cover probabilities.

Data source: [`nflreadpy`](https://nflreadpy.nflverse.com/). No API key required.
The optional advanced-stat candidate uses public
[`nflverse` PFR releases](https://github.com/nflverse/nflverse-data/releases/tag/pfr_advstats)
derived from [Pro Football Reference](https://www.pro-football-reference.com/) data; credit
for those fields belongs to nflverse and Pro Football Reference.

Design: `docs/superpowers/specs/2026-07-23-nfl-game-model-design.md`

## Setup

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

## Tests

    .\.venv\Scripts\python.exe -m pytest

## Ridge-v2 research dataset

Ridge v2 is a research track under evaluation. It does not serve the dashboard, and nothing
in `web/` reads it.

    # Report sources, coverage and digests; write nothing (the default):
    .\.venv\Scripts\python.exe scripts\build_v2_dataset.py --dry-run

    # Write the two Ridge-v2 artifacts:
    .\.venv\Scripts\python.exe scripts\build_v2_dataset.py --write

`--dry-run` and `--write` are mutually exclusive, and `--traceback` prints the failing stack
instead of a one-line error. The build takes roughly 30 minutes, re-downloads its sources,
and peaks near 11 GB of committed memory -- do not run it concurrently with
`scripts/backtest.py`, and launch it detached rather than in a foreground shell you need
back. It writes only `data/processed/game_features_ridge_v2.parquet` and
`data/processed/ridge_v2_manifest.json`, and refuses a Ridge-v1 destination outright; the
three packaged v1 artifacts are left byte-identical.

Only the manifest is committed. The feature parquet is gitignored and rebuilt from source,
so a fresh clone must run `--write` before anything can read it.

The locked research experiment runs on top of that artifact:

    .\.venv\Scripts\python.exe scripts\backtest_v2.py --dry-run
    .\.venv\Scripts\python.exe scripts\backtest_v2.py --write

It reproduces Ridge v1 over 2019-2025, runs the Ridge-v2 nested walk-forward over the same
span, evaluates promotion on the identical 2021-2025 rows, and prints promotion gates 1-10
with gate 11 pending. It takes about 22 minutes, defaults to dry-run, and writes only the
four research artifacts (`ridge_v2_outer_predictions.parquet`, `ridge_v2_evaluation.json`,
`ridge_v2_ablation.parquet`, `ridge_v2_calibration.json`) -- all four are committed, and a
Ridge-v1 destination is refused. `--require-research-gates` makes a failed gate exit
nonzero; without it the run reports and exits 0.

### FTN charting (E1, research only)

    .\.venv\Scripts\python.exe scripts\backtest_v2_ftn.py --dry-run

FTN charting begins in 2022, which is too short a history for the production candidate, so
this is a separate experiment that cannot promote anything. It compares two arms trained on
identical rows -- the C0 core schema, and that schema plus the FTN features -- for each outer
season the 2022+ history supports (2023-2025). It refuses to write over any Ridge-v1 or
Task 13-17 artifact, defaults to dry-run, and takes about a minute once the sources are
cached. **Result: FTN did not help** -- see CLAUDE.md.

## Web dashboard operations

The dashboard serves three checked-in artifacts:
`data/processed/game_features.parquet`, `data/processed/schedule_2026.parquet`, and
`data/processed/tracker_ledger.parquet`. It reads all three at startup; there is no
website refresh endpoint, and the web service cannot mutate any artifact. The feature
artifact contains the frozen historical corpus plus the current 2026 prediction weeks;
the schedule artifact contains all 272 regular-season games.

### Run locally

Use the explicit unauthenticated mode only on the local machine. It binds to the
numeric loopback address (`127.0.0.1`) and cannot be combined with `ACCESS_CODE`.

```powershell
# Explicit loopback-only local mode
.\.venv\Scripts\python.exe scripts\game_app.py --no-auth
```

For a protected local or network-accessible run, set a non-empty code in the
environment before starting. `ACCESS_CODE` is required in every normal startup;
when it is absent, blank, or whitespace, the process exits nonzero rather than
starting without protection. Do not commit a real code.

```powershell
# Protected local/network mode
$env:ACCESS_CODE = "local-test-only"
.\.venv\Scripts\python.exe scripts\game_app.py
```

The protected server listens on `0.0.0.0` and honors `PORT` (default `8000`). The
unauthenticated `--no-auth` server always binds only to loopback. A protected login
sets a secure, HTTP-only session cookie, so a plain-HTTP browser will not retain it.
Use HTTPS for an end-to-end protected-browser test; the automated `TestClient`
coverage exercises the cookie behavior.

### Dashboard and API

The page initializes to the latest available season and the first unplayed prediction
week in that season, advancing to the next week after the current week's games become
final. If every available week is final it selects the latest one. Available seasons,
prediction weeks, and estimators come from the feature artifact. The default estimator
is `ridge` and the default edge threshold is `2.0`. The table compares model and market
spread/total values, and an edge marker is informational rather than betting advice.

For 2026, spread and total lines are refreshed from the nflverse schedule feed through
`nflreadpy`. Each server process caches one validated season snapshot for five minutes
and waits at most five seconds for an upstream refresh. A failed cold request falls back
to the packaged schedule; after a successful request, an upstream error or timeout
returns the last snapshot marked stale. Spread and total availability are independent:
a missing value stays visibly missing and never changes a model prediction.

The browser uses these read-only endpoints:

- `GET /health` returns `{"ok": true}` and is public; it is available for an optional configured HTTP health check.
- `GET /api/options` returns selectors and their defaults; `GET /api/weeks?season=...`
  returns valid weeks for one season.
- `GET /api/slate?season=...&week=...&estimator=ridge&edge_threshold=2.0` returns the
  active slate as JSON.
- `GET /api/slate.csv` accepts the same parameters and downloads CSV with the same
  rows and ordering as the displayed slate.

- `GET /schedule` serves the full 2026 schedule page; `GET /api/schedule?season=2026`
  returns all regular-season games plus the market source, observation time, and stale
  indicator.

### Performance tracker

`/tracker` separates historical walk-forward backtests from live published picks.
Historical records cover Ridge `ridge-v1`, 2021–2025, against closing lines. Qualified
picks use 2+ points, and spread groups are cumulative 5+/10+/15+; pushes do not enter
win-rate denominators. The live section starts in 2026. In Stage 1 it remains read-only
and empty while official writes are disabled. Once separately approved for Stage 2,
official live grades use frozen published lines; closing-line value (CLV) and the close
record are secondary and cannot rewrite the official result.

The four tracker routes are read-only:

- `GET /tracker` serves the tracker page.
- `GET /api/tracker/options` returns the available record types and seasons.
- `GET /api/tracker/summary?record_type=backtest&season=all` returns the selected record
  summary.
- `GET /api/tracker/games?record_type=backtest&season=2025` returns the selected season's
  game-level audit records.

In protected mode, unauthenticated browser routes redirect to `/login` and API routes
return `401`. Wrong login codes return `401`; repeated failed attempts from one client
address are temporarily throttled with `429`.

### 2026 artifact and tracker operations

Both update commands default to preview mode. Running either command without
`--dry-run` or `--write` validates and reports but does not change an artifact. The mode
flags are mutually exclusive.

Refresh the full 2026 regular-season schedule and the current/next prediction-week
features from nflverse, review the reported row counts and digests, then write the two
artifacts atomically:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
.\.venv\Scripts\python.exe scripts\refresh_2026.py --write
```

Preview the official tracker lifecycle separately. `--write` is permitted only after
Stage 2 approval; it atomically replaces the ledger only when deterministic bytes have
changed. `--now` accepts a timezone-aware UTC lifecycle time for an audited manual run.

```powershell
.\.venv\Scripts\python.exe scripts\update_live_tracker.py --dry-run
# Stage 2 only:
.\.venv\Scripts\python.exe scripts\update_live_tracker.py --write
```

Publication and grading rules are immutable once a game is recorded:

- A Ridge `ridge-v1` prediction becomes eligible for publication under the lead-time rule
  below. Its model margin, model total, model version, publication time, and original
  kickoff are frozen.
- Picks lock **4 days before kickoff**, floored so that no game is published until the
  features artifact has been rebuilt from a complete prior week. In practice the Sunday
  and Monday slate gets the full 4 days; Thursday games and Thanksgiving are held by the
  floor at roughly 2.3–2.6 days.
- Spread and total publish independently. An available line is frozen with its observation
  time. A missing line remains pending until one hour before kickoff, then becomes excluded
  with `missing_line_at_deadline`; a record first seen after that deadline is excluded with
  `publication_window_missed`. An excluded market never enters its win-rate denominator.
- Six hours after the current kickoff, a completed score and the then-current closing lines
  are captured. Official wins/losses/pushes are graded against the frozen published lines;
  pushes return the stake and are excluded from win-rate denominators. CLV compares the
  published and closing lines and is secondary to the official record.
- A postponement updates only `current_kickoff_at`; it cannot rewrite the original kickoff,
  prediction, publication facts, or already-frozen lines. A missing game remains under the
  same lifecycle checks.
- A manual void is explicit and repeatable only with the same nonblank reason:
  `.\.venv\Scripts\python.exe scripts\update_live_tracker.py --write --void-game
  "GAME_ID=reason"`. A voided record is retained for audit and is not graded.
- If a record is still missing required final facts seven days after kickoff, the updater
  fails loudly. Inspect the upstream game and recorded lines; correct the feed/input or
  apply an audited void rather than deleting or silently rewriting the row.

Every artifact change must retain exactly 1,359 historical tracker rows and the acceptance
records below. Never rebuild historical facts as part of a routine live update.

### Automated refresh and staged tracker enablement

`.github/workflows/refresh-2026-model.yml` runs daily at 10:30 UTC and by manual dispatch.
It tests, refreshes the feature and schedule artifacts, builds the container, and commits
only when either artifact changed. `.github/workflows/update-2026-tracker.yml` runs every
15 minutes during August-February and by manual dispatch. Both workflows use the shared
`nfl-generated-data-writer` concurrency group, never cancel an in-progress writer, and
skip empty commits. Before pushing to `master`, each fetches the remote and rejects the
push unless the checked-out commit still contains the current remote tip as an ancestor.

Stage 1 is fail-closed: when repository variable `ENABLE_OFFICIAL_TRACKER` is absent or
not exactly `true`, the tracker workflow runs `--dry-run` and has no write/commit step.
Manual dispatch is useful for observing candidates but does not bypass this gate:

```powershell
gh workflow run refresh-2026-model.yml
gh workflow run update-2026-tracker.yml
```

Do not set `ENABLE_OFFICIAL_TRACKER` during Stage 1. Stage 2 requires a separate review of
the proposed game IDs, publication timestamps, frozen lines, model version, edge values,
and excluded markets, followed by explicit approval in a later action.

If nflverse is unavailable, the command or workflow fails before replacing artifacts.
Leave the reviewed files in place, wait for recovery, and manually rerun the relevant
workflow; do not force a partial commit. The website continues with its last cached market
snapshot marked stale, or the packaged schedule when no live snapshot exists.

To roll back a bad artifact release, revert the exact artifact commit, verify that all
three files come from one reviewed revision, rerun the full release gate below, and deploy
that revert. Do not hand-edit Parquet files. If the safe-push ancestor check rejects a
workflow run, update from the new `master` tip and rerun instead of force-pushing.

### Reproducible container inputs

`Dockerfile` pins the official `python:3.12-slim` multi-platform OCI index to
`sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de`.
That digest was checked at `2026-07-29T23:58:48Z` against the official Docker
Registry v2 manifest endpoint and independently matched by Docker Hub's official tag API:

```powershell
$tokenUri = 'https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull'
$registryToken = (Invoke-RestMethod -Uri $tokenUri).token
curl.exe -sS -I -H "Authorization: Bearer $registryToken" `
  -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json" `
  'https://registry-1.docker.io/v2/library/python/manifests/3.12-slim'
Invoke-RestMethod 'https://hub.docker.com/v2/repositories/library/python/tags/3.12-slim' |
  Select-Object digest,last_updated
```

The registry response was `200`, content type `application/vnd.oci.image.index.v1+json`,
and reported the pinned `docker-content-digest`; the Hub tag was last updated
`2026-07-16T11:07:24.538892Z`. `requirements-prod.txt` locks the CPython 3.12,
x86_64 Linux (`manylinux_2_28`) runtime closure to exact selected-wheel hashes. It was
resolved with `uv 0.12.0` using:

```powershell
uv pip compile pyproject.toml `
  --python-platform x86_64-unknown-linux-gnu `
  --python-version 3.12 `
  --only-binary :all: `
  --generate-hashes `
  --no-annotate `
  --upgrade `
  -o requirements-prod.txt
```

`requirements-build.txt` separately pins and hashes pip and setuptools. Docker installs
both locks with `--require-hashes`, then installs this project with dependency resolution
and build isolation disabled. Re-resolve and re-verify the runtime lock if the production
architecture changes.

### Render deployment and proxy boundary

`render.yaml` defines the Docker Blueprint service named `ashburn-nfl`. In Render,
create the Blueprint from the reviewed integrated `master` branch, confirm that service
name, and set a non-empty private `ACCESS_CODE` secret. Do not put the secret in the
repository, a command history, request logs, or a deployment note. The image packages
all three immutable Parquet artifacts and starts `scripts/game_app.py`; production therefore
fails closed if `ACCESS_CODE` is missing.

Login throttling reads the first `X-Forwarded-For` address when that header is present.
Rely on it only at the deployed Render ingress boundary, where the application is not
directly exposed and forwarded client metadata is controlled by the proxy. The application
itself does not verify who supplied this header: a directly reachable server lets callers
choose their throttle identity by sending it. Never expose the application port directly
to the Internet. For a direct local protected smoke test, do not send an
`X-Forwarded-For` header; that smoke does not prove the production proxy boundary.

Before DNS cutover, deploy and verify the Render hostname from a cookie-free external
client. Do not add `nfl.ashburn-capital.com` until these checks pass:

```powershell
$renderBase = (Read-Host "Paste the exact Render service URL").TrimEnd("/")
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" "$renderBase/"
curl.exe -s -o NUL -w "%{http_code}" "$renderBase/api/options"
$wrongBody = '{"code":"definitely-wrong"}'
$wrongBody | curl.exe -s -o NUL -w "%{http_code}" `
  -H "Content-Type: application/json" --data-binary '@-' "$renderBase/login"
```

Expected results are `303` redirecting to `/login`, `401`, and `401`. Restart the
Render service and repeat the same cookie-free checks; an old session must no longer
authorize the API. DNS cutover must not happen until these Render-hostname checks pass.

For the custom-domain cutover, add `nfl.ashburn-capital.com` in Render first and record
the exact CNAME target Render displays. Before changing DNS, record the current DNS
record type, value, and TTL (or explicitly record that no prior record exists). Create or
replace the DNS record with the exact Render-provided CNAME target, wait for propagation,
and wait for Render domain verification and TLS issuance. Then repeat the same cookie-free
checks against `https://nfl.ashburn-capital.com` and sign in interactively to confirm
selector changes and the CSV download.

If Render verification, TLS issuance, or the post-cutover auth smoke fails, roll back:
restore the recorded prior DNS type, value, and TTL; if there was no prior target, remove
the new record. Wait for propagation, then re-run the known-good cookie-free checks on
the Render hostname above. Do not retry the custom-domain cutover until the problem is
fixed and the Render-hostname checks pass again.

### Release verification and troubleshooting

Run every locally available gate against the exact commit being reviewed:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
.\.venv\Scripts\python.exe scripts\update_live_tracker.py --dry-run
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 `
  .github/workflows/refresh-2026-model.yml .github/workflows/update-2026-tracker.yml
docker build -t nfl-game-model:2026 .
Get-FileHash data\processed\game_features.parquet,`
  data\processed\schedule_2026.parquet,`
  data\processed\tracker_ledger.parquet -Algorithm SHA256
.\.venv\Scripts\python.exe -c "import pandas as pd; from pathlib import Path; [print(p, len(pd.read_parquet(p))) for p in map(Path, ['data/processed/game_features.parquet', 'data/processed/schedule_2026.parquet', 'data/processed/tracker_ledger.parquet'])]"
```

A missing local executable is an unverified gate, not a pass: record it and run it on a
machine that has the tool before release. Do not install release tools ad hoc on a
reviewed workstation.

Re-run the statistical acceptance baseline after any artifact refresh. This normalized
acceptance summary preserves the invariants (the CLI renders `total  MAE` with two spaces).
Stop and investigate if any value moves; a web-only change must not change these results.

```powershell
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
```

```text
games:            1359
margin MAE:       10.274   market: 9.752
total MAE:        10.684   market: 10.309
ATS hit rate:     0.4977   n=1326
O/U hit rate:     0.5022   n=1348
model_coef:       -0.0218
market_coef:      1.0755
r2:               0.2083
```

The following release gate remains mandatory for the exact reviewed commit accepted
for deployment: Docker and Render must build that exact commit; a container started
without `ACCESS_CODE` must exit nonzero; with a non-default protected `PORT`, `/health`
must return `200`, `/` and `/tracker` must return `303` to `/login`, and `/api/options`
and `/api/tracker/options` must return `401`; the built image must contain all three
packaged Parquet artifacts; and the Render Blueprint must validate and build successfully.
If Docker, Go/actionlint, or Render validation is unavailable locally, record that exact
limitation; it remains an unpassed release gate that must be completed before deployment.

If startup reports that `ACCESS_CODE` is required, set a non-empty secret for protected
mode or use `--no-auth` only for loopback development. If it reports a missing packaged
artifact, restore `data/processed/game_features.parquet`,
`data/processed/schedule_2026.parquet`, and `data/processed/tracker_ledger.parquet` from
the same accepted revision and rebuild the image. A protected page that returns to
`/login` over plain HTTP is expected: the secure
cookie requires HTTPS. A `422` from slate or tracker endpoints usually means a requested
season, week, estimator, threshold, or record type is not an available valid selection;
reload `/api/options` or `/api/tracker/options` and choose its advertised values. A `429`
login response requires waiting for its `Retry-After` interval rather than repeatedly
retrying.

### 2026 container data smoke

After building `nfl-game-model:2026`, smoke the image internally in loopback-only no-auth
mode. This does not weaken or replace the production secure-cookie check.

```powershell
function Wait-NflContainer {
    param([string]$ContainerName)
    $probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"
    $deadline = (Get-Date).AddSeconds(30)
    do {
        docker exec $ContainerName python -c $probe
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    docker logs $ContainerName
    throw "$ContainerName did not become healthy"
}

$endpointProbe = @"
import json
import urllib.request
for path in (
    "/health",
    "/api/options",
    "/api/schedule?season=2026",
    "/api/slate",
    "/api/tracker/options",
):
    with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=10) as response:
        json.load(response)
    print(path)
"@

$staleProbe = @"
import json
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/api/slate", timeout=10) as response:
    payload = json.load(response)
assert payload["market"]["source"] == "packaged"
assert payload["market"]["stale"] is True
"@

docker rm -f nfl-game-smoke nfl-game-stale 2>$null | Out-Null
docker run -d --name nfl-game-smoke nfl-game-model:2026 python scripts/game_app.py --no-auth
try {
    Wait-NflContainer "nfl-game-smoke"
    docker exec nfl-game-smoke python -c $endpointProbe
    if ($LASTEXITCODE -ne 0) { throw "connected container smoke failed" }
} finally {
    docker rm -f nfl-game-smoke | Out-Null
}

docker run -d --name nfl-game-stale --network none nfl-game-model:2026 python scripts/game_app.py --no-auth
try {
    Wait-NflContainer "nfl-game-stale"
    docker exec nfl-game-stale python -c $staleProbe
    if ($LASTEXITCODE -ne 0) { throw "offline fallback smoke failed" }
} finally {
    docker rm -f nfl-game-stale | Out-Null
}
```

After Stage 1 is deployed behind HTTPS, sign in with the private access code and repeat
`/api/options`, `/api/schedule?season=2026`, `/api/slate`, and `/api/tracker/options` in
the authenticated browser session. Confirm the tracker is still read-only and has no
unapproved live records.

### Executable Docker and Blueprint gate

Run this gate from a clean checkout of the exact reviewed commit that is being accepted
for release. Substitute a non-secret local value only for the smoke container; set the
real private value only in Render.

```powershell
$acceptedCommit = (git rev-parse HEAD)
git status --short                         # expected: no tracked output
docker build --tag "ashburn-nfl:$acceptedCommit" .

# Expected: nonzero exit and an ACCESS_CODE-required startup error.
docker run --rm --name ashburn-nfl-missing-code "ashburn-nfl:$acceptedCommit"
if ($LASTEXITCODE -eq 0) { throw "Container started without ACCESS_CODE" }

# Expected: 200; 303 with a /login redirect; 401; 303 with a /login redirect; 401.
$container = docker run --detach --rm --name ashburn-nfl-smoke `
  -e ACCESS_CODE=local-test-only -e PORT=8765 -p 8765:8765 "ashburn-nfl:$acceptedCommit"
try {
  Start-Sleep -Seconds 2
  curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8765/health
  curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" http://127.0.0.1:8765/
  curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8765/api/options
  curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" http://127.0.0.1:8765/tracker
  curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8765/api/tracker/options
} finally {
  docker stop $container | Out-Null
}

# Render CLI 2.7.1+; expected: Blueprint validation succeeds.
render blueprints validate render.yaml
```

Then create the Blueprint in Render from that same accepted integrated `master` commit,
provide the private `ACCESS_CODE` when prompted, and require a successful Render Docker
build before continuing to the external checks above. The Blueprint validation command
does not create or modify Render resources; deployment and DNS changes remain the
separate post-integration release task.
