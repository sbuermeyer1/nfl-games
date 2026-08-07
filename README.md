# NFL Game Model

Predicts NFL game margins and totals from EPA + Next Gen Stats team ratings, then compares
those predictions against the closing spread and total.

The model is **market-blind**: it never sees the betting line when predicting. A separate
layer compares model output to the market and reports calibrated cover probabilities.

Data source: [`nflreadpy`](https://nflreadpy.nflverse.com/). No API key required.

Design: `docs/superpowers/specs/2026-07-23-nfl-game-model-design.md`

## Setup

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

## Tests

    .\.venv\Scripts\python.exe -m pytest

## Web dashboard operations

The dashboard serves the checked-in, immutable `data/processed/game_features.parquet`
and `data/processed/tracker_ledger.parquet` artifacts. It reads both artifacts at
startup; there is no website refresh endpoint and the web service cannot mutate either
artifact.

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

The page initializes to the latest available season and its latest week. The
available seasons, weeks, and estimators come from the packaged artifact; changing a
season reloads its valid weeks. The default estimator is `ridge` and the default edge
threshold is `2.0`. The table compares model and market spread/total values, and an
edge marker is informational rather than betting advice.

The browser uses these read-only endpoints:

- `GET /health` returns `{"ok": true}` and is public; it is available for an optional configured HTTP health check.
- `GET /api/options` returns selectors and their defaults; `GET /api/weeks?season=...`
  returns valid weeks for one season.
- `GET /api/slate?season=...&week=...&estimator=ridge&edge_threshold=2.0` returns the
  active slate as JSON.
- `GET /api/slate.csv` accepts the same parameters and downloads CSV with the same
  rows and ordering as the displayed slate.

### Performance tracker

`/tracker` separates historical walk-forward backtests from live published picks.
Historical records cover Ridge `ridge-v1`, 2021–2025, against closing lines. Qualified
picks use 2+ points, and spread groups are cumulative 5+/10+/15+; pushes do not enter
win-rate denominators. The live section starts in 2026 and remains unavailable until the
separate live workflow is built. Future official live grades use frozen published lines,
while CLV and the close record are secondary.

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

### Refresh the packaged artifact

Refreshes are an offline, reviewed data change. After rebuilding features, rebuild the
tracker with `.\.venv\Scripts\python.exe scripts\build_tracker.py`. Build, test,
backtest, and commit both Parquet artifacts together:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py --start-season 2016 --end-season 2026
.\.venv\Scripts\python.exe scripts\build_tracker.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
git add data/processed/game_features.parquet data/processed/tracker_ledger.parquet
git commit -m "data: refresh packaged NFL artifacts"
```

When upstream results change, both Parquet artifacts must be reviewed, committed, and
deployed together.

The website has no refresh endpoint and cannot mutate the artifact. Do not use a web
deployment as a substitute for this workflow.

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
both immutable Parquet artifacts and starts `scripts/game_app.py`; production therefore
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

Run the local test and style suite before accepting a release:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

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
and `/api/tracker/options` must return `401`; the built image must contain both packaged
Parquet artifacts; and the Render Blueprint must validate and build successfully. Local
Docker verification was unavailable on the Task 6 workstation, so it is not a passed
check and must be completed before release.

If startup reports that `ACCESS_CODE` is required, set a non-empty secret for protected
mode or use `--no-auth` only for loopback development. If it reports a missing packaged
dataset or tracker ledger, restore `data/processed/game_features.parquet` and
`data/processed/tracker_ledger.parquet` from the same accepted revision and rebuild the
image. A protected page that returns to `/login` over plain HTTP is expected: the secure
cookie requires HTTPS. A `422` from slate or tracker endpoints usually means a requested
season, week, estimator, threshold, or record type is not an available valid selection;
reload `/api/options` or `/api/tracker/options` and choose its advertised values. A `429`
login response requires waiting for its `Retry-After` interval rather than repeatedly
retrying.

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
