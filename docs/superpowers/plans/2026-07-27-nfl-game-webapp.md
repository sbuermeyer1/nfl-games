# NFL Game Model Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy an access-code-protected weekly NFL model dashboard at `nfl.ashburn-capital.com`, using the existing prebuilt game-feature dataset and model logic.

**Architecture:** Add a focused `nfl_game.web` package around the existing `data -> ratings -> model -> market` pipeline. A thread-safe `SlateService` loads one immutable parquet dataset and caches fitted model/calibrator pairs by season and estimator; FastAPI routes expose the service as a bare-bones HTML dashboard, JSON API, and CSV download. Production startup fails closed without `ACCESS_CODE`, while explicit local `--no-auth` mode binds only to loopback.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, FastAPI, Uvicorn, pytest, HTTPX/TestClient, Docker, Render

## Global Constraints

- Preserve the existing statistical model, calibration logic, market sign convention, and 2021–2025 backtest baseline.
- Use only the packaged `data/processed/game_features.parquet`; the web process must never rebuild or mutate model data.
- Expose season, week, estimator (`ridge` or `gbm`), and a finite non-negative edge threshold; keep ridge alpha fixed at `DEFAULT_ALPHA == 1.0`.
- Default to the latest packaged season/week, `ridge`, and edge threshold `2.0`.
- Missing market values must remain null in JSON/CSV and display as `n/a`, never `nan`.
- Production/default startup without `ACCESS_CODE` must fail; only explicit `--no-auth` local mode may bypass login, and that mode must bind to `127.0.0.1`.
- Keep HTML/CSS/JavaScript embedded and framework-free.
- Do not add accounts, persistence, refresh jobs, model editing, betting advice, or a separate frontend build.
- DNS cutover is blocked until external unauthenticated checks pass on the Render hostname before and after a service restart.

---

## File Structure

### New application files

- `src/nfl_game/web/__init__.py` — web package boundary and stable public exports.
- `src/nfl_game/web/service.py` — immutable dataset validation, options, cached fitting, slate generation, JSON records, and CSV serialization.
- `src/nfl_game/web/session.py` — bounded, expiring opaque authentication sessions.
- `src/nfl_game/web/throttle.py` — bounded per-IP failed-login throttling.
- `src/nfl_game/web/auth.py` — cookie parsing and pure-ASGI access middleware.
- `src/nfl_game/web/login.py` — login HTML and GET/POST routes.
- `src/nfl_game/web/app.py` — dashboard HTML, API routes, response conversion, and client-safe exception handlers.
- `src/nfl_game/web/runtime.py` — fail-closed environment/CLI configuration and dataset loading.
- `scripts/game_app.py` — thin Uvicorn production/local entry point.

### New tests

- `tests/test_web_service.py` — service validation, defaults, caching, slate parity, null conversion, and CSV behavior.
- `tests/test_web_session.py` — token expiry and bounded session storage.
- `tests/test_web_throttle.py` — lockout, reset, proxy IP, and memory bounds.
- `tests/test_web_auth.py` — middleware challenges, health/login exemptions, and cookie acceptance.
- `tests/test_web_login.py` — correct/wrong/non-ASCII codes, secure cookie flags, and throttling.
- `tests/test_webapp.py` — HTML, options, weeks, slate JSON, errors, and CSV parity.
- `tests/test_web_runtime.py` — fail-closed configuration, loopback-only no-auth, and parquet startup errors.


### Modified/generated files

- `pyproject.toml` — add FastAPI/Uvicorn runtime dependencies and HTTPX dev dependency.
- `.gitignore` — allow only `data/processed/game_features.parquet` through the processed-data ignore.
- `data/processed/game_features.parquet` — tracked, prebuilt deployment artifact.
- `Dockerfile` — production image.
- `render.yaml` — `ashburn-nfl` Docker service and private `ACCESS_CODE`.
- `README.md` — local use, artifact refresh, deployment, and external verification runbook.

---

### Task 1: Slate Service and Runtime Dependencies

**Files:**

- Create: `src/nfl_game/web/__init__.py`
- Create: `src/nfl_game/web/service.py`
- Create: `tests/test_web_service.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: `walk_forward(features_df, test_seasons, estimator, alpha)`, `GameModel(estimator, alpha)`, `Calibrator`, `build_slate`, `ESTIMATORS`, `DEFAULT_ALPHA`, `FEATURE_COLS`.
- Produces: `SlateService.from_parquet(path)`, `SlateService.options()`, `SlateService.weeks(season)`, `SlateService.slate(season, week, estimator="ridge", edge_threshold=2.0)`, `SlateService.records(season, week, estimator="ridge", edge_threshold=2.0)`, `SlateService.csv(season, week, estimator="ridge", edge_threshold=2.0)`, `SlateInputError`, `SlateNotFoundError`, and `SlateUnavailableError`.

- [ ] **Step 1: Add web-test dependencies and write failing option/validation tests**

Add these dependencies without changing existing version floors:

```toml
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "scipy>=1.13",
    "nflreadpy>=0.1",
    "pyarrow>=16.0",
    "fastapi>=0.111",
    "uvicorn>=0.30",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
    "httpx>=0.27",
]
```

Create a compact fixture with all `FEATURE_COLS`, two seasons, and two weeks:

```python
import math

import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.web.service import SlateInputError, SlateService


def feature_rows() -> pd.DataFrame:
    rows = []
    for season, weeks in ((2024, (1, 2)), (2025, (1, 3))):
        for week in weeks:
            row = {column: 0.1 for column in FEATURE_COLS}
            row.update(
                game_id=f"{season}_{week:02d}_AAA_BBB",
                season=season,
                week=week,
                away_team="AAA",
                home_team="BBB",
                spread_line=2.5,
                total_line=44.5,
                margin=3.0,
                total_points=45.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_options_default_to_latest_packaged_week():
    service = SlateService(feature_rows())
    assert service.options() == {
        "seasons": [2024, 2025],
        "weeks": [1, 3],
        "estimators": ["gbm", "ridge"],
        "default_estimator": "ridge",
        "default_edge_threshold": 2.0,
        "latest": {"season": 2025, "week": 3},
    }


@pytest.mark.parametrize("threshold", [-0.1, math.inf, -math.inf, math.nan])
def test_threshold_must_be_finite_and_non_negative(threshold):
    with pytest.raises(SlateInputError, match="edge threshold"):
        SlateService(feature_rows()).slate(2025, 1, "ridge", threshold)


def test_season_week_pair_must_exist():
    with pytest.raises(SlateInputError, match="week 2 is not available for season 2025"):
        SlateService(feature_rows()).slate(2025, 2, "ridge", 2.0)
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_service.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'nfl_game.web'`.

- [ ] **Step 3: Implement dataset validation and option discovery**

Create `service.py` with these contracts:

```python
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.market.compare import build_slate
from nfl_game.model.calibrate import Calibrator
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import (
    DEFAULT_ALPHA,
    ESTIMATORS,
    DegenerateFeatureError,
    GameModel,
)

DEFAULT_EDGE_THRESHOLD = 2.0
REQUIRED_COLUMNS = {
    "game_id", "season", "week", "away_team", "home_team",
    "spread_line", "total_line", "margin", "total_points", *FEATURE_COLS,
}


class SlateInputError(ValueError):
    """A requested option is invalid for the packaged dataset."""


class SlateUnavailableError(RuntimeError):
    """The selected slate cannot be modeled from the available prior data."""


class SlateNotFoundError(SlateUnavailableError):
    """The valid selection contains no games to return."""


@dataclass(frozen=True)
class ModelBundle:
    model: GameModel
    calibrator: Calibrator


class SlateService:
    def __init__(self, features: pd.DataFrame):
        missing = sorted(REQUIRED_COLUMNS - set(features.columns))
        if missing:
            raise ValueError(f"game features missing required columns: {missing}")
        if features.empty:
            raise ValueError("game features dataset is empty")
        if features["game_id"].duplicated().any():
            raise ValueError("game features contain duplicate game_id values")
        self._features = features.copy()
        self._cache: dict[tuple[int, str], ModelBundle] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_parquet(cls, path: str | Path) -> "SlateService":
        return cls(pd.read_parquet(path))

    def weeks(self, season: int) -> list[int]:
        seasons = set(int(value) for value in self._features["season"].unique())
        if season not in seasons:
            raise SlateInputError(f"season {season} is not available")
        values = self._features.loc[self._features["season"] == season, "week"].unique()
        return sorted(int(value) for value in values)

    def options(self) -> dict:
        seasons = sorted(int(value) for value in self._features["season"].unique())
        latest_season = seasons[-1]
        weeks = self.weeks(latest_season)
        return {
            "seasons": seasons,
            "weeks": weeks,
            "estimators": sorted(ESTIMATORS),
            "default_estimator": "ridge",
            "default_edge_threshold": DEFAULT_EDGE_THRESHOLD,
            "latest": {"season": latest_season, "week": weeks[-1]},
        }
```

Export the stable names from `src/nfl_game/web/__init__.py`:

```python
from nfl_game.web.service import (
    SlateInputError,
    SlateNotFoundError,
    SlateService,
    SlateUnavailableError,
)

__all__ = [
    "SlateInputError",
    "SlateNotFoundError",
    "SlateService",
    "SlateUnavailableError",
]
```

- [ ] **Step 4: Run the focused tests and confirm option/validation tests pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_service.py -v
```

Expected: option and validation tests pass; slate-generation tests have not been added yet.

- [ ] **Step 5: Write failing cache, parity, null-record, and CSV tests**

Use monkeypatched model dependencies so these tests stay fast and deterministic:

```python
def test_reuses_bundle_across_weeks(monkeypatch):
    service = SlateService(feature_rows())
    calls = {"fit": 0}

    class FakeModel:
        def __init__(self, estimator, alpha):
            self.estimator = estimator

        def fit(self, train):
            calls["fit"] += 1
            return self

        def predict(self, target):
            return pd.DataFrame({
                "game_id": target["game_id"],
                "model_margin": 4.0,
                "model_total": 46.0,
            })

    class FakeCalibrator:
        def fit(self, oos):
            return self

        def predict(self, merged):
            return pd.DataFrame({
                "game_id": merged["game_id"],
                "cover_prob": 0.6,
                "over_prob": 0.55,
            })

    monkeypatch.setattr("nfl_game.web.service.GameModel", FakeModel)
    monkeypatch.setattr("nfl_game.web.service.Calibrator", FakeCalibrator)
    monkeypatch.setattr(
        "nfl_game.web.service.walk_forward",
        lambda features, seasons, estimator, alpha: features.assign(
            model_margin=3.0, model_total=45.0
        ),
    )

    service.slate(2025, 1, "ridge", 2.0)
    service.slate(2025, 3, "ridge", 3.0)
    assert calls["fit"] == 1


def test_estimator_has_its_own_cache_key(monkeypatch):
    service, calls = fake_fitted_service(monkeypatch)
    service.slate(2025, 1, "ridge", 2.0)
    service.slate(2025, 1, "gbm", 2.0)
    assert calls["fit"] == 2


def test_concurrent_requests_fit_one_bundle(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time

    service = SlateService(feature_rows())
    calls = {"fit": 0}
    bundle = object()

    def slow_fit(season, estimator):
        calls["fit"] += 1
        time.sleep(0.05)
        return bundle

    monkeypatch.setattr(service, "_fit_bundle", slow_fit)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda _: service._bundle(2025, "ridge"),
            range(8),
        ))
    assert calls["fit"] == 1
    assert all(result is bundle for result in results)


def test_records_convert_nan_to_none(monkeypatch):
    service, _ = fake_fitted_service(monkeypatch, spread_line=float("nan"))
    row = service.records(2025, 1, "ridge", 2.0)[0]
    assert row["market_spread"] is None
    assert row["spread_gap"] is None
    assert row["cover_prob"] is None


def test_csv_uses_same_rows_and_never_writes_nan(monkeypatch):
    service, _ = fake_fitted_service(monkeypatch, total_line=float("nan"))
    csv_text = service.csv(2025, 1, "ridge", 2.0)
    assert csv_text.startswith("game_id,season,week,away_team,home_team")
    assert "nan" not in csv_text.lower()
```

Define the shared helper explicitly:

```python
def fake_fitted_service(monkeypatch, spread_line=2.5, total_line=44.5):
    rows = feature_rows()
    rows.loc[rows["season"] == 2025, "spread_line"] = spread_line
    rows.loc[rows["season"] == 2025, "total_line"] = total_line
    service = SlateService(rows)
    calls = {"fit": 0}

    class FakeModel:
        def __init__(self, estimator, alpha):
            self.estimator = estimator

        def fit(self, train):
            calls["fit"] += 1
            return self

        def predict(self, target):
            return pd.DataFrame({
                "game_id": target["game_id"].to_numpy(),
                "model_margin": [4.0] * len(target),
                "model_total": [46.0] * len(target),
            })

    class FakeCalibrator:
        def fit(self, oos):
            return self

        def predict(self, merged):
            cover = [float("nan") if pd.isna(value) else 0.6 for value in merged["spread_line"]]
            over = [float("nan") if pd.isna(value) else 0.55 for value in merged["total_line"]]
            return pd.DataFrame({
                "game_id": merged["game_id"].to_numpy(),
                "cover_prob": cover,
                "over_prob": over,
            })

    monkeypatch.setattr("nfl_game.web.service.GameModel", FakeModel)
    monkeypatch.setattr("nfl_game.web.service.Calibrator", FakeCalibrator)
    monkeypatch.setattr(
        "nfl_game.web.service.walk_forward",
        lambda features, seasons, estimator, alpha: features.assign(
            model_margin=3.0, model_total=45.0
        ),
    )
    return service, calls
```

- [ ] **Step 6: Run the new tests and confirm they fail on missing methods**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_service.py -v
```

Expected: failures name missing `SlateService.slate`, `records`, and `csv`.

- [ ] **Step 7: Implement validation, fitting, caching, slate generation, records, and CSV**

Add these methods to `SlateService`:

```python
    def _validate(self, season: int, week: int, estimator: str, edge_threshold: float) -> None:
        if estimator not in ESTIMATORS:
            raise SlateInputError(
                f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}"
            )
        if not math.isfinite(edge_threshold) or edge_threshold < 0:
            raise SlateInputError("edge threshold must be a finite non-negative number")
        weeks = self.weeks(season)
        if week not in weeks:
            raise SlateInputError(f"week {week} is not available for season {season}")

    def _fit_bundle(self, season: int, estimator: str) -> ModelBundle:
        prior_seasons = sorted(
            int(value)
            for value in self._features.loc[
                self._features["season"] < season, "season"
            ].unique()
        )
        if not prior_seasons:
            raise SlateUnavailableError(
                f"no calibration data is available before season {season}"
            )
        oos = walk_forward(
            self._features, prior_seasons, estimator=estimator, alpha=DEFAULT_ALPHA
        )
        if oos.empty:
            raise SlateUnavailableError(
                f"no calibration data is available before season {season}"
            )
        train = self._features[self._features["season"] < season]
        try:
            calibrator = Calibrator().fit(oos)
            model = GameModel(estimator=estimator, alpha=DEFAULT_ALPHA).fit(train)
        except (DegenerateFeatureError, ValueError) as exc:
            raise SlateUnavailableError(
                f"cannot train {estimator} for season {season}: {exc}"
            ) from exc
        return ModelBundle(model=model, calibrator=calibrator)

    def _bundle(self, season: int, estimator: str) -> ModelBundle:
        key = (season, estimator)
        with self._cache_lock:
            bundle = self._cache.get(key)
            if bundle is None:
                bundle = self._fit_bundle(season, estimator)
                self._cache[key] = bundle
            return bundle

    def slate(
        self,
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    ) -> pd.DataFrame:
        self._validate(season, week, estimator, edge_threshold)
        target = self._features[
            (self._features["season"] == season) & (self._features["week"] == week)
        ]
        if target.empty:
            raise SlateInputError(f"week {week} is not available for season {season}")
        bundle = self._bundle(season, estimator)
        preds = bundle.model.predict(target)
        probs_input = target.merge(preds, on="game_id", validate="one_to_one")
        probs = bundle.calibrator.predict(probs_input)
        slate = build_slate(target, preds, probs, edge_threshold=edge_threshold)
        if slate.empty:
            raise SlateNotFoundError(f"no games are available for season {season} week {week}")
        return slate

    def records(self, *args, **kwargs) -> list[dict]:
        slate = self.slate(*args, **kwargs).astype(object)
        return slate.where(pd.notna(slate), None).to_dict(orient="records")

    def csv(self, *args, **kwargs) -> str:
        return self.slate(*args, **kwargs).to_csv(index=False, na_rep="")
```

- [ ] **Step 8: Run service tests and the existing market tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_service.py tests/test_compare.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit the service**

```powershell
git add pyproject.toml src/nfl_game/web/__init__.py src/nfl_game/web/service.py tests/test_web_service.py
git commit -m "feat: add cached web slate service"
```

---

### Task 2: Authentication Sessions, Throttling, Middleware, and Login

**Files:**

- Create: `src/nfl_game/web/session.py`
- Create: `src/nfl_game/web/throttle.py`
- Create: `src/nfl_game/web/auth.py`
- Create: `src/nfl_game/web/login.py`
- Create: `tests/test_web_session.py`
- Create: `tests/test_web_throttle.py`
- Create: `tests/test_web_auth.py`
- Create: `tests/test_web_login.py`

**Interfaces:**

- Consumes: FastAPI `Request`, `HTMLResponse`, `JSONResponse`, and Pydantic `BaseModel`.
- Produces: `SessionStore.create() -> str`, `SessionStore.get(token) -> Session | None`, `LoginThrottle`, `AccessCodeMiddleware(app, store, enabled)`, and `add_login_routes(app, store, access_code, throttle=None)`.

- [ ] **Step 1: Write failing session-store tests**

```python
from nfl_game.web.session import COOKIE_NAME, SessionStore


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_created_token_resolves_to_a_session():
    store = SessionStore()
    token = store.create()
    assert len(token) >= 32
    assert store.get(token) is not None


def test_expired_token_is_removed():
    clock = Clock()
    store = SessionStore(ttl=10, clock=clock)
    token = store.create()
    clock.now += 11
    assert store.get(token) is None


def test_store_evicts_oldest_at_capacity():
    clock = Clock()
    store = SessionStore(max_sessions=2, clock=clock)
    oldest = store.create()
    clock.now += 1
    store.create()
    clock.now += 1
    store.create()
    assert store.get(oldest) is None
    assert len(store) == 2


```

- [ ] **Step 2: Run the session tests and confirm the missing-module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_session.py -v
```

Expected: `ModuleNotFoundError` for `nfl_game.web.session`.

- [ ] **Step 3: Implement the bounded token-only session store**

Use the fantasy store's lazy-expiry pattern without `DraftState`:

```python
import secrets
import time
from dataclasses import dataclass

COOKIE_NAME = "nfl_session"
SESSION_TTL_SECONDS = 6 * 3600
MAX_SESSIONS = 500


@dataclass
class Session:
    last_seen: float


class SessionStore:
    def __init__(self, ttl=SESSION_TTL_SECONDS, max_sessions=MAX_SESSIONS, clock=time.monotonic):
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl
        self._max = max_sessions
        self._clock = clock

    def __len__(self):
        return len(self._sessions)

    def create(self) -> str:
        self._evict_expired()
        while len(self._sessions) >= self._max:
            self._evict_oldest()
        token = secrets.token_urlsafe(32)
        self._sessions[token] = Session(last_seen=self._clock())
        return token

    def get(self, token: str | None) -> Session | None:
        self._evict_expired()
        session = self._sessions.get(token) if token else None
        if session is not None:
            session.last_seen = self._clock()
        return session

    def _evict_expired(self) -> None:
        cutoff = self._clock() - self._ttl
        expired = [
            token
            for token, session in self._sessions.items()
            if session.last_seen < cutoff
        ]
        for token in expired:
            del self._sessions[token]

    def _evict_oldest(self) -> None:
        oldest = min(
            self._sessions,
            key=lambda token: self._sessions[token].last_seen,
        )
        del self._sessions[oldest]
```

- [ ] **Step 4: Run session tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_session.py -v
```

Expected: all pass.

- [ ] **Step 5: Port throttle tests and implementation from the fantasy app**

Create tests that pin:

```python
def test_three_failures_lock_the_ip_for_configured_window():
    clock = Clock()
    throttle = LoginThrottle(max_failures=3, lockout=60, clock=clock)
    for _ in range(3):
        throttle.record_failure("203.0.113.4")
    assert throttle.retry_after("203.0.113.4") == 61
    clock.now += 61
    assert throttle.retry_after("203.0.113.4") == 0


def test_success_clears_failures():
    throttle = LoginThrottle(max_failures=3)
    throttle.record_failure("203.0.113.4")
    throttle.record_success("203.0.113.4")
    assert throttle.retry_after("203.0.113.4") == 0


def test_client_ip_prefers_first_forwarded_address():
    request = SimpleNamespace(
        headers={"x-forwarded-for": "203.0.113.4, 10.0.0.2"},
        client=SimpleNamespace(host="10.0.0.1"),
    )
    assert client_ip(request) == "203.0.113.4"
```

Port `LoginThrottle` and `client_ip` from `nfl_ffm.draft.throttle`, changing only the import/module path. Preserve `MAX_FAILURES = 5`, `LOCKOUT_SECONDS = 60`, and `MAX_TRACKED_IPS = 5000`.

- [ ] **Step 6: Run throttle tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_throttle.py -v
```

Expected: all pass.

- [ ] **Step 7: Write failing middleware tests**

Create a FastAPI fixture with `/`, `/api/options`, `/login`, and `/health`. Assert:

```python
def test_page_redirects_without_session():
    client, _ = protected_client()
    response = client.get("/")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_api_returns_json_401_without_session():
    client, _ = protected_client()
    response = client.get("/api/options")
    assert response.status_code == 401
    assert response.json() == {"error": "session expired"}


def test_login_and_health_are_public():
    client, _ = protected_client()
    assert client.get("/login").status_code == 200
    assert client.get("/health").status_code == 200


def test_valid_cookie_passes_through():
    client, store = protected_client()
    client.cookies.set(COOKIE_NAME, store.create())
    assert client.get("/api/options").status_code == 200


def test_disabled_middleware_allows_local_requests():
    client, _ = unprotected_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/options").status_code == 200
```

- [ ] **Step 8: Implement cookie parsing and pure-ASGI middleware**

```python
LOGIN_PATH = "/login"
PUBLIC_PATHS = {LOGIN_PATH, "/health"}


def read_cookie(scope, name: str) -> str | None:
    headers = dict(scope.get("headers") or [])
    raw = headers.get(b"cookie", b"").decode("latin-1")
    for part in raw.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == name:
            return value
    return None


class AccessCodeMiddleware:
    def __init__(self, app, store: SessionStore, enabled: bool):
        self.app = app
        self.store = store
        self.enabled = enabled

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in PUBLIC_PATHS or self.store.get(read_cookie(scope, COOKIE_NAME)) is not None:
            await self.app(scope, receive, send)
            return
        await self._challenge(send, path)
```

Add the exact challenge method to `AccessCodeMiddleware`:

```python
    @staticmethod
    async def _challenge(send, path: str) -> None:
        if path.startswith("/api/"):
            status = 401
            headers = [(b"content-type", b"application/json")]
            body = b'{"error":"session expired"}'
        else:
            status = 303
            headers = [
                (b"location", LOGIN_PATH.encode()),
                (b"content-type", b"text/plain; charset=utf-8"),
            ]
            body = b""
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
```

- [ ] **Step 9: Run middleware tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_auth.py -v
```

Expected: all pass.

- [ ] **Step 10: Write failing login tests**

Create an HTTPS `TestClient` and pin:

```python
def test_correct_code_sets_secure_session_cookie():
    client, store = login_client("letmein")
    response = client.post("/login", json={"code": "letmein"})
    assert response.status_code == 200
    assert len(store) == 1
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "secure" in header
    assert "samesite=lax" in header
    assert "path=/" in header


def test_wrong_and_non_ascii_codes_are_clean_401s():
    client, store = login_client("letmein")
    assert client.post("/login", json={"code": "wrong"}).status_code == 401
    assert client.post("/login", json={"code": "letmein\u2019"}).status_code == 401
    assert len(store) == 0


def test_repeated_failures_return_429():
    throttle = LoginThrottle(max_failures=3, lockout=60)
    client, _ = login_client("letmein", throttle)
    for _ in range(3):
        assert client.post("/login", json={"code": "wrong"}).status_code == 401
    assert client.post("/login", json={"code": "wrong"}).status_code == 429
```

- [ ] **Step 11: Implement the game-specific login page and routes**

Port the fantasy login page, changing the heading/title to `NFL model access`. Define:

```python
class LoginRequest(BaseModel):
    code: str


def add_login_routes(
    app,
    store: SessionStore,
    access_code: str,
    throttle: LoginThrottle | None = None,
) -> None:
    throttle = throttle if throttle is not None else LoginThrottle()

    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return LOGIN_PAGE

    @app.post("/login")
    def login(req: LoginRequest, request: Request):
        ip = client_ip(request)
        wait = throttle.retry_after(ip)
        if wait:
            return JSONResponse(
                {"error": f"Too many attempts. Try again in {wait}s."},
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        if not hmac.compare_digest(req.code.encode(), access_code.encode()):
            throttle.record_failure(ip)
            return JSONResponse({"error": "Incorrect code"}, status_code=401)
        throttle.record_success(ip)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            COOKIE_NAME,
            store.create(),
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
        )
        return response
```

- [ ] **Step 12: Run all authentication tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_session.py tests/test_web_throttle.py tests/test_web_auth.py tests/test_web_login.py -v
```

Expected: all pass.

- [ ] **Step 13: Commit authentication**

```powershell
git add src/nfl_game/web/session.py src/nfl_game/web/throttle.py src/nfl_game/web/auth.py src/nfl_game/web/login.py tests/test_web_session.py tests/test_web_throttle.py tests/test_web_auth.py tests/test_web_login.py
git commit -m "feat: add fail-closed web authentication"
```

---

### Task 3: FastAPI Dashboard, JSON API, and CSV Download

**Files:**

- Create: `src/nfl_game/web/app.py`
- Create: `tests/test_webapp.py`
- Modify: `src/nfl_game/web/__init__.py`

**Interfaces:**

- Consumes: `SlateService`, `SlateInputError`, `SlateUnavailableError`, `SessionStore`, `AccessCodeMiddleware`, and `add_login_routes`.
- Produces: `create_app(service: SlateService, access_code: str | None) -> FastAPI`.

- [ ] **Step 1: Write failing route tests with a fake service**

```python
import math

from fastapi.testclient import TestClient

from nfl_game.web.app import create_app
from nfl_game.web.service import (
    SlateInputError,
    SlateNotFoundError,
    SlateUnavailableError,
)


class FakeService:
    def options(self):
        return {
            "seasons": [2024, 2025],
            "weeks": [1, 3],
            "estimators": ["gbm", "ridge"],
            "default_estimator": "ridge",
            "default_edge_threshold": 2.0,
            "latest": {"season": 2025, "week": 3},
        }

    def weeks(self, season):
        if season != 2025:
            raise SlateInputError(f"season {season} is not available")
        return [1, 3]

    def records(self, season, week, estimator, edge_threshold):
        if week == 2:
            raise SlateInputError("week 2 is not available for season 2025")
        if week == 3 and estimator == "gbm":
            raise SlateUnavailableError("cannot train gbm for season 2025")
        if week == 4:
            raise SlateNotFoundError("no games are available for season 2025 week 4")
        if week == 5:
            raise RuntimeError("database password leaked in internal trace")
        return [{
            "game_id": "2025_01_AAA_BBB",
            "season": season,
            "week": week,
            "away_team": "AAA",
            "home_team": "BBB",
            "model_spread": 4.0,
            "market_spread": 2.5,
            "spread_gap": 1.5,
            "cover_prob": 0.55,
            "model_total": 46.0,
            "market_total": None,
            "total_gap": None,
            "over_prob": None,
            "edge_flag": 0,
        }]

    def csv(self, season, week, estimator, edge_threshold):
        return "game_id,season,week\\n2025_01_AAA_BBB,2025,1\\n"


def client():
    return TestClient(create_app(FakeService(), access_code=None))


def test_page_contains_all_controls_and_disclaimer():
    html = client().get("/").text
    for control_id in ("season", "week", "estimator", "edge", "run", "download"):
        assert f'id="{control_id}"' in html
    assert "home-team margins" in html
    assert "not betting advice" in html


def test_options_and_weeks_routes():
    assert client().get("/api/options").json()["latest"] == {"season": 2025, "week": 3}
    assert client().get("/api/weeks", params={"season": 2025}).json() == {"weeks": [1, 3]}


def test_slate_json_preserves_nulls():
    response = client().get(
        "/api/slate",
        params={"season": 2025, "week": 1, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert response.status_code == 200
    assert response.json()["games"][0]["market_total"] is None


def test_csv_has_matching_filename_and_content_type():
    response = client().get(
        "/api/slate.csv",
        params={"season": 2025, "week": 1, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="slate_2025_wk01_ridge.csv"' in response.headers["content-disposition"]


def test_input_unavailable_and_empty_errors_are_client_safe():
    bad_week = client().get(
        "/api/slate",
        params={"season": 2025, "week": 2, "estimator": "ridge", "edge_threshold": 2.0},
    )
    unavailable = client().get(
        "/api/slate",
        params={"season": 2025, "week": 3, "estimator": "gbm", "edge_threshold": 2.0},
    )
    empty = client().get(
        "/api/slate",
        params={"season": 2025, "week": 4, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert bad_week.status_code == 422
    assert bad_week.json() == {"error": "week 2 is not available for season 2025"}
    assert unavailable.status_code == 409
    assert empty.status_code == 404
    assert "traceback" not in unavailable.text.lower()


def test_unexpected_error_is_generic_and_hides_internal_message():
    safe_client = TestClient(
        create_app(FakeService(), access_code=None),
        raise_server_exceptions=False,
    )
    response = safe_client.get(
        "/api/slate",
        params={"season": 2025, "week": 5, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert response.status_code == 500
    assert response.json() == {"error": "Unexpected server error"}
    assert "database password" not in response.text
```

- [ ] **Step 2: Run the route tests and confirm the missing-app failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_webapp.py -v
```

Expected: `ModuleNotFoundError` for `nfl_game.web.app`.

- [ ] **Step 3: Implement the app factory and API routes**

```python
import logging

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response

from nfl_game.web.auth import AccessCodeMiddleware
from nfl_game.web.login import add_login_routes
from nfl_game.web.service import (
    DEFAULT_EDGE_THRESHOLD,
    SlateInputError,
    SlateNotFoundError,
    SlateService,
    SlateUnavailableError,
)
from nfl_game.web.session import SessionStore

logger = logging.getLogger(__name__)


def create_app(service: SlateService, access_code: str | None) -> FastAPI:
    app = FastAPI(title="NFL game model")
    store = SessionStore()

    @app.exception_handler(SlateInputError)
    async def input_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(SlateNotFoundError)
    async def not_found_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(SlateUnavailableError)
    async def unavailable_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.exception("Unhandled web request failure")
        return JSONResponse({"error": "Unexpected server error"}, status_code=500)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/options")
    def options():
        return service.options()

    @app.get("/api/weeks")
    def weeks(season: int):
        return {"weeks": service.weeks(season)}

    @app.get("/api/slate")
    def slate(
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = Query(DEFAULT_EDGE_THRESHOLD),
    ):
        return {"games": service.records(season, week, estimator, edge_threshold)}

    @app.get("/api/slate.csv")
    def slate_csv(
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = Query(DEFAULT_EDGE_THRESHOLD),
    ):
        content = service.csv(season, week, estimator, edge_threshold)
        filename = f"slate_{season}_wk{week:02d}_{estimator}.csv"
        return Response(
            content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if access_code is not None:
        add_login_routes(app, store, access_code)
    app.add_middleware(
        AccessCodeMiddleware,
        store=store,
        enabled=access_code is not None,
    )
    return app
```

- [ ] **Step 4: Add the complete embedded dashboard**

Define one `PAGE` constant with:

```html
<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Game Model</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem; color: #191919; }
  main { max-width: 72rem; margin: auto; }
  .controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: end; }
  label { display: grid; gap: .25rem; }
  select, input, button { font: inherit; padding: .45rem; }
  button { cursor: pointer; }
  #message { min-height: 1.4rem; color: #a00; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; white-space: nowrap; }
  th, td { border-bottom: 1px solid #ddd; padding: .5rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .edge { font-weight: 700; color: #087443; }
  .note { color: #555; font-size: .9rem; }
</style>
<main>
  <h1>NFL Game Model</h1>
  <p>Weekly model-versus-market margins and totals.</p>
  <div class="controls">
    <label>Season <select id="season"></select></label>
    <label>Week <select id="week"></select></label>
    <label>Estimator <select id="estimator"></select></label>
    <label>Edge threshold <input id="edge" type="number" min="0" step="0.5"></label>
    <button id="run" type="button">Run slate</button>
    <button id="download" type="button">Download CSV</button>
  </div>
  <p id="message" role="status"></p>
  <div class="table-wrap"><table id="results"></table></div>
  <p class="note">Spreads are home-team margins. An edge flag shows model/market
  disagreement and is not betting advice.</p>
</main>
```

The script must:

- fetch `/api/options` on `DOMContentLoaded`;
- populate season/week/estimator controls and select the returned latest/default values;
- fetch `` `/api/weeks?season=${encodeURIComponent(season)}` `` after season changes and select that season's latest week;
- disable `run` during `/api/slate` requests and render a readable error on non-2xx responses;
- render the ten user-facing columns in the design, formatting null as `n/a` and probabilities as percentages;
- assign class `edge` only when `edge_flag === 1`;
- navigate to `/api/slate.csv` with the exact active query when Download CSV is clicked.

Use `textContent` and DOM node creation for returned values; do not interpolate API strings into `innerHTML`. Append this script to `PAGE`:

```html
<script>
const season = document.getElementById('season');
const week = document.getElementById('week');
const estimator = document.getElementById('estimator');
const edge = document.getElementById('edge');
const runButton = document.getElementById('run');
const downloadButton = document.getElementById('download');
const message = document.getElementById('message');
const results = document.getElementById('results');

function replaceOptions(select, values, selected) {
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = String(value);
    option.selected = value === selected;
    select.appendChild(option);
  }
}

function queryString() {
  return new URLSearchParams({
    season: season.value,
    week: week.value,
    estimator: estimator.value,
    edge_threshold: edge.value,
  }).toString();
}

async function jsonOrError(url) {
  const response = await fetch(url);
  if (response.status === 401) {
    window.location = '/login';
    throw new Error('Session expired');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function formatted(value, kind) {
  if (value === null || value === undefined) return 'n/a';
  if (kind === 'probability') return `${(Number(value) * 100).toFixed(1)}%`;
  if (kind === 'signed') {
    const number = Number(value);
    return `${number >= 0 ? '+' : ''}${number.toFixed(1)}`;
  }
  return Number(value).toFixed(1);
}

function renderGames(games) {
  const columns = [
    ['Game', game => `${game.away_team} @ ${game.home_team}`],
    ['Model', game => formatted(game.model_spread, 'signed')],
    ['Market', game => formatted(game.market_spread, 'signed')],
    ['Gap', game => formatted(game.spread_gap, 'signed')],
    ['Cover%', game => formatted(game.cover_prob, 'probability')],
    ['Model O/U', game => formatted(game.model_total, 'number')],
    ['Market O/U', game => formatted(game.market_total, 'number')],
    ['Gap', game => formatted(game.total_gap, 'signed')],
    ['Over%', game => formatted(game.over_prob, 'probability')],
    ['Edge', game => game.edge_flag === 1 ? '*' : ''],
  ];
  results.replaceChildren();
  const header = document.createElement('tr');
  for (const [label] of columns) {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  }
  results.appendChild(header);
  for (const game of games) {
    const row = document.createElement('tr');
    for (const [, value] of columns) {
      const cell = document.createElement('td');
      cell.textContent = value(game);
      row.appendChild(cell);
    }
    if (game.edge_flag === 1) row.className = 'edge';
    results.appendChild(row);
  }
}

async function loadWeeks(runAfter = false) {
  const body = await jsonOrError(`/api/weeks?season=${encodeURIComponent(season.value)}`);
  const latest = body.weeks[body.weeks.length - 1];
  replaceOptions(week, body.weeks, latest);
  if (runAfter) await runSlate();
}

async function runSlate() {
  runButton.disabled = true;
  message.textContent = 'Loading…';
  try {
    const body = await jsonOrError(`/api/slate?${queryString()}`);
    renderGames(body.games);
    message.textContent = `${body.games.length} games`;
  } catch (error) {
    results.replaceChildren();
    message.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
}

async function initialize() {
  try {
    const options = await jsonOrError('/api/options');
    replaceOptions(season, options.seasons, options.latest.season);
    replaceOptions(week, options.weeks, options.latest.week);
    replaceOptions(estimator, options.estimators, options.default_estimator);
    edge.value = String(options.default_edge_threshold);
    await runSlate();
  } catch (error) {
    message.textContent = error.message;
  }
}

season.addEventListener('change', () => loadWeeks(true));
runButton.addEventListener('click', runSlate);
downloadButton.addEventListener('click', () => {
  window.location = `/api/slate.csv?${queryString()}`;
});
document.addEventListener('DOMContentLoaded', initialize);
</script>
```

- [ ] **Step 5: Run route tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_webapp.py -v
```

Expected: all pass.

- [ ] **Step 6: Export the app factory and run web/service/auth tests**

Add to `src/nfl_game/web/__init__.py`:

```python
from nfl_game.web.app import create_app

__all__ = [
    "SlateInputError",
    "SlateNotFoundError",
    "SlateService",
    "SlateUnavailableError",
    "create_app",
]
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_service.py tests/test_web_session.py tests/test_web_throttle.py tests/test_web_auth.py tests/test_web_login.py tests/test_webapp.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit the web app**

```powershell
git add src/nfl_game/web/__init__.py src/nfl_game/web/app.py tests/test_webapp.py
git commit -m "feat: add NFL slate web dashboard"
```

---

### Task 4: Fail-Closed Runtime and Uvicorn Entry Point

**Files:**

- Create: `src/nfl_game/web/runtime.py`
- Create: `scripts/game_app.py`
- Create: `tests/test_web_runtime.py`

**Interfaces:**

- Consumes: `PROCESSED_DIR`, `SlateService.from_parquet`, and `create_app`.
- Produces: `RuntimeConfig`, `resolve_runtime(no_auth, environ)`, `load_app(config, dataset_path)`, and `scripts/game_app.py::main(argv=None)`.

- [ ] **Step 1: Write failing fail-closed runtime tests**

```python
from pathlib import Path

import pytest

from nfl_game.web.runtime import RuntimeConfigError, resolve_runtime


def test_default_startup_requires_access_code():
    with pytest.raises(RuntimeConfigError, match="ACCESS_CODE is required"):
        resolve_runtime(no_auth=False, environ={})


def test_protected_runtime_binds_all_interfaces():
    config = resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "letmein", "PORT": "9000"})
    assert config.access_code == "letmein"
    assert config.host == "0.0.0.0"
    assert config.port == 9000


def test_explicit_no_auth_binds_loopback_only():
    config = resolve_runtime(no_auth=True, environ={})
    assert config.access_code is None
    assert config.host == "127.0.0.1"
    assert config.port == 8000


def test_no_auth_rejects_access_code_to_avoid_ambiguous_intent():
    with pytest.raises(RuntimeConfigError, match="cannot be combined"):
        resolve_runtime(no_auth=True, environ={"ACCESS_CODE": "letmein"})


def test_blank_access_code_is_missing():
    with pytest.raises(RuntimeConfigError, match="ACCESS_CODE is required"):
        resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "   "})


def test_invalid_port_is_configuration_error():
    with pytest.raises(RuntimeConfigError, match="PORT"):
        resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "letmein", "PORT": "abc"})
```

- [ ] **Step 2: Run tests and confirm the missing-runtime failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_runtime.py -v
```

Expected: `ModuleNotFoundError` for `nfl_game.web.runtime`.

- [ ] **Step 3: Implement runtime resolution and dataset loading**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nfl_game.web.app import create_app
from nfl_game.web.service import SlateService


class RuntimeConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    access_code: str | None
    host: str
    port: int


def resolve_runtime(no_auth: bool, environ: Mapping[str, str]) -> RuntimeConfig:
    raw_code = environ.get("ACCESS_CODE", "")
    access_code = raw_code.strip() or None
    if no_auth and access_code is not None:
        raise RuntimeConfigError("--no-auth cannot be combined with ACCESS_CODE")
    if not no_auth and access_code is None:
        raise RuntimeConfigError(
            "ACCESS_CODE is required; use --no-auth only for loopback local development"
        )
    try:
        port = int(environ.get("PORT", "8000"))
    except ValueError as exc:
        raise RuntimeConfigError("PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise RuntimeConfigError("PORT must be between 1 and 65535")
    return RuntimeConfig(
        access_code=None if no_auth else access_code,
        host="127.0.0.1" if no_auth else "0.0.0.0",
        port=port,
    )


def load_app(config: RuntimeConfig, dataset_path: str | Path):
    path = Path(dataset_path)
    if not path.is_file():
        raise RuntimeConfigError(f"packaged dataset not found: {path}")
    try:
        service = SlateService.from_parquet(path)
    except Exception as exc:
        raise RuntimeConfigError(f"cannot load packaged dataset {path}: {exc}") from exc
    return create_app(service, access_code=config.access_code)
```

Add these exact loading tests:

```python
def test_load_app_rejects_missing_dataset(tmp_path):
    config = resolve_runtime(no_auth=True, environ={})
    with pytest.raises(RuntimeConfigError, match="packaged dataset not found"):
        load_app(config, tmp_path / "missing.parquet")


def test_load_app_wraps_parquet_read_failure(tmp_path, monkeypatch):
    dataset = tmp_path / "broken.parquet"
    dataset.write_bytes(b"not parquet")

    def fail(path):
        raise ValueError("invalid parquet footer")

    monkeypatch.setattr("nfl_game.web.runtime.SlateService.from_parquet", fail)
    config = resolve_runtime(no_auth=True, environ={})
    with pytest.raises(RuntimeConfigError, match="cannot load packaged dataset") as caught:
        load_app(config, dataset)
    assert "invalid parquet footer" in str(caught.value)
```

- [ ] **Step 4: Implement the thin entry point**

```python
"""Serve the packaged NFL game model dashboard."""

import argparse
import os

import uvicorn

from nfl_game.paths import PROCESSED_DIR
from nfl_game.web.runtime import RuntimeConfigError, load_app, resolve_runtime


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="local-only: disable login and bind to 127.0.0.1",
    )
    args = parser.parse_args(argv)
    try:
        config = resolve_runtime(args.no_auth, os.environ)
        app = load_app(config, PROCESSED_DIR / "game_features.parquet")
    except RuntimeConfigError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run runtime, web, and smoke tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_runtime.py tests/test_webapp.py tests/test_smoke.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit runtime behavior**

```powershell
git add src/nfl_game/web/runtime.py scripts/game_app.py tests/test_web_runtime.py
git commit -m "feat: add fail-closed web runtime"
```

---

### Task 5: Deployment Artifact, Docker Image, and Render Blueprint

**Files:**

- Create: `Dockerfile`
- Create: `render.yaml`
- Modify: `.gitignore`
- Add: `data/processed/game_features.parquet`

**Interfaces:**

- Consumes: `scripts/game_app.py`, package metadata/source, and the single packaged parquet artifact.
- Produces: Docker image listening on injected `PORT` and Render service `ashburn-nfl`.

**TDD exception:** The user approved behavioral verification instead of source-text change-detector tests for `.gitignore`, `Dockerfile`, and `render.yaml`. Validate these configuration artifacts by building and running the image, inspecting the tracked artifact contract, and performing external Render checks.

- [ ] **Step 1: Narrow the processed-data ignore**

Append:

```gitignore
!data/processed/game_features.parquet
```

Confirm only that artifact becomes visible:

```powershell
git status --short --untracked-files=all data/processed
```

Expected: `game_features.parquet` appears; backup parquet and generated slate files remain ignored.

- [ ] **Step 2: Create the production Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts
COPY data/processed/game_features.parquet ./data/processed/game_features.parquet

EXPOSE 8000
CMD ["python", "scripts/game_app.py"]
```

- [ ] **Step 3: Create the Render Blueprint**

```yaml
services:
  - type: web
    name: ashburn-nfl
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    envVars:
      - key: ACCESS_CODE
        sync: false
```

- [ ] **Step 4: Inspect the tracked artifact contract**

```powershell
git check-ignore -v data/processed/game_features_fixed.parquet
git status --short --untracked-files=all data/processed
git diff --check
```

Expected: backup artifacts remain ignored; only `game_features.parquet` is untracked/trackable; no whitespace errors are reported.

- [ ] **Step 5: Build and behaviorally smoke-test the container**

```powershell
docker build -t nfl-game-web .
docker run --rm nfl-game-web
docker run --rm -d --name nfl-game-web-smoke -p 18000:8000 -e ACCESS_CODE=smoke-test-only nfl-game-web
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:18000/health
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" http://127.0.0.1:18000/
docker stop nfl-game-web-smoke
```

Expected: the first run exits nonzero because `ACCESS_CODE` is required; with an access code, health is `200`, root is `303` with redirect to `/login`, and the container stops cleanly.

- [ ] **Step 6: Commit deployment packaging**

```powershell
git add .gitignore Dockerfile render.yaml data/processed/game_features.parquet
git commit -m "deploy: package NFL game web service"
```
---

### Task 6: Documentation and Full Local Verification

**Files:**

- Modify: `README.md`

**Interfaces:**

- Consumes: implemented CLI, Docker, Render Blueprint, and refresh workflow.
- Produces: exact operator instructions for local mode, protected mode, dataset refresh, deployment, and security verification.

- [ ] **Step 1: Add local run instructions**

Document:

```powershell
# Explicit loopback-only local mode
.\.venv\Scripts\python.exe scripts\game_app.py --no-auth

# Protected local/network mode
$env:ACCESS_CODE = "local-test-only"
.\.venv\Scripts\python.exe scripts\game_app.py
```

State that plain HTTP browsers will not retain the secure session cookie; use HTTPS for an end-to-end protected-browser test, while TestClient covers cookie behavior.

- [ ] **Step 2: Add the immutable artifact refresh workflow**

Document these exact steps:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py --start-season 2016 --end-season 2026
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
git add data/processed/game_features.parquet
git commit -m "data: refresh packaged game features"
```

State explicitly that the website has no refresh endpoint and cannot mutate the artifact.

- [ ] **Step 3: Add the deployment security gate**

Document the pre-DNS external checks:

```powershell
$renderBase = (Read-Host "Paste the exact Render service URL").TrimEnd("/")
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" "$renderBase/"
curl.exe -s -o NUL -w "%{http_code}" "$renderBase/api/options"
curl.exe -s -o NUL -w "%{http_code}" -H "Content-Type: application/json" -d "{\"code\":\"definitely-wrong\"}" "$renderBase/login"
```

Expected: `303` to `/login`, `401`, and `401`. Require the same checks after a Render restart and again on `https://nfl.ashburn-capital.com`. State that DNS cutover must not happen until the Render-hostname checks pass.

- [ ] **Step 4: Run the complete test and style suite**

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Re-run the statistical acceptance baseline**

```powershell
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
```

Expected invariant:

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

Stop and investigate if any value moves; a web-only change must not alter these results.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md
git commit -m "docs: add NFL web app operations runbook"
```

---

### Post-Integration Task 7: Render Deployment, External Verification, and Custom Domain

**Files:**

- No repository files unless the platform exposes a configuration correction that must be recorded.

**Interfaces:**

- Consumes: pushed `master`, Render Blueprint, private `ACCESS_CODE`, and DNS access for `ashburn-capital.com`.
- Produces: verified production service at `https://nfl.ashburn-capital.com`.

Task 7 begins only after Tasks 1–6 pass, the final whole-branch review is clean, and `superpowers:finishing-a-development-branch` has integrated the feature into `master`. Deploy the integrated `master`, never the isolated feature branch.

- [ ] **Step 1: Verify integrated master and push the reviewed implementation**

```powershell
git status --short
git log --oneline origin/master..master
git push origin master
```

Expected: the primary checkout is on `master`, the worktree is clean, the reviewed feature commits are integrated, and the push succeeds.

- [ ] **Step 2: Create the Render Blueprint service**

In Render, create a Blueprint from `sbuermeyer1/nfl-games`, confirm the service name is `ashburn-nfl`, set a non-empty private `ACCESS_CODE`, and deploy. Do not add the custom domain yet.

- [ ] **Step 3: Verify fail-closed behavior on the Render hostname**

From an external client with no stored cookies:

```powershell
$renderBase = (Read-Host "Paste the exact Render service URL").TrimEnd("/")
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" "$renderBase/"
curl.exe -s -o NUL -w "%{http_code}" "$renderBase/api/options"
curl.exe -s -o NUL -w "%{http_code}" -H "Content-Type: application/json" -d "{\"code\":\"definitely-wrong\"}" "$renderBase/login"
```

Expected: root returns `303` to `/login`; API returns `401`; wrong code returns `401`.

- [ ] **Step 4: Verify successful login without exposing the code in logs**

Open the Render hostname in a browser, enter the configured code interactively, confirm the latest slate renders, change the estimator and threshold, and download CSV. Confirm the downloaded rows/order match the table.

- [ ] **Step 5: Restart the Render service and repeat unauthenticated checks**

After restart, repeat Step 3 from a cookie-free client. Expected results must remain `303`, `401`, `401`; a previous session cookie must no longer authorize API access.

- [ ] **Step 6: Add and verify the custom domain**

Add `nfl.ashburn-capital.com` in Render, copy Render's exact CNAME target into the DNS manager for `ashburn-capital.com`, and wait for Render's domain verification and TLS certificate.

- [ ] **Step 7: Run final custom-domain verification**

```powershell
curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" https://nfl.ashburn-capital.com/
curl.exe -s -o NUL -w "%{http_code}" https://nfl.ashburn-capital.com/api/options
curl.exe -s -o NUL -w "%{http_code}" -H "Content-Type: application/json" -d "{\"code\":\"definitely-wrong\"}" https://nfl.ashburn-capital.com/login
```

Expected: `303` to `/login`, `401`, and `401`. Then sign in through the browser and verify the dashboard and CSV one final time.

- [ ] **Step 8: Record production verification**

Add a dated README note only if the project convention is to record deployments; otherwise record the Render hostname, custom-domain verification, restart check, and final smoke-test result in the task handoff without changing the repository.
