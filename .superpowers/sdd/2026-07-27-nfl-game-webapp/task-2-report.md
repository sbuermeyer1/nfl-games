# Task 2 report: fail-closed web authentication

## Summary

Added only the requested web authentication building blocks:

- bounded, token-only in-memory sessions scoped to the NFL game app;
- deterministic per-IP failed-login throttling;
- a pure-ASGI access-code gate with explicit `enabled` configuration;
- a game-specific access-code login page and route that issues hardened secure
  cookies.

The login route fails closed when the configured access code is empty. It uses
byte comparisons with `hmac.compare_digest`, so non-ASCII guesses receive the
same clean authentication failure instead of a `TypeError`.

## Files

- `src/nfl_game/web/session.py`
- `src/nfl_game/web/throttle.py`
- `src/nfl_game/web/auth.py`
- `src/nfl_game/web/login.py`
- `tests/test_web_session.py`
- `tests/test_web_throttle.py`
- `tests/test_web_auth.py`
- `tests/test_web_login.py`

## Red-green evidence

Every production module was added after its focused test first failed:

| Component | Red evidence | Green evidence |
| --- | --- | --- |
| Session store | `test_web_session.py`: `ModuleNotFoundError: nfl_game.web.session` | 3 passed |
| Login throttle | `test_web_throttle.py`: `ModuleNotFoundError: nfl_game.web.throttle` | 3 passed |
| ASGI middleware | Initial collection exposed the missing declared FastAPI dependency; after installing the project dependencies, `ModuleNotFoundError: nfl_game.web.auth` | 5 passed |
| Login routes | `test_web_login.py`: `ModuleNotFoundError: nfl_game.web.login` | 4 passed |

Focused combined verification:

```text
pytest tests/test_web_session.py tests/test_web_throttle.py \
       tests/test_web_auth.py tests/test_web_login.py -v
15 passed, 1 warning
```

Final project verification:

```text
pytest -v
168 passed, 2 warnings
ruff check src tests
All checks passed!
git diff --check
exit 0
```

All pytest commands used `LOKY_MAX_CPU_COUNT=1` and an isolated worktree-local
`--basetemp` under this task directory.

## Commit

Implementation commit: `0e4f094b5f9033a43f5ad50fb39a9ef763be164a`
(`feat: add fail-closed web authentication`)

## Self-review

- Confirmed the session store expires lazily, refreshes active sessions, and
  evicts the least-recently-used session at capacity.
- Confirmed throttle timing is clock-injected and deterministic; tracked IPs
  remain bounded without letting an attacker immediately evict an active lock.
- Confirmed missing sessions receive JSON `401` for API requests and a `303`
  redirect for pages; login and health remain public.
- Confirmed login cookies are `HttpOnly`, `Secure`, `SameSite=Lax`, and scoped
  to `/`; no account, persistence, dashboard, model-control, or deployment
  work was added.
- Added explicit coverage for an empty configured access code so this task does
  not introduce an authentication-disabled production default.

## Concerns

The final test run has two pre-existing/non-code warnings: a Starlette
`TestClient` deprecation warning emitted by the installed framework version,
and the expected runtime warning from the existing degenerate-backtest test.
`git diff --check` also prints Windows LF-to-CRLF conversion notices but exits
successfully with no whitespace errors.
