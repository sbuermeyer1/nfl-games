"""Access-code login page and routes for the NFL game web app."""

import hmac

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from nfl_game.web.session import COOKIE_NAME, SESSION_TTL_SECONDS, SessionStore
from nfl_game.web.throttle import LoginThrottle, client_ip


class LoginRequest(BaseModel):
    code: str


LOGIN_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL model access</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; display: grid;
         place-items: center; min-height: 100vh; background: #111; color: #eee; }
  form { display: grid; gap: 12px; width: min(320px, 86vw); }
  h1 { font-size: 1.25rem; margin: 0 0 4px; }
  input, button { font-size: 1rem; padding: 10px; border-radius: 6px;
                  border: 1px solid #444; }
  input { background: #1c1c1c; color: #eee; }
  button { background: #2f6f4f; color: #fff; border: 0; cursor: pointer; }
  .err { color: #ff8a8a; min-height: 1.2em; margin: 0; font-size: 0.9rem; }
</style>
<form id="f" onsubmit="return submitCode(event)">
  <h1>NFL model access</h1>
  <input id="code" type="password" placeholder="Access code" autofocus>
  <button id="submit" type="submit">Enter</button>
  <p class="err" id="err"></p>
</form>
<script>
let loginPending = false;

async function submitCode(e) {
  e.preventDefault();
  if (loginPending) return false;
  const err = document.getElementById('err');
  const codeInput = document.getElementById('code');
  const submitButton = document.getElementById('submit');
  err.textContent = '';
  loginPending = true;
  codeInput.disabled = true;
  submitButton.disabled = true;
  try {
    const resp = await fetch('/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code: codeInput.value}),
    });
    let body;
    try {
      body = await resp.json();
    } catch (error) {
      throw new Error('Login response was not JSON');
    }
    if (resp.ok) {
      window.location = '/';
      return false;
    }
    if (resp.status >= 500) {
      err.textContent = 'Unable to sign in right now. Please try again.';
      return false;
    }
    err.textContent = body.error || (
      resp.status === 401
        ? 'Incorrect code'
        : 'Unable to sign in right now. Please try again.'
    );
  } catch (error) {
    err.textContent = 'Unable to sign in right now. Please try again.';
  } finally {
    loginPending = false;
    codeInput.disabled = false;
    submitButton.disabled = false;
  }
  return false;
}
</script>
"""


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
        successful = bool(access_code) and hmac.compare_digest(
            req.code.encode(), access_code.encode()
        )
        wait = throttle.check_and_record(ip, successful)
        if wait:
            return JSONResponse(
                {"error": f"Too many attempts. Try again in {wait}s."},
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        if not successful:
            return JSONResponse({"error": "Incorrect code"}, status_code=401)
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
