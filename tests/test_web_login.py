import json
import re
import threading

import pytest
import quickjs
from fastapi import FastAPI
from fastapi.testclient import TestClient

import nfl_game.web.login as login_module
from nfl_game.web.login import add_login_routes
from nfl_game.web.session import SessionStore
from nfl_game.web.throttle import LoginThrottle


def login_client(access_code: str, throttle: LoginThrottle | None = None):
    """Build the real login route over HTTPS so secure cookies can be exercised."""
    app = FastAPI()
    store = SessionStore()
    add_login_routes(app, store, access_code, throttle)
    return TestClient(app, base_url="https://testserver"), store


LOGIN_SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def login_page_state(responses, actions):
    """Execute the served login script with controlled DOM and fetch behavior."""
    http_client, _ = login_client("letmein")
    page = http_client.get("/login").text
    script = LOGIN_SCRIPT.search(page)
    assert script, "login page must contain an executable script"
    payload = json.dumps({"script": script.group(1), "responses": responses, "actions": actions})
    context = quickjs.Context()
    context.eval(f"const input = JSON.parse({json.dumps(payload)});")
    context.eval(LOGIN_HARNESS)
    while context.execute_pending_job():
        pass
    return json.loads(context.eval("globalThis.__state"))


LOGIN_HARNESS = r"""
const pending = {};
const calls = [];
const unhandled = [];
let prevented = 0;

class Element {
  constructor() {
    this.textContent = '';
    this.value = 'letmein';
    this.disabled = false;
  }
}

const nodes = {
  err: new Element(),
  code: new Element(),
  submit: new Element(),
};
globalThis.document = { getElementById: id => nodes[id] };
globalThis.window = { location: '' };

function makeResponse(spec) {
  return {
    status: spec.status ?? 200,
    ok: (spec.status ?? 200) >= 200 && (spec.status ?? 200) < 300,
    json: async () => {
      if (spec.nonJson) throw new Error('response was not JSON');
      return spec.body ?? {};
    },
  };
}

globalThis.fetch = (url, options) => {
  calls.push({url, body: options.body});
  const spec = input.responses[url];
  if (!spec) return Promise.reject(new Error(`Unexpected URL: ${url}`));
  if (spec.network) return Promise.reject(new Error('network unavailable'));
  if (spec.deferred) {
    return new Promise((resolve, reject) => { pending[url] = { resolve, reject }; });
  }
  return Promise.resolve(makeResponse(spec));
};

async function settle() {
  for (let index = 0; index < 12; index += 1) await Promise.resolve();
}

function invokeSubmit(wait) {
  const promise = Promise.resolve().then(() => submitCode({
    preventDefault: () => { prevented += 1; },
  })).catch(error => { unhandled.push(error.message); });
  return wait ? promise : Promise.resolve();
}

eval(input.script);

(async () => {
  for (const action of input.actions) {
    if (action.type === 'submit') await invokeSubmit(action.wait);
    if (action.type === 'resolve') pending['/login'].resolve(makeResponse(action.response));
    if (action.type === 'reject') pending['/login'].reject(new Error(action.message));
    if (action.type === 'settle') await settle();
  }
  await settle();
  globalThis.__state = JSON.stringify({
    calls,
    unhandled,
    prevented,
    error: nodes.err.textContent,
    codeDisabled: nodes.code.disabled,
    submitDisabled: nodes.submit.disabled,
    location: window.location,
  });
})().catch(error => { globalThis.__state = JSON.stringify({harnessError: error.stack}); });
"""


GENERIC_LOGIN_ERROR = "Unable to sign in right now. Please try again."


@pytest.mark.parametrize(
    "response_spec",
    [
        {"network": True},
        {"status": 200, "nonJson": True},
        {"status": 503, "body": {"error": "private upstream details"}},
    ],
    ids=["fetch-rejection", "non-json", "server-failure"],
)
def test_login_page_reports_generic_connection_or_server_failure(response_spec):
    """Catch transport or server failures becoming unhandled or misleading login errors."""
    state = login_page_state(
        {"/login": response_spec},
        [{"type": "submit", "wait": True}],
    )

    assert state["error"] == GENERIC_LOGIN_ERROR
    assert state["location"] == ""
    assert state["codeDisabled"] is False
    assert state["submitDisabled"] is False
    assert state["unhandled"] == []


def test_login_page_preserves_incorrect_code_message_for_json_401():
    """Catch a normal invalid credential being mislabeled as a connection failure."""
    state = login_page_state(
        {"/login": {"status": 401, "body": {"error": "Incorrect code"}}},
        [{"type": "submit", "wait": True}],
    )

    assert state["error"] == "Incorrect code"
    assert state["location"] == ""
    assert state["codeDisabled"] is False
    assert state["submitDisabled"] is False
    assert state["unhandled"] == []


def test_login_page_redirects_after_json_success_and_restores_controls():
    """Catch successful authentication failing to redirect or leaving controls locked."""
    state = login_page_state(
        {"/login": {"status": 200, "body": {"ok": True}}},
        [{"type": "submit", "wait": True}],
    )

    assert state["error"] == ""
    assert state["location"] == "/"
    assert state["codeDisabled"] is False
    assert state["submitDisabled"] is False
    assert state["unhandled"] == []


def test_login_page_blocks_duplicate_submission_while_request_is_pending():
    """Catch double submits issuing duplicate login attempts before the first completes."""
    state = login_page_state(
        {"/login": {"deferred": True}},
        [
            {"type": "submit", "wait": False},
            {"type": "settle"},
            {"type": "submit", "wait": False},
            {"type": "settle"},
        ],
    )

    assert len(state["calls"]) == 1
    assert state["prevented"] == 2
    assert state["codeDisabled"] is True
    assert state["submitDisabled"] is True
    assert state["error"] == ""
    assert state["unhandled"] == []


def test_correct_code_sets_secure_session_cookie():
    """Catch a successful login that omits its hardened browser session cookie."""
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
    """Catch invalid Unicode input crashing login or creating a session."""
    client, store = login_client("letmein")

    assert client.post("/login", json={"code": "wrong"}).status_code == 401
    assert client.post("/login", json={"code": "letmein\u2019"}).status_code == 401
    assert len(store) == 0


def test_repeated_failures_return_429():
    """Catch brute-force attempts that continue after the configured lockout."""
    throttle = LoginThrottle(max_failures=3, lockout=60)
    client, _ = login_client("letmein", throttle)

    for _ in range(3):
        assert client.post("/login", json={"code": "wrong"}).status_code == 401

    assert client.post("/login", json={"code": "wrong"}).status_code == 429


def test_empty_configured_code_never_authenticates():
    """Catch an absent production access code silently disabling authentication."""
    client, store = login_client("")

    assert client.post("/login", json={"code": ""}).status_code == 401
    assert len(store) == 0


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_lockout_response_exposes_exact_retry_after_value():
    """Catch a rate-limit response whose header disagrees with its lockout."""
    clock = Clock()
    throttle = LoginThrottle(max_failures=1, lockout=60, clock=clock)
    client, _ = login_client("letmein", throttle)

    assert client.post("/login", json={"code": "wrong"}).status_code == 401
    response = client.post("/login", json={"code": "wrong"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "61"
    assert response.json() == {"error": "Too many attempts. Try again in 61s."}


def test_concurrent_wrong_logins_allow_only_one_attempt_before_lockout(monkeypatch):
    """Catch two requests that both pass the lock check before either records failure."""
    barrier = threading.Barrier(2)
    original_compare = login_module.hmac.compare_digest

    def synchronized_compare(left, right):
        barrier.wait(timeout=2)
        return original_compare(left, right)

    monkeypatch.setattr(login_module.hmac, "compare_digest", synchronized_compare)
    app = FastAPI()
    store = SessionStore()
    throttle = LoginThrottle(max_failures=1, lockout=60, clock=Clock())
    add_login_routes(app, store, "letmein", throttle)
    statuses: list[int] = []

    def wrong_login():
        with TestClient(app, base_url="https://testserver") as client:
            statuses.append(client.post("/login", json={"code": "wrong"}).status_code)

    threads = [threading.Thread(target=wrong_login) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(statuses) == [401, 429]


def test_successful_login_clears_prior_failures_before_later_attempts():
    """Catch a success that leaves a previous wrong-code count in place."""
    clock = Clock()
    throttle = LoginThrottle(max_failures=2, lockout=60, clock=clock)
    client, _ = login_client("letmein", throttle)

    assert client.post("/login", json={"code": "wrong"}).status_code == 401
    assert client.post("/login", json={"code": "letmein"}).status_code == 200
    assert client.post("/login", json={"code": "wrong"}).status_code == 401
    assert client.post("/login", json={"code": "wrong"}).status_code == 401
    response = client.post("/login", json={"code": "wrong"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "61"
