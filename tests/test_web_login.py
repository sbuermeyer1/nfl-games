from fastapi import FastAPI
from fastapi.testclient import TestClient

from nfl_game.web.login import add_login_routes
from nfl_game.web.session import SessionStore
from nfl_game.web.throttle import LoginThrottle


def login_client(access_code: str, throttle: LoginThrottle | None = None):
    """Build the real login route over HTTPS so secure cookies can be exercised."""
    app = FastAPI()
    store = SessionStore()
    add_login_routes(app, store, access_code, throttle)
    return TestClient(app, base_url="https://testserver"), store


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
