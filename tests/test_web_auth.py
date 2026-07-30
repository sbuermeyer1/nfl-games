from fastapi import FastAPI
from fastapi.testclient import TestClient

from nfl_game.web.auth import AccessCodeMiddleware, read_cookie
from nfl_game.web.login import add_login_routes
from nfl_game.web.session import COOKIE_NAME, SessionStore


def _app():
    app = FastAPI()

    @app.get("/")
    def page():
        return {"page": True}

    @app.get("/api/options")
    def options():
        return {"options": True}

    @app.get("/login")
    def login():
        return {"login": True}

    @app.get("/health")
    def health():
        return {"health": True}

    return app


def protected_client():
    """Build the real ASGI app with its access-code gate enabled."""
    app = _app()
    store = SessionStore()
    app.add_middleware(AccessCodeMiddleware, store=store, enabled=True)
    return TestClient(app, follow_redirects=False), store


def unprotected_client():
    """Build the real ASGI app with an explicitly disabled local-only gate."""
    app = _app()
    store = SessionStore()
    app.add_middleware(AccessCodeMiddleware, store=store, enabled=False)
    return TestClient(app, follow_redirects=False), store


def test_page_redirects_without_session():
    """Catch a protected page that is reachable without a valid session."""
    client, _ = protected_client()

    response = client.get("/")

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_api_returns_json_401_without_session():
    """Catch an API request redirected to HTML rather than given its JSON error."""
    client, _ = protected_client()

    response = client.get("/api/options")

    assert response.status_code == 401
    assert response.json() == {"error": "session expired"}


def test_login_and_health_are_public():
    """Catch middleware that blocks login or deployment health checks."""
    client, _ = protected_client()

    assert client.get("/login").status_code == 200
    assert client.get("/health").status_code == 200


def test_valid_cookie_passes_through():
    """Catch middleware that rejects a currently valid session cookie."""
    client, store = protected_client()
    client.cookies.set(COOKIE_NAME, store.create())

    assert client.get("/api/options").status_code == 200


def test_disabled_middleware_allows_local_requests():
    """Catch the explicitly disabled local gate continuing to challenge requests."""
    client, _ = unprotected_client()

    assert client.get("/").status_code == 200
    assert client.get("/api/options").status_code == 200


def test_read_cookie_joins_duplicate_asgi_cookie_headers():
    """Catch an earlier session cookie being discarded by duplicate headers."""
    scope = {
        "headers": [
            (b"cookie", b"nfl_session=valid-token"),
            (b"cookie", b"theme=dark"),
        ]
    }

    assert read_cookie(scope, COOKIE_NAME) == "valid-token"


def test_login_issued_cookie_passes_through_middleware():
    """Catch a secure login cookie that the access-code gate cannot consume."""
    app = _app()
    store = SessionStore()
    add_login_routes(app, store, "letmein")
    app.add_middleware(AccessCodeMiddleware, store=store, enabled=True)
    client = TestClient(app, base_url="https://testserver", follow_redirects=False)

    assert client.post("/login", json={"code": "letmein"}).status_code == 200
    assert client.get("/api/options").status_code == 200
