"""Pure-ASGI access-code middleware for the NFL game web app."""

from nfl_game.web.session import COOKIE_NAME, SessionStore

LOGIN_PATH = "/login"
PUBLIC_PATHS = {LOGIN_PATH, "/health"}


def read_cookie(scope, name: str) -> str | None:
    """Read one unescaped cookie value from all ASGI Cookie header fields."""
    raw = b"; ".join(
        value for header, value in scope.get("headers") or [] if header == b"cookie"
    ).decode("latin-1")
    for part in raw.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key == name:
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
