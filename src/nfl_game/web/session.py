"""Bounded, in-memory sessions for the NFL game web app."""

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
    def __init__(
        self,
        ttl: float = SESSION_TTL_SECONDS,
        max_sessions: int = MAX_SESSIONS,
        clock=time.monotonic,
    ):
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl
        self._max = max_sessions
        self._clock = clock

    def __len__(self) -> int:
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
            token for token, session in self._sessions.items() if session.last_seen < cutoff
        ]
        for token in expired:
            del self._sessions[token]

    def _evict_oldest(self) -> None:
        oldest = min(self._sessions, key=lambda token: self._sessions[token].last_seen)
        del self._sessions[oldest]
