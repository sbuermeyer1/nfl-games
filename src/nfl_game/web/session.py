"""Bounded, in-memory sessions for the NFL game web app."""

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

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
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._ttl = ttl
        self._max = max_sessions
        self._clock = clock
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._evict_expired()
            while len(self._sessions) >= self._max:
                self._evict_oldest()
            while token in self._sessions:
                token = secrets.token_urlsafe(32)
            self._sessions[token] = Session(last_seen=self._clock())
            return token

    def get(self, token: str | None) -> Session | None:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(token) if token else None
            if session is not None:
                session.last_seen = self._clock()
                self._sessions.move_to_end(token)
            return session

    def _evict_expired(self) -> None:
        """Remove expired sessions; caller must hold ``_lock``."""
        cutoff = self._clock() - self._ttl
        expired = [token for token, session in self._sessions.items() if session.last_seen < cutoff]
        for token in expired:
            del self._sessions[token]

    def _evict_oldest(self) -> None:
        """Remove the least-recently-used session; caller must hold ``_lock``."""
        self._sessions.popitem(last=False)
