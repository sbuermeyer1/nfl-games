import secrets
import threading
import time
from collections import OrderedDict

import nfl_game.web.session as session_module
from nfl_game.web.session import SessionStore


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_created_token_resolves_to_a_session():
    """Catch a store that creates a token without retaining its session."""
    store = SessionStore()

    token = store.create()

    assert len(token) >= 32
    assert store.get(token) is not None


def test_expired_token_is_removed():
    """Catch a store that accepts a session after its configured lifetime."""
    clock = Clock()
    store = SessionStore(ttl=10, clock=clock)
    token = store.create()

    clock.now += 11

    assert store.get(token) is None


def test_store_evicts_oldest_at_capacity():
    """Catch unbounded session growth or eviction of the newest session."""
    clock = Clock()
    store = SessionStore(max_sessions=2, clock=clock)
    oldest = store.create()
    clock.now += 1
    store.create()
    clock.now += 1
    store.create()

    assert store.get(oldest) is None
    assert len(store) == 2


def test_token_at_exact_ttl_boundary_is_still_valid():
    """Catch a store that expires a session at, rather than after, its TTL."""
    clock = Clock()
    store = SessionStore(ttl=10, clock=clock)
    token = store.create()

    clock.now += 10

    assert store.get(token) is not None


def test_same_tick_read_makes_session_most_recent_for_eviction():
    """Catch eviction that ignores a read when timestamps tie."""
    clock = Clock()
    store = SessionStore(max_sessions=2, clock=clock)
    first = store.create()
    second = store.create()

    assert store.get(first) is not None
    store.create()

    assert store.get(first) is not None
    assert store.get(second) is None


def test_concurrent_creates_never_exceed_capacity(monkeypatch):
    """Catch competing creates both claiming the same final session slot."""
    barrier = threading.Barrier(2)
    original = secrets.token_urlsafe

    def synchronized_token(size):
        barrier.wait(timeout=2)
        return original(size)

    monkeypatch.setattr(session_module.secrets, "token_urlsafe", synchronized_token)
    store = SessionStore(max_sessions=1)
    tokens: list[str] = []

    threads = [threading.Thread(target=lambda: tokens.append(store.create())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert len(store) == 1
    assert sum(store.get(token) is not None for token in tokens) == 1


def test_concurrent_get_and_create_do_not_mutate_during_expiry_iteration():
    """Catch a concurrent create raising while a reader cleans expired sessions."""
    iteration_started = threading.Event()

    class YieldingSessions(OrderedDict):
        def items(self):
            for token, session in super().items():
                if not iteration_started.is_set():
                    iteration_started.set()
                    time.sleep(0.05)
                yield token, session

    store = SessionStore(max_sessions=3)
    token = store.create()
    store._sessions = YieldingSessions(store._sessions)
    errors: list[Exception] = []

    def read():
        try:
            store.get(token)
        except RuntimeError as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def create():
        assert iteration_started.wait(timeout=2)
        try:
            store.create()
        except RuntimeError as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    reader = threading.Thread(target=read)
    writer = threading.Thread(target=create)
    reader.start()
    writer.start()
    reader.join(timeout=2)
    writer.join(timeout=2)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert errors == []
    assert len(store) <= 3
