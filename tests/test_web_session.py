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
