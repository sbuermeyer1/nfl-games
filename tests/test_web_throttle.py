from types import SimpleNamespace

from nfl_game.web.throttle import LoginThrottle, client_ip


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_three_failures_lock_the_ip_for_configured_window():
    """Catch a throttle that permits guesses during the configured lockout."""
    clock = Clock()
    throttle = LoginThrottle(max_failures=3, lockout=60, clock=clock)

    for _ in range(3):
        throttle.record_failure("203.0.113.4")

    assert throttle.retry_after("203.0.113.4") == 61
    clock.now += 61
    assert throttle.retry_after("203.0.113.4") == 0


def test_success_clears_failures():
    """Catch a throttle that keeps a prior failed attempt after successful login."""
    throttle = LoginThrottle(max_failures=3)
    throttle.record_failure("203.0.113.4")

    throttle.record_success("203.0.113.4")

    assert throttle.retry_after("203.0.113.4") == 0


def test_client_ip_prefers_first_forwarded_address():
    """Catch proxy deployments that throttle the proxy rather than the client."""
    request = SimpleNamespace(
        headers={"x-forwarded-for": "203.0.113.4, 10.0.0.2"},
        client=SimpleNamespace(host="10.0.0.1"),
    )

    assert client_ip(request) == "203.0.113.4"
