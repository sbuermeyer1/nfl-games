import threading
import time
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


class SlowRecord:
    def __init__(self):
        self._failures = 0
        self.locked_until = 0.0
        self.last_seen = 0.0

    @property
    def failures(self):
        time.sleep(0.01)
        return self._failures

    @failures.setter
    def failures(self, value):
        time.sleep(0.01)
        self._failures = value


def test_concurrent_failures_for_one_ip_reach_the_lockout_threshold():
    """Catch racing increments that let one IP avoid its login lockout."""
    clock = Clock()
    throttle = LoginThrottle(max_failures=2, lockout=60, clock=clock)
    ip = "203.0.113.4"
    throttle._records[ip] = SlowRecord()
    barrier = threading.Barrier(2)

    def fail():
        barrier.wait(timeout=2)
        throttle.record_failure(ip)

    threads = [threading.Thread(target=fail) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert throttle.retry_after(ip) == 61


def test_concurrent_ip_churn_stays_bounded_and_preserves_active_lock():
    """Catch concurrent record-map churn that loses an active lock or the cap."""
    clock = Clock()
    throttle = LoginThrottle(max_failures=5, lockout=60, clock=clock, max_tracked=4)
    active_ip = "203.0.113.4"
    for _ in range(5):
        throttle.record_failure(active_ip)

    iteration_started = threading.Event()

    class YieldingRecords(dict):
        def items(self):
            for ip, record in super().items():
                if not iteration_started.is_set():
                    iteration_started.set()
                    time.sleep(0.05)
                yield ip, record

    throttle._records = YieldingRecords(throttle._records)
    errors: list[Exception] = []
    barrier = threading.Barrier(64)

    def churn(index):
        barrier.wait(timeout=2)
        try:
            throttle.record_failure(f"198.51.100.{index}")
        except RuntimeError as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=churn, args=(index,)) for index in range(64)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(throttle._records) <= 4
    assert throttle.retry_after(active_ip) == 61
