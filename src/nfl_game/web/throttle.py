"""Per-IP failed-login throttle for the shared access code."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

MAX_FAILURES = 5
LOCKOUT_SECONDS = 60
MAX_TRACKED_IPS = 5000


@dataclass
class _Record:
    failures: int
    locked_until: float
    last_seen: float


def client_ip(request) -> str:
    """Return the client address, honoring the first proxy-forwarded address."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class LoginThrottle:
    def __init__(
        self,
        max_failures: int = MAX_FAILURES,
        lockout: float = LOCKOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_tracked: int = MAX_TRACKED_IPS,
    ):
        self._records: dict[str, _Record] = {}
        self._max = max_failures
        self._lockout = lockout
        self._clock = clock
        self._max_tracked = max_tracked
        self._lock = RLock()

    def retry_after(self, ip: str) -> int:
        """Return seconds until ``ip`` may try again, or zero when unlocked."""
        with self._lock:
            return self._retry_after(ip)

    def check_and_record(self, ip: str, successful: bool) -> int:
        """Atomically check lockout and record an allowed login outcome.

        Returns a nonzero retry delay when an existing lockout rejects the
        attempt. A threshold-reaching failure is recorded but still returns
        zero, so its caller returns the normal incorrect-code response.
        """
        with self._lock:
            wait = self._retry_after(ip)
            if wait:
                return wait
            if successful:
                self._records.pop(ip, None)
            else:
                self._record_failure(ip)
            return 0

    def record_failure(self, ip: str) -> None:
        """Record a failure; retained for callers that do not need a decision."""
        with self._lock:
            self._record_failure(ip)

    def record_success(self, ip: str) -> None:
        """Clear prior failures; retained for callers that do not need a decision."""
        with self._lock:
            self._records.pop(ip, None)

    def _retry_after(self, ip: str) -> int:
        """Return the current retry delay; caller must hold ``_lock``."""
        self._clear_expired(ip)
        record = self._records.get(ip)
        if record is None or record.locked_until == 0.0:
            return 0
        return int(record.locked_until - self._clock()) + 1

    def _record_failure(self, ip: str) -> None:
        """Record one failure and enforce bounds; caller must hold ``_lock``."""
        self._clear_expired(ip)
        record = self._records.setdefault(
            ip, _Record(failures=0, locked_until=0.0, last_seen=self._clock())
        )
        record.failures += 1
        record.last_seen = self._clock()
        if record.failures >= self._max:
            record.locked_until = self._clock() + self._lockout
        if len(self._records) > self._max_tracked:
            self._evict_stale_lockouts()
            if len(self._records) > self._max_tracked:
                self._evict_oldest()

    def _clear_expired(self, ip: str) -> None:
        """Clear an elapsed lockout; caller must hold ``_lock``."""
        record = self._records.get(ip)
        if (
            record is not None
            and record.locked_until > 0.0
            and self._clock() >= record.locked_until
        ):
            del self._records[ip]

    def _evict_stale_lockouts(self) -> None:
        """Drop elapsed lockouts; caller must hold ``_lock``."""
        now = self._clock()
        stale = [
            ip
            for ip, record in self._records.items()
            if record.locked_until and now >= record.locked_until
        ]
        for ip in stale:
            del self._records[ip]

    def _evict_oldest(self) -> None:
        """Bound tracking while preserving active locks; caller must hold ``_lock``."""
        now = self._clock()
        unlocked = {
            ip: record
            for ip, record in self._records.items()
            if not (record.locked_until and record.locked_until > now)
        }
        pool = unlocked if unlocked else self._records
        oldest = min(pool, key=lambda ip: pool[ip].last_seen)
        del self._records[oldest]
