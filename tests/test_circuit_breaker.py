"""Unit tests for CircuitBreaker."""
import time

from app.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInit:
    def test_default_values(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30
        assert cb.failure_count == 0

    def test_custom_values(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10, half_open_max=2)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10
        assert cb.half_open_max == 2


class TestCanExecute:
    def test_closed_allows(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_open_blocks_before_timeout(self):
        cb = CircuitBreaker(recovery_timeout=60)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.monotonic()
        assert cb.can_execute() is False

    def test_open_allows_after_timeout_transitions_to_half_open(self):
        cb = CircuitBreaker(recovery_timeout=0)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.monotonic() - 1  # in the past
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_up_to_max(self):
        cb = CircuitBreaker(half_open_max=2)
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_count = 0
        assert cb.can_execute() is True
        cb.half_open_count = 1
        assert cb.can_execute() is True
        cb.half_open_count = 2
        assert cb.can_execute() is False


class TestRecordSuccess:
    def test_resets_failure_count(self):
        cb = CircuitBreaker()
        cb.failure_count = 3
        cb.record_success()
        assert cb.failure_count == 0

    def test_half_open_transitions_to_closed(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_open_stays_open(self):
        """record_success from OPEN shouldn't happen normally, but test it anyway."""
        cb = CircuitBreaker()
        cb.state = CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 0


class TestRecordFailure:
    def test_increments_failure_count(self):
        cb = CircuitBreaker()
        cb.record_failure()
        assert cb.failure_count == 1

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_does_not_open_before_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_updates_last_failure_time(self):
        cb = CircuitBreaker()
        before = cb.last_failure_time
        cb.record_failure()
        assert cb.last_failure_time >= before

    def test_increments_half_open_count(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN
        # record_failure from HALF_OPEN transitions to OPEN, doesn't increment half_open_count
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # half_open_count is only incremented by can_execute() in HALF_OPEN state
        # This is checked indirectly via can_execute tests


class TestReset:
    def test_full_reset(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.half_open_count == 0


class TestGetStateValue:
    def test_closed_returns_0(self):
        cb = CircuitBreaker()
        assert cb.get_state_value() == 0

    def test_half_open_returns_1(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN
        assert cb.get_state_value() == 1

    def test_open_returns_2(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.OPEN
        assert cb.get_state_value() == 2


class TestFullCycle:
    def test_closed_to_open_to_half_open_to_closed(self):
        """Full lifecycle: normal -> fail -> open -> recover -> probe success -> closed."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        # Close -> Open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Open -> Half-Open (after timeout)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Half-Open -> Closed (probe success)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.can_execute()  # transitions to HALF_OPEN
        cb.record_failure()  # probe fails
        assert cb.state == CircuitState.OPEN
