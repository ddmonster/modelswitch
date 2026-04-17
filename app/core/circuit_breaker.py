from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Dict

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"        # 正常，允许请求
    OPEN = "open"            # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 半开，允许少量探测


class CircuitBreaker:
    """每个 provider 实例一个熔断器

    Recovery timeout is capped at max 5 seconds for faster recovery.
    """

    # Maximum recovery timeout (seconds) - capped for faster recovery
    MAX_RECOVERY_TIMEOUT = 5

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 5,  # Default 5s, max 5s
        half_open_max: int = 1
    ):
        # Cap recovery timeout to max 5 seconds
        self.recovery_timeout = min(recovery_timeout, self.MAX_RECOVERY_TIMEOUT)
        self.failure_threshold = failure_threshold
        self.half_open_max = half_open_max
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_count = 0
        self._lock = threading.Lock()  # Thread safety for state transitions

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_count = 0
                    logger.info(f"熔断器进入半开状态，允许探测")
                    return True
                return False
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_count < self.half_open_max:
                    self.half_open_count += 1
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("熔断器半开探测成功，恢复为关闭状态")
                self.state = CircuitState.CLOSED
            self.failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN:
                logger.info("熔断器半开探测失败，重新熔断")
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"熔断器触发：连续失败 {self.failure_count} 次，"
                    f"熔断 {self.recovery_timeout} 秒"
                )
                self.state = CircuitState.OPEN

    def reset(self) -> None:
        """重置熔断器"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.half_open_count = 0

    def get_state_value(self) -> int:
        """返回 Prometheus 使用的数值：0=closed, 1=half_open, 2=open"""
        return {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}[self.state]
