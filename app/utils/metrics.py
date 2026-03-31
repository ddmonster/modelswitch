from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# 请求总数
REQUESTS_TOTAL = Counter(
    "modelswitch_requests_total",
    "Total requests",
    ["model", "provider", "api_key_alias", "status"],
)

# 请求延迟
REQUEST_LATENCY = Histogram(
    "modelswitch_request_latency_seconds",
    "Request latency in seconds",
    ["model", "provider"],
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
)

# 活跃请求
ACTIVE_REQUESTS = Gauge(
    "modelswitch_active_requests",
    "Currently active in-flight requests",
)

# 熔断器状态 (0=closed, 1=half_open, 2=open)
CIRCUIT_BREAKER_STATE = Gauge(
    "modelswitch_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["provider"],
)

# Provider 失败数
PROVIDER_FAILURES = Counter(
    "modelswitch_provider_failures_total",
    "Provider failure count",
    ["provider", "status_code"],
)


def record_request(model: str, provider: str, api_key_alias: str, status: int, latency_ms: float) -> None:
    """记录一次请求的指标"""
    REQUESTS_TOTAL.labels(
        model=model, provider=provider or "none",
        api_key_alias=api_key_alias or "unknown", status=str(status)
    ).inc()
    if provider:
        REQUEST_LATENCY.labels(model=model, provider=provider).observe(latency_ms / 1000)


def record_provider_failure(provider: str, status_code: int) -> None:
    PROVIDER_FAILURES.labels(provider=provider, status_code=str(status_code)).inc()
