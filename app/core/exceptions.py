from __future__ import annotations


class GatewayError(Exception):
    """网关基础异常"""
    def __init__(self, message: str, status_code: int = 500, detail: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}
        super().__init__(message)


class ModelNotFoundError(GatewayError):
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found",
            status_code=404,
            detail={"type": "model_not_found", "model": model}
        )


class NoAdapterAvailableError(GatewayError):
    def __init__(self, model: str):
        super().__init__(
            message=f"No enabled adapters for model '{model}'",
            status_code=503,
            detail={"type": "no_adapter", "model": model}
        )


class AllAdaptersFailedError(GatewayError):
    def __init__(self, model: str, attempts: list | None = None):
        super().__init__(
            message=f"All adapters failed for model '{model}'",
            status_code=502,
            detail={"type": "all_adapters_failed", "model": model, "attempts": attempts or []}
        )


class AuthenticationError(GatewayError):
    def __init__(self):
        super().__init__(
            message="Invalid API key",
            status_code=401,
            detail={"type": "auth_error"}
        )


class RateLimitError(GatewayError):
    def __init__(self, retry_after: float = 1.0):
        super().__init__(
            message="Rate limit exceeded",
            status_code=429,
            detail={"type": "rate_limit", "retry_after": retry_after}
        )
        self.retry_after = retry_after
