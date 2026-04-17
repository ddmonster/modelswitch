from __future__ import annotations

import time
from typing import Any


class GatewayError(Exception):
    """Base gateway exception with structured context for debugging."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: dict | None = None,
        request_id: str = "",
        adapter_name: str = "",
        model_name: str = "",
        latency_ms: float = 0.0,
    ):
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.adapter_name = adapter_name
        self.model_name = model_name
        self.latency_ms = latency_ms
        self.detail = detail or {}
        self.timestamp = time.monotonic()
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format message with context for logging."""
        parts = [self.message]
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.adapter_name:
            parts.append(f"adapter={self.adapter_name}")
        if self.model_name:
            parts.append(f"model={self.model_name}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON response."""
        return {
            "message": self.message,
            "type": self.detail.get("type", "gateway_error"),
            "request_id": self.request_id,
            "adapter": self.adapter_name,
            "model": self.model_name,
            **self.detail,
        }


class ModelNotFoundError(GatewayError):
    """Model not found in configuration."""

    def __init__(self, model: str, request_id: str = ""):
        super().__init__(
            message=f"Model '{model}' not found",
            status_code=404,
            request_id=request_id,
            model_name=model,
            detail={"type": "model_not_found", "model": model},
        )


class NoAdapterAvailableError(GatewayError):
    """No enabled adapters available for the model."""

    def __init__(self, model: str, request_id: str = ""):
        super().__init__(
            message=f"No enabled adapters for model '{model}'",
            status_code=503,
            request_id=request_id,
            model_name=model,
            detail={"type": "no_adapter", "model": model},
        )


class AllAdaptersFailedError(GatewayError):
    """All adapters in the chain failed."""

    def __init__(
        self,
        model: str,
        attempts: list[dict] | None = None,
        request_id: str = "",
    ):
        super().__init__(
            message=f"All adapters failed for model '{model}'",
            status_code=502,
            request_id=request_id,
            model_name=model,
            detail={
                "type": "all_adapters_failed",
                "model": model,
                "attempts": attempts or [],
                "failed_count": len(attempts or []),
            },
        )


class AdapterRequestError(GatewayError):
    """Error during adapter request to upstream provider."""

    def __init__(
        self,
        message: str,
        adapter_name: str,
        model_name: str,
        status_code: int,
        request_id: str = "",
        latency_ms: float = 0.0,
        upstream_error: str = "",
        request_params: dict | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            request_id=request_id,
            adapter_name=adapter_name,
            model_name=model_name,
            latency_ms=latency_ms,
            detail={
                "type": "adapter_request_error",
                "upstream_error": upstream_error,
                "request_params": request_params or {},
            },
        )


class ResponseParseError(GatewayError):
    """Error parsing response from upstream provider."""

    def __init__(
        self,
        message: str,
        adapter_name: str,
        model_name: str,
        request_id: str = "",
        raw_response: str = "",
        parse_stage: str = "",
    ):
        super().__init__(
            message=f"Failed to parse response: {message}",
            status_code=502,
            request_id=request_id,
            adapter_name=adapter_name,
            model_name=model_name,
            detail={
                "type": "response_parse_error",
                "parse_stage": parse_stage,
                "raw_response_preview": raw_response[:500] if raw_response else "",
            },
        )


class StreamError(GatewayError):
    """Error during stream processing."""

    def __init__(
        self,
        message: str,
        adapter_name: str,
        model_name: str,
        request_id: str = "",
        chunk_index: int = -1,
        chunk_preview: str = "",
    ):
        super().__init__(
            message=f"Stream error: {message}",
            status_code=502,
            request_id=request_id,
            adapter_name=adapter_name,
            model_name=model_name,
            detail={
                "type": "stream_error",
                "chunk_index": chunk_index,
                "chunk_preview": chunk_preview[:200] if chunk_preview else "",
            },
        )


class AuthenticationError(GatewayError):
    """Invalid or missing API key."""

    def __init__(self, request_id: str = "", key_preview: str = ""):
        super().__init__(
            message="Invalid or missing API key",
            status_code=401,
            request_id=request_id,
            detail={"type": "auth_error", "key_preview": key_preview},
        )


class RateLimitError(GatewayError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: float = 1.0, request_id: str = "", api_key_name: str = ""):
        super().__init__(
            message="Rate limit exceeded",
            status_code=429,
            request_id=request_id,
            detail={
                "type": "rate_limit",
                "retry_after": retry_after,
                "api_key_name": api_key_name,
            },
        )
        self.retry_after = retry_after


class CircuitBreakerOpenError(GatewayError):
    """Circuit breaker is open for the provider."""

    def __init__(
        self,
        adapter_name: str,
        model_name: str,
        request_id: str = "",
        failure_count: int = 0,
        recovery_timeout: int = 30,
    ):
        super().__init__(
            message=f"Circuit breaker open for '{adapter_name}'",
            status_code=503,
            request_id=request_id,
            adapter_name=adapter_name,
            model_name=model_name,
            detail={
                "type": "circuit_breaker_open",
                "failure_count": failure_count,
                "recovery_timeout": recovery_timeout,
            },
        )
