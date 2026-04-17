from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.exceptions import AuthenticationError, GatewayError, RateLimitError


class RateLimiter:
    """按 API Key 的双维度令牌桶限流器"""

    def __init__(self):
        self._lock = threading.Lock()  # Thread safety for bucket operations
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 3600  # Cleanup every hour
        # 每分钟限流：{api_key: [token_count, last_reset_time]}
        self._minute_buckets: Dict[str, list] = {}
        # 每日限流：{api_key: [token_count, date_str]}
        self._daily_buckets: Dict[str, list] = {}

    @staticmethod
    def _today() -> str:
        from datetime import date
        return date.today().isoformat()

    def _maybe_cleanup(self) -> None:
        """Remove stale entries to prevent unbounded memory growth."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # Remove minute buckets older than 2 hours
        stale_minute = [
            k for k, v in self._minute_buckets.items()
            if now - v[1] > 7200
        ]
        for k in stale_minute:
            del self._minute_buckets[k]

        # Remove daily buckets from previous days
        today = self._today()
        stale_daily = [
            k for k, v in self._daily_buckets.items()
            if v[1] != today
        ]
        for k in stale_daily:
            del self._daily_buckets[k]

        self._last_cleanup = now

    def check(self, api_key: str, rate_limit: int, daily_limit: int) -> Optional[float]:
        """
        检查是否允许请求。
        返回 None 表示允许，返回 float 表示需要等待的秒数（被限流）。
        """
        with self._lock:
            self._maybe_cleanup()
            now = time.monotonic()
            today = self._today()

            # 检查每分钟限流
            if rate_limit > 0:
                bucket = self._minute_buckets.get(api_key)
                if bucket is None:
                    bucket = [0, now]
                    self._minute_buckets[api_key] = bucket
                if now - bucket[1] >= 60:
                    bucket[0] = 0
                    bucket[1] = now
                if bucket[0] >= rate_limit:
                    retry_after = 60 - (now - bucket[1])
                    return max(retry_after, 0.5)
                bucket[0] += 1

            # 检查每日限流
            if daily_limit > 0:
                bucket = self._daily_buckets.get(api_key)
                if bucket is None:
                    bucket = [0, today]
                    self._daily_buckets[api_key] = bucket
                if bucket[1] != today:
                    bucket[0] = 0
                    bucket[1] = today
                if bucket[0] >= daily_limit:
                    return 3600.0  # 等待 1 小时
                bucket[0] += 1

            return None

    def reset_key(self, api_key: str) -> None:
        """重置某个 key 的限流计数"""
        with self._lock:
            self._minute_buckets.pop(api_key, None)
            self._daily_buckets.pop(api_key, None)


class GatewayMiddleware:
    """网关中间件：认证 + 限流 + CORS + 活跃请求计数（纯 ASGI 中间件）

    config 和 active_requests_counter 通过共享引用传入，
    热重载时修改 config 对象即可自动生效。
    """

    def __init__(self, app: ASGIApp, config=None, active_requests_counter=None):
        self.app = app
        self._config = config
        self._rate_limiter = RateLimiter()
        self._active_requests = active_requests_counter
        self._counter_lock = threading.Lock()  # Thread safety for active request counter
        self._api_keys: Dict[str, dict] = {}
        if config:
            self._load_api_keys(config)

    def _load_api_keys(self, config) -> None:
        """从配置加载 API Keys"""
        self._api_keys = {}
        for key_config in config.api_keys:
            self._api_keys[key_config.key] = key_config.model_dump()

    def reload_config(self, config) -> None:
        """热重载配置"""
        self._config = config
        self._load_api_keys(config)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = scope.get("path", "")

        # CORS preflight
        if request.method == "OPTIONS":
            response = self._cors_response()
            await response(scope, receive, send)
            return

        # Public path bypass
        if self._is_public_path(path):
            await self.app(scope, receive, send)
            return

        # Authentication
        api_key, key_config = self._authenticate(request)
        if not api_key:
            await self._create_error_response(401, "Invalid or missing API key", "auth_error")(scope, receive, send)
            return

        # Key validity check
        error_response = self._check_key_validity(key_config)
        if error_response:
            await error_response(scope, receive, send)
            return

        # Rate limiting
        error_response = self._check_rate_limit(api_key, key_config)
        if error_response:
            await error_response(scope, receive, send)
            return

        # Inject state
        self._inject_scope_state(scope, api_key, key_config)

        # Active request counting
        if self._active_requests is not None:
            with self._counter_lock:
                self._active_requests["count"] += 1

        try:
            await self.app(scope, receive, send)
        finally:
            if self._active_requests is not None:
                with self._counter_lock:
                    self._active_requests["count"] -= 1

    def _create_error_response(self, status_code: int, message: str, error_type: str, headers: dict = None) -> JSONResponse:
        """Create error JSONResponse."""
        return JSONResponse(
            status_code=status_code,
            content={"error": {"message": message, "type": error_type}},
            headers=headers or {},
        )

    def _check_key_validity(self, key_config: dict) -> JSONResponse | None:
        """Check if key is enabled and not expired. Returns error response if invalid."""
        if not key_config.get("enabled", True):
            return JSONResponse(
                status_code=403,
                content={"error": {"message": "API key is disabled", "type": "forbidden"}},
            )

        expires_at = key_config.get("expires_at")
        if expires_at:
            from datetime import datetime
            try:
                if datetime.fromisoformat(expires_at) < datetime.now():
                    return JSONResponse(
                        status_code=403,
                        content={"error": {"message": "API key has expired", "type": "forbidden"}},
                    )
            except ValueError:
                pass

        return None

    def _check_rate_limit(self, api_key: str, key_config: dict) -> JSONResponse | None:
        """Check rate limit. Returns error response if exceeded."""
        rate_limit = key_config.get("rate_limit", 60)
        daily_limit = key_config.get("daily_limit", 0)
        retry_after = self._rate_limiter.check(api_key, rate_limit, daily_limit)
        if retry_after is not None:
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "Rate limit exceeded", "type": "rate_limit"}},
                headers={"Retry-After": str(int(retry_after))},
            )
        return None

    def _inject_scope_state(self, scope: dict, api_key: str, key_config: dict) -> None:
        """Inject API key info into scope state."""
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["api_key"] = api_key
        key_name = key_config.get("name", "")
        if not key_name:
            # fallback: mask key as sk-****xxxx
            key_name = api_key[:3] + "****" + api_key[-4:] if len(api_key) > 7 else api_key[:3] + "****"
        scope["state"]["api_key_name"] = key_name
        scope["state"]["api_key_config"] = key_config

    def _authenticate(self, request: Request):
        """从请求中提取并验证 API Key"""
        auth = request.headers.get("Authorization", "")
        api_key = ""

        if auth.startswith("Bearer "):
            api_key = auth[7:]
        elif auth.startswith("sk-"):
            api_key = auth
        else:
            # Anthropic 风格：x-api-key
            api_key = request.headers.get("x-api-key", "")

        if not api_key:
            return None, None

        key_config = self._api_keys.get(api_key)
        if not key_config:
            return None, None

        return api_key, key_config

    def _is_public_path(self, path: str) -> bool:
        """判断路径是否需要认证"""
        public_paths = {
            "/", "/health", "/metrics",
            "/docs", "/openapi.json", "/redoc",
        }
        if path in public_paths:
            return True
        if path.startswith("/web/") or path.startswith("/static/"):
            return True
        if path.startswith("/api/logs") or path.startswith("/api/usage"):
            return True
        if path.startswith("/api/config") or path.startswith("/api/providers"):
            return True
        if path.startswith("/api/keys"):
            return True
        if path.startswith("/api/conversations"):
            return True
        return False

    def _add_cors_headers(self, response: Response) -> None:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, x-api-key, X-Request-ID"

    def _cors_response(self) -> Response:
        response = JSONResponse(content={})
        self._add_cors_headers(response)
        return response
