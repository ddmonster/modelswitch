from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.exceptions import AuthenticationError, GatewayError, RateLimitError


class RateLimiter:
    """按 API Key 的双维度令牌桶限流器"""

    def __init__(self):
        # 每分钟限流：{api_key: [token_count, last_reset_time]}
        self._minute_buckets: Dict[str, list] = defaultdict(
            lambda: [0, time.monotonic()]
        )
        # 每日限流：{api_key: [token_count, date_str]}
        self._daily_buckets: Dict[str, list] = defaultdict(lambda: [0, self._today()])

    @staticmethod
    def _today() -> str:
        from datetime import date

        return date.today().isoformat()

    def check(self, api_key: str, rate_limit: int, daily_limit: int) -> Optional[float]:
        """
        检查是否允许请求。
        返回 None 表示允许，返回 float 表示需要等待的秒数（被限流）。
        """
        now = time.monotonic()
        today = self._today()

        # 检查每分钟限流
        if rate_limit > 0:
            bucket = self._minute_buckets[api_key]
            if now - bucket[1] >= 60:
                bucket[0] = 0
                bucket[1] = now
            if bucket[0] >= rate_limit:
                retry_after = 60 - (now - bucket[1])
                return max(retry_after, 0.5)
            bucket[0] += 1

        # 检查每日限流
        if daily_limit > 0:
            bucket = self._daily_buckets[api_key]
            if bucket[1] != today:
                bucket[0] = 0
                bucket[1] = today
            if bucket[0] >= daily_limit:
                return 3600.0  # 等待 1 小时
            bucket[0] += 1

        return None

    def reset_key(self, api_key: str) -> None:
        """重置某个 key 的限流计数"""
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

        from starlette.requests import Request
        from starlette.responses import Response as StarletteResponse

        request = Request(scope, receive)
        path = scope.get("path", "")

        # CORS 预检
        if request.method == "OPTIONS":
            response = self._cors_response()
            await response(scope, receive, send)
            return

        # 跳过不需要认证的路径
        if self._is_public_path(path):
            await self.app(scope, receive, send)
            return

        # /api/* 路径需要鉴权
        if path.startswith("/api/"):
            # 认证
            api_key, key_config = self._authenticate(request)
            if not api_key:
                response = JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Authentication required",
                            "type": "auth_error",
                        }
                    },
                )
                await response(scope, receive, send)
                return

            if not key_config.get("enabled", True):
                response = JSONResponse(
                    status_code=403,
                    content={
                        "error": {"message": "API key is disabled", "type": "forbidden"}
                    },
                )
                await response(scope, receive, send)
                return

            # 检查过期
            expires_at = key_config.get("expires_at")
            if expires_at:
                from datetime import datetime

                try:
                    if datetime.fromisoformat(expires_at) < datetime.now():
                        response = JSONResponse(
                            status_code=403,
                            content={
                                "error": {
                                    "message": "API key has expired",
                                    "type": "forbidden",
                                }
                            },
                        )
                        await response(scope, receive, send)
                        return
                except ValueError:
                    pass

            # 检查 admin 权限
            if self._is_admin_required(path, request.method):
                roles = key_config.get("roles", ["user"])
                if "admin" not in roles:
                    response = JSONResponse(
                        status_code=403,
                        content={
                            "error": {
                                "message": "Admin role required",
                                "type": "forbidden",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    return

            # 注入 key 信息到 scope["state"]
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["api_key"] = api_key
            key_name = key_config.get("name", "")
            if not key_name:
                key_name = (
                    api_key[:3] + "****" + api_key[-4:]
                    if len(api_key) > 7
                    else api_key[:3] + "****"
                )
            scope["state"]["api_key_name"] = key_name
            scope["state"]["api_key_config"] = key_config

            # 注入角色信息
            scope["state"]["api_key_roles"] = key_config.get("roles", ["user"])

            await self.app(scope, receive, send)
            return

        # /v1/*, /openai/*, /anthropic/* — 原有鉴权逻辑保持不变
        # 认证
        api_key, key_config = self._authenticate(request)
        if not api_key:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key",
                        "type": "auth_error",
                    }
                },
            )
            await response(scope, receive, send)
            return

        if not key_config.get("enabled", True):
            response = JSONResponse(
                status_code=403,
                content={
                    "error": {"message": "API key is disabled", "type": "forbidden"}
                },
            )
            await response(scope, receive, send)
            return

        # 检查过期
        expires_at = key_config.get("expires_at")
        if expires_at:
            from datetime import datetime

            try:
                if datetime.fromisoformat(expires_at) < datetime.now():
                    response = JSONResponse(
                        status_code=403,
                        content={
                            "error": {
                                "message": "API key has expired",
                                "type": "forbidden",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        # 限流
        rate_limit = key_config.get("rate_limit", 60)
        daily_limit = key_config.get("daily_limit", 0)
        retry_after = self._rate_limiter.check(api_key, rate_limit, daily_limit)
        if retry_after is not None:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {"message": "Rate limit exceeded", "type": "rate_limit"}
                },
                headers={"Retry-After": str(int(retry_after))},
            )
            await response(scope, receive, send)
            return

        # 注入 key 信息到 scope["state"]
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["api_key"] = api_key
        key_name = key_config.get("name", "")
        if not key_name:
            # fallback: mask key as sk-****xxxx
            key_name = (
                api_key[:3] + "****" + api_key[-4:]
                if len(api_key) > 7
                else api_key[:3] + "****"
            )
        scope["state"]["api_key_name"] = key_name
        scope["state"]["api_key_config"] = key_config

        # 活跃请求计数
        if self._active_requests is not None:
            self._active_requests["count"] += 1

        try:
            await self.app(scope, receive, send)
        finally:
            if self._active_requests is not None:
                self._active_requests["count"] -= 1

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
        """判断路径是否需要认证（完全不鉴权）"""
        public_paths = {
            "/",
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        }
        if path in public_paths:
            return True
        if path.startswith("/web/") or path.startswith("/static/"):
            return True
        return False

    def _is_admin_required(self, path: str, method: str) -> bool:
        """判断路径是否需要 admin 角色

        所有 /api/config/* 的 POST/PUT/DELETE/PATCH 需要 admin
        所有 /api/keys/* 的 POST/PUT/DELETE/PATCH 需要 admin
        GET /api/config/* 和 GET /api/keys/* 需要 admin（含完整配置）
        GET /api/usage 和 GET /api/logs 只需要任意有效 key
        GET /api/conversations 只需要任意有效 key
        """
        # /api/config/* — 全部需要 admin
        if path.startswith("/api/config"):
            return True
        # /api/keys/* — 全部需要 admin
        if path.startswith("/api/keys"):
            return True
        return False

    def _add_cors_headers(self, response: Response) -> None:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, x-api-key, X-Request-ID"
        )

    def _cors_response(self) -> Response:
        response = JSONResponse(content={})
        self._add_cors_headers(response)
        return response
