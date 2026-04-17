from __future__ import annotations

import inspect
import json
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("modelswitch")

# 内存环形缓冲：存储最近日志供 API 查询
_log_buffer: deque = deque(maxlen=1000)


class JSONFormatter(logging.Formatter):
    """JSON 结构化日志格式"""

    def format(self, record: logging.LogRecord) -> str:
        # Get relative path from project root
        pathname = record.pathname
        # Try to make path relative to project
        if "modelswitch" in pathname or "app" in pathname:
            # Find app/ or modelswitch/ in path and use relative from there
            for prefix in ["app/", "modelswitch/"]:
                idx = pathname.find(prefix)
                if idx >= 0:
                    pathname = pathname[idx:]
                    break

        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "location": f"{pathname}:{record.lineno}",
            "request_id": getattr(record, "request_id", ""),
        }
        return json.dumps(log_entry, ensure_ascii=False)


class TrackingRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that tracks byte offsets of written records.

    After emit(), self.last_byte_offset contains the file position
    where the last record started, and self.current_baseFilename
    contains the current log file path.
    """

    def __init__(self, filename, *args, **kwargs):
        super().__init__(filename, *args, **kwargs)
        self.last_byte_offset = 0
        self._offset_lock = threading.Lock()

    def emit(self, record):
        with self._offset_lock:
            if self.stream and not self.stream.closed:
                try:
                    self.last_byte_offset = self.stream.tell()
                except (OSError, ValueError):
                    self.last_byte_offset = 0
        super().emit(record)

    @property
    def current_base_filename(self) -> str:
        """Return the current log file path."""
        return self.baseFilename


# Module-level reference to the conversation log handler for byte offset tracking
_conv_handler: Optional[TrackingRotatingFileHandler] = None


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 104857600,
    backup_count: int = 30,
) -> None:
    """配置日志系统"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除已有 handler
    root_logger.handlers.clear()

    # stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(stdout_handler)

    # 文件 handler（按天轮转）
    file_handler = RotatingFileHandler(
        f"{log_dir}/gateway.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # 会话日志 logger（专用，不传播到根 logger）
    conv_logger = logging.getLogger("modelswitch.conversations")
    conv_logger.setLevel(logging.INFO)
    conv_logger.propagate = False
    conv_logger.handlers.clear()
    conv_handler = TrackingRotatingFileHandler(
        f"{log_dir}/conversations.jsonl",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    conv_handler.setFormatter(logging.Formatter("%(message)s"))
    conv_logger.addHandler(conv_handler)

    # Export for other modules to access tracking info
    global _conv_handler
    _conv_handler = conv_handler


def get_conv_handler() -> Optional[TrackingRotatingFileHandler]:
    """Get the conversation log file handler for byte offset tracking."""
    return _conv_handler


def add_log_to_buffer(
    request_id: str,
    level: str,
    message: str,
    location: str = "",
    **extra
) -> None:
    """添加日志到内存缓冲"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "level": level,
        "message": message,
    }
    if location:
        entry["location"] = location
    entry.update(extra)
    _log_buffer.append(entry)


def _get_caller_location(skip_frames: int = 2) -> str:
    """Get caller's file path and line number.

    Args:
        skip_frames: Number of frames to skip (default 2 for this function + caller)

    Returns:
        Relative path with line number, e.g., "app/api/routes.py:42"
    """
    frame = inspect.stack()[skip_frames]
    filename = frame.filename
    lineno = frame.lineno

    # Make path relative to project
    for prefix in ["app/", "modelswitch/"]:
        idx = filename.find(prefix)
        if idx >= 0:
            filename = filename[idx:]
            break

    return f"{filename}:{lineno}"


def get_log_buffer() -> list:
    """获取缓冲区日志"""
    return list(_log_buffer)


def clear_log_buffer() -> None:
    """清空日志缓冲区（用于测试）"""
    _log_buffer.clear()


def get_logs_filtered(
    tail: int = 100,
    level: Optional[str] = None,
    request_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list:
    """带过滤条件的日志查询"""
    logs = list(_log_buffer)
    if level:
        logs = [l for l in logs if l.get("level") == level.upper()]
    if request_id:
        logs = [l for l in logs if l.get("request_id") == request_id]
    if api_key:
        logs = [l for l in logs if api_key in l.get("message", "")]
    return logs[-tail:]


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件：注入 request_id，记录请求/响应日志"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成 request_id
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id

        start = time.monotonic()

        # 记录请求日志
        logger.debug(
            f"request method={request.method} path={request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}",
            extra={"request_id": request_id},
        )

        try:
            response = await call_next(request)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(
                f"unhandled_error path={request.url.path} "
                f"error={e} latency={latency:.0f}ms",
                extra={"request_id": request_id},
            )
            return JSONResponse(
                status_code=500, content={"error": {"message": "Internal server error"}}
            )

        latency = (time.monotonic() - start) * 1000

        # 添加响应头
        response.headers["X-Request-ID"] = request_id


# ─────────────────────────────────────────────────────────────────────────────
# Adapter Debug Logging Helpers
# ─────────────────────────────────────────────────────────────────────────────


class AdapterLogger:
    """Structured logger for adapter requests/responses with request context."""

    def __init__(self, adapter_name: str, request_id: str = ""):
        self.adapter_name = adapter_name
        self.request_id = request_id
        self._logger = logging.getLogger(f"modelswitch.adapter.{adapter_name}")

    def _log(self, level: int, message: str, **context) -> None:
        """Log with structured context."""
        extra = {"request_id": self.request_id, "adapter": self.adapter_name}
        full_msg = f"[{self.request_id}] {message}"
        for k, v in context.items():
            if v is not None:
                full_msg += f" {k}={v}"
        self._logger.log(level, full_msg, extra=extra)
        # Only add to buffer for info/warning/error levels (skip debug)
        # Debug logs are too verbose for in-memory buffer and journalctl
        if level >= logging.INFO:
            location = _get_caller_location(skip_frames=3)  # Skip _log -> debug/info/warning/error -> actual caller
            add_log_to_buffer(
                self.request_id,
                logging.getLevelName(level),
                full_msg,
                location=location,
                api_key=context.get("api_key", ""),
            )

    def debug(self, message: str, **context) -> None:
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context) -> None:
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context) -> None:
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context) -> None:
        self._log(logging.ERROR, message, **context)

    def log_request(
        self,
        model: str,
        stream: bool,
        timeout: int,
        messages_count: int,
        params: dict | None = None,
    ) -> None:
        """Log outgoing request to upstream."""
        self.debug(
            f"adapter_request_start",
            model=model,
            stream=stream,
            timeout=timeout,
            messages_count=messages_count,
            params_count=len(params or {}),
        )
        # Log params at debug level (may contain sensitive data)
        if params:
            safe_params = self._sanitize_params(params)
            self.debug(f"request_params {json.dumps(safe_params)}")

    def log_response_start(
        self,
        model: str,
        latency_ms: float,
        status_code: int,
        usage: dict | None = None,
    ) -> None:
        """Log successful response from upstream."""
        self.info(
            f"adapter_response_ok",
            model=model,
            latency_ms=f"{latency_ms:.0f}",
            status=status_code,
            prompt_tokens=usage.get("prompt_tokens", 0) if usage else 0,
            completion_tokens=usage.get("completion_tokens", 0) if usage else 0,
        )

    def log_stream_chunk(
        self,
        chunk_index: int,
        content_preview: str = "",
        tool_calls_count: int = 0,
    ) -> None:
        """Log stream chunk processing."""
        preview = content_preview[:50] if content_preview else ""
        self.debug(
            f"stream_chunk",
            chunk_index=chunk_index,
            content_preview=preview,
            tool_calls=tool_calls_count,
        )

    def log_stream_complete(
        self,
        model: str,
        total_chunks: int,
        latency_ms: float,
        usage: dict | None = None,
    ) -> None:
        """Log stream completion."""
        self.info(
            f"adapter_stream_complete",
            model=model,
            chunks=total_chunks,
            latency_ms=f"{latency_ms:.0f}",
            prompt_tokens=usage.get("prompt_tokens", 0) if usage else 0,
            completion_tokens=usage.get("completion_tokens", 0) if usage else 0,
        )

    def log_error(
        self,
        model: str,
        error_type: str,
        error_message: str,
        latency_ms: float,
        status_code: int,
        upstream_response: str | None = None,
    ) -> None:
        """Log adapter error with details."""
        # Include upstream response preview in error log for debugging
        error_preview = error_message[:500]
        if upstream_response:
            error_preview += f" | upstream: {upstream_response[:300]}"

        self.error(
            f"adapter_error",
            model=model,
            error_type=error_type,
            status=status_code,
            latency_ms=f"{latency_ms:.0f}",
            error=error_preview,
        )

    def log_parse_error(
        self,
        model: str,
        parse_stage: str,
        error_message: str,
        raw_data_preview: str = "",
    ) -> None:
        """Log parsing error during response processing."""
        self.error(
            f"parse_error",
            model=model,
            stage=parse_stage,
            error=error_message[:200],
            raw_preview=raw_data_preview[:100],
        )

    def _sanitize_params(self, params: dict) -> dict:
        """Remove sensitive fields from params for logging."""
        sensitive_keys = {"api_key", "key", "token", "authorization", "password"}
        result = {}
        for k, v in params.items():
            if k.lower() in sensitive_keys:
                result[k] = "[REDACTED]"
            elif isinstance(v, dict):
                result[k] = self._sanitize_params(v)
            elif isinstance(v, list):
                result[k] = f"[list:{len(v)}]"
            else:
                # Truncate large values
                if isinstance(v, str) and len(v) > 100:
                    result[k] = v[:100] + "..."
                else:
                    result[k] = v
        return result


def get_adapter_logger(adapter_name: str, request_id: str = "") -> AdapterLogger:
    """Get an adapter logger instance."""
    return AdapterLogger(adapter_name, request_id)
