from __future__ import annotations

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
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
            "module": record.module,
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


def add_log_to_buffer(request_id: str, level: str, message: str, **extra) -> None:
    """添加日志到内存缓冲"""
    _log_buffer.append(
        {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "level": level,
            "message": message,
            **extra,
        }
    )


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

        logger.debug(
            f"response status={response.status_code} latency={latency:.0f}ms",
            extra={"request_id": request_id},
        )

        return response
