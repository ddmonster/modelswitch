from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from app.models.config_models import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class AdapterResponse:
    """适配器统一响应封装"""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None                        # 非流式：完整响应对象
    stream: Optional[AsyncGenerator] = None  # 流式：AsyncGenerator
    latency_ms: float = 0.0
    adapter_name: str = ""
    model_name: str = ""
    success: bool = True
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None   # {"prompt_tokens": x, "completion_tokens": y}
    request_id: str = ""                     # Request ID for tracking
    error_detail: Optional[Dict[str, Any]] = None  # Structured error info: {"type": "...", "message": "..."}
    _stream_adapter_info: Optional[Dict[str, Any]] = None  # chain 模式流式回传 adapter 信息


def create_error_response(
    adapter_name: str,
    model_name: str,
    latency_ms: float,
    status_code: int,
    error: str,
    request_id: str = "",
    error_type: str = "adapter_error",
) -> AdapterResponse:
    """Create standardized error AdapterResponse.

    Args:
        adapter_name: Provider name
        model_name: Model being called
        latency_ms: Request latency in milliseconds
        status_code: HTTP status code for the error
        error: Error message string
        request_id: Request ID for tracking
        error_type: Error category for programmatic handling

    Returns:
        AdapterResponse with error details
    """
    log_level = logging.WARNING if status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        f"[{request_id}] api_error provider={adapter_name} "
        f"status={status_code} error={error}"
    )
    return AdapterResponse(
        status_code=status_code,
        success=False,
        adapter_name=adapter_name,
        model_name=model_name,
        latency_ms=latency_ms,
        error=error,
        request_id=request_id,
        error_detail={"type": error_type, "message": error},
    )


class BaseAdapter(ABC):
    """适配器基类，定义统一接口"""

    def __init__(self, provider_config: ProviderConfig):
        self.provider = provider_config
        self.name = provider_config.name

    @abstractmethod
    async def chat_completion(
        self,
        model_name: str,
        messages: list,
        stream: bool = False,
        timeout: int = 60,
        request_id: str = "",
        **kwargs: Any,
    ) -> AdapterResponse:
        """调用上游 API，返回统一封装的响应。"""

    async def health_check(self) -> bool:
        """检查上游是否可达"""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.provider.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.provider.api_key}",
                        **self.provider.custom_headers,
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False
