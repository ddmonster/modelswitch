from __future__ import annotations

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
    _stream_adapter_info: Optional[Dict[str, Any]] = None  # chain 模式流式回传 adapter 信息


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
