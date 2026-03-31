"""兼容性模块 — 重导出 AdapterResponse 和 create_adapter 工厂函数。

所有新代码应直接从 app.adapters.base / openai_adapter / anthropic_adapter 导入。
"""
from __future__ import annotations

from app.adapters.base import AdapterResponse, BaseAdapter  # noqa: F401
from app.adapters.openai_adapter import OpenAIAdapter  # noqa: F401
from app.adapters.anthropic_adapter import AnthropicAdapter  # noqa: F401
from app.models.config_models import ProviderConfig


def create_adapter(provider_config: ProviderConfig) -> BaseAdapter:
    """根据 provider 类型创建对应的适配器实例。"""
    if provider_config.provider == "anthropic":
        return AnthropicAdapter(provider_config)
    return OpenAIAdapter(provider_config)


# 向后兼容别名
LiteLLMAdapter = OpenAIAdapter
