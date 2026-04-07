from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GatewaySettings(BaseModel):
    """网关基础设置"""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_max_bytes: int = 104857600  # 100MB
    log_backup_count: int = 30
    log_request_body: bool = True
    log_response_body: bool = False
    log_max_body_length: int = 4096
    usage_db: str = "data/usage.db"
    usage_flush_interval: int = 10


class ProviderConfig(BaseModel):
    """顶层 Provider 配置 - 供应商连接定义"""

    name: str = Field(..., description="唯一标识，如 dashscope")
    provider: str = Field(default="openai", description="协议类型：openai | anthropic")
    base_url: str = Field(..., description="上游 API 地址")
    api_key: str = Field(default="", description="API Key，支持 ${ENV_VAR}")
    custom_headers: Dict[str, str] = Field(
        default_factory=dict, description="自定义请求头"
    )
    enabled: bool = True
    # 请求队列配置
    max_concurrent: int = Field(default=0, description="最大并发请求数，0=不限制")
    max_queue_size: int = Field(default=100, description="最大队列长度")
    queue_timeout: float = Field(default=300.0, description="队列等待超时时间（秒）")


class ModelAdapterRef(BaseModel):
    """模型对 adapter 的引用配置"""

    adapter: str = Field(..., description="引用 ProviderConfig.name")
    model_name: str = Field(..., description="传给上游的实际模型名")
    priority: int = Field(default=1, description="chain 模式优先级，数字越小越优先")
    timeout: int = Field(default=60, description="超时秒数")


class ModelConfig(BaseModel):
    """虚拟模型配置"""

    mode: str = Field(default="chain", description="调用模式：chain | adapter")
    description: str = ""
    adapters: List[ModelAdapterRef] = Field(default_factory=list)


class ApiKeyConfig(BaseModel):
    """API Key 配置"""

    key: str = Field(..., description="API Key 值")
    name: str = Field(default="", description="显示别名")
    enabled: bool = True
    rate_limit: int = Field(default=60, description="每分钟请求上限，0=不限")
    daily_limit: int = Field(default=0, description="每日请求上限，0=不限")
    allowed_models: List[str] = Field(
        default_factory=list, description="允许的模型，空=全部"
    )
    roles: List[str] = Field(
        default_factory=lambda: ["user"], description="角色列表: admin | user"
    )
    expires_at: Optional[str] = Field(default=None, description="过期时间 ISO 格式")
    created_at: str = ""
    description: str = ""


class GatewayConfig(BaseModel):
    """全局配置"""

    gateway: GatewaySettings = GatewaySettings()
    providers: List[ProviderConfig] = Field(default_factory=list)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    api_keys: List[ApiKeyConfig] = Field(default_factory=list)


def _resolve_env_vars(value: str) -> str:
    """解析 ${ENV_VAR} 格式的环境变量引用"""
    if not isinstance(value, str):
        return value

    def replacer(match):
        env_name = match.group(1)
        import os

        return os.environ.get(env_name, match.group(0))

    return re.sub(r"\$\{(\w+)\}", replacer, value)


def resolve_config_env(config: GatewayConfig) -> GatewayConfig:
    """递归解析配置中所有 ${ENV_VAR} 引用"""
    for provider in config.providers:
        provider.api_key = _resolve_env_vars(provider.api_key)
        provider.base_url = _resolve_env_vars(provider.base_url)
    for key_config in config.api_keys:
        key_config.key = _resolve_env_vars(key_config.key)
    return config
