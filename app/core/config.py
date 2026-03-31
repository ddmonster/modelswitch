from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from app.models.config_models import GatewayConfig, resolve_config_env


def load_config(config_path: str = "config.yaml") -> GatewayConfig:
    """从 YAML 文件加载配置，解析环境变量"""
    path = Path(config_path)
    if not path.exists():
        return GatewayConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = GatewayConfig(**raw)
    config = resolve_config_env(config)
    return config


def save_config(config: GatewayConfig, config_path: str = "config.yaml") -> None:
    """将配置保存到 YAML 文件"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json")
    # 将敏感字段还原为 ${ENV_VAR} 格式暂不处理，直接保存实际值
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
