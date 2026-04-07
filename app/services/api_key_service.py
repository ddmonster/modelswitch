from __future__ import annotations

import secrets
from datetime import datetime
from typing import Dict, List, Optional

from app.models.config_models import ApiKeyConfig


class ApiKeyService:
    """API Key CRUD + 验证"""

    def __init__(self, keys: List[ApiKeyConfig]):
        self._keys: Dict[str, ApiKeyConfig] = {k.key: k for k in keys}

    def get_all(self) -> List[ApiKeyConfig]:
        return list(self._keys.values())

    def get(self, key: str) -> Optional[ApiKeyConfig]:
        return self._keys.get(key)

    def create(
        self,
        name: str = "",
        description: str = "",
        rate_limit: int = 60,
        daily_limit: int = 0,
        allowed_models: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
    ) -> ApiKeyConfig:
        key = f"sk-{secrets.token_urlsafe(32)}"
        key_config = ApiKeyConfig(
            key=key,
            name=name or "unnamed",
            enabled=True,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            allowed_models=allowed_models or [],
            roles=roles or ["user"],
            created_at=datetime.now().isoformat(),
            description=description,
        )
        self._keys[key] = key_config
        return key_config

    def has_role(self, key: str, role: str) -> bool:
        """检查 key 是否拥有指定角色"""
        config = self._keys.get(key)
        if not config:
            return False
        return role in config.roles

    def update(self, key: str, **kwargs) -> Optional[ApiKeyConfig]:
        config = self._keys.get(key)
        if not config:
            return None
        for field, value in kwargs.items():
            if hasattr(config, field) and value is not None:
                setattr(config, field, value)
        return config

    def delete(self, key: str) -> bool:
        return self._keys.pop(key, None) is not None

    def toggle(self, key: str) -> Optional[ApiKeyConfig]:
        config = self._keys.get(key)
        if not config:
            return None
        config.enabled = not config.enabled
        return config

    def validate(self, key: str) -> Optional[ApiKeyConfig]:
        config = self._keys.get(key)
        if not config:
            return None
        if not config.enabled:
            return None
        # 检查过期
        if config.expires_at:
            try:
                if datetime.fromisoformat(config.expires_at) < datetime.now():
                    return None
            except ValueError:
                pass
        return config

    @staticmethod
    def mask_key(key: str) -> str:
        """脱敏显示 key"""
        if len(key) <= 8:
            return key[:4] + "***"
        return key[:7] + "***" + key[-4:]

    def reload(self, keys: List[ApiKeyConfig]) -> None:
        """热重载 keys"""
        self._keys = {k.key: k for k in keys}

    def to_list(self) -> List[dict]:
        """返回脱敏后的 key 列表"""
        result = []
        for key, config in self._keys.items():
            data = config.model_dump()
            data["key"] = self.mask_key(key)
            data["key_raw"] = key  # 供前端编辑使用，不直接展示
            result.append(data)
        return result
