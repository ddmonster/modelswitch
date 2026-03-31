from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    name: str = ""
    description: str = ""
    rate_limit: int = 60
    daily_limit: int = 0
    allowed_models: list[str] = []


class UpdateKeyRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rate_limit: Optional[int] = None
    daily_limit: Optional[int] = None
    allowed_models: Optional[list[str]] = None
    expires_at: Optional[str] = None


@router.get("")
async def list_keys(request: Request):
    """获取所有 API Key 列表（脱敏）"""
    service = request.app.state.api_key_service
    return service.to_list()


@router.post("")
async def create_key(req: CreateKeyRequest, request: Request):
    """创建新 API Key"""
    service = request.app.state.api_key_service
    key_config = service.create(
        name=req.name,
        description=req.description,
        rate_limit=req.rate_limit,
        daily_limit=req.daily_limit,
        allowed_models=req.allowed_models,
    )
    # 持久化到配置文件
    await _sync_config(request)

    return JSONResponse(
        status_code=201,
        content={
            "key": key_config.key,
            "name": key_config.name,
            "message": "API Key created. Save the key, it won't be shown again.",
        },
    )


@router.put("/{key}")
async def update_key(key: str, req: UpdateKeyRequest, request: Request):
    """更新 API Key 配置"""
    service = request.app.state.api_key_service
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    result = service.update(key, **updates)
    if not result:
        return JSONResponse(status_code=404, content={"error": {"message": "API Key not found"}})
    # 更新配置并保存
    await _sync_config(request)
    return {"message": "API Key updated"}


@router.delete("/{key}")
async def delete_key(key: str, request: Request):
    """删除 API Key"""
    service = request.app.state.api_key_service
    if not service.delete(key):
        return JSONResponse(status_code=404, content={"error": {"message": "API Key not found"}})
    await _sync_config(request)
    return {"message": "API Key deleted"}


@router.patch("/{key}/toggle")
async def toggle_key(key: str, request: Request):
    """启用/禁用 API Key"""
    service = request.app.state.api_key_service
    result = service.toggle(key)
    if not result:
        return JSONResponse(status_code=404, content={"error": {"message": "API Key not found"}})
    await _sync_config(request)
    return {"key": service.mask_key(key), "enabled": result.enabled}


@router.get("/{key}/usage")
async def key_usage(key: str, request: Request, date_from: Optional[str] = None, date_to: Optional[str] = None):
    """获取指定 Key 的用量"""
    service = request.app.state.api_key_service
    key_config = service.get(key)
    if not key_config:
        return JSONResponse(status_code=404, content={"error": {"message": "API Key not found"}})

    tracker = request.app.state.usage_tracker
    result = await tracker.aggregate(
        group_by="api_key",
        date_from=date_from or datetime.now().strftime("%Y-%m-%d"),
        date_to=date_to or datetime.now().strftime("%Y-%m-%d"),
    )
    # 过滤出当前 key 的数据
    result["groups"] = [g for g in result["groups"] if g["name"] == key_config.name]
    return result


async def _sync_config(request: Request):
    """同步 service 中的变更回 config 并持久化"""
    service = request.app.state.api_key_service
    config = request.app.state.config
    config.api_keys = service.get_all()

    # 更新中间件
    if hasattr(request.app.state, "middleware"):
        request.app.state.middleware.reload_config(config)

    # 保存到文件
    from app.core.config import save_config
    save_config(config, request.app.state.config_path)
