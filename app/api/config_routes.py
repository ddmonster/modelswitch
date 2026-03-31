from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.models.config_models import ProviderConfig, ModelConfig, ModelAdapterRef

router = APIRouter(prefix="/api/config", tags=["config"])


# ========== 全局配置 ==========

@router.get("")
async def get_config(request: Request):
    """获取完整配置（API Key 脱敏）"""
    config = request.app.state.config
    data = config.model_dump(mode="json")
    # 脱敏 API Keys
    for key_cfg in data.get("api_keys", []):
        k = key_cfg.get("key", "")
        key_cfg["key"] = k[:7] + "***" if len(k) > 11 else k[:4] + "***"
    return data


@router.put("")
async def update_config(request: Request):
    """替换完整配置"""
    body = await request.json()
    from app.models.config_models import GatewayConfig
    new_config = GatewayConfig(**body)

    # 重新加载
    request.app.state.config = new_config
    request.app.state.chain_router.reload_config(new_config)
    request.app.state.api_key_service.reload(new_config.api_keys)
    if hasattr(request.app.state, "middleware"):
        request.app.state.middleware.reload_config(new_config)

    # 持久化
    from app.core.config import save_config
    save_config(new_config, request.app.state.config_path)

    return {"message": "Configuration updated and reloaded"}


# ========== Providers CRUD ==========

class CreateProviderRequest(BaseModel):
    name: str
    provider: str = "openai"
    base_url: str
    api_key: str = ""
    custom_headers: dict = {}
    enabled: bool = True


class UpdateProviderRequest(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    custom_headers: Optional[dict] = None
    enabled: Optional[bool] = None


@router.get("/providers")
async def list_providers(request: Request):
    config = request.app.state.config
    return [p.model_dump() for p in config.providers]


@router.post("/providers")
async def create_provider(req: CreateProviderRequest, request: Request):
    config = request.app.state.config
    # 检查名称唯一
    if any(p.name == req.name for p in config.providers):
        return JSONResponse(status_code=409, content={"error": {"message": f"Provider '{req.name}' already exists"}})
    provider = ProviderConfig(**req.model_dump())
    config.providers.append(provider)
    await _reload(request, config)
    return {"message": "Provider created", "name": req.name}


@router.put("/providers/{name}")
async def update_provider(name: str, req: UpdateProviderRequest, request: Request):
    config = request.app.state.config
    provider = next((p for p in config.providers if p.name == name), None)
    if not provider:
        return JSONResponse(status_code=404, content={"error": {"message": "Provider not found"}})
    for k, v in req.model_dump(exclude_none=True).items():
        setattr(provider, k, v)
    await _reload(request, config)
    return {"message": "Provider updated"}


@router.delete("/providers/{name}")
async def delete_provider(name: str, request: Request):
    config = request.app.state.config
    config.providers = [p for p in config.providers if p.name != name]
    await _reload(request, config)
    return {"message": "Provider deleted"}


@router.patch("/providers/{name}/toggle")
async def toggle_provider(name: str, request: Request):
    config = request.app.state.config
    provider = next((p for p in config.providers if p.name == name), None)
    if not provider:
        return JSONResponse(status_code=404, content={"error": {"message": "Provider not found"}})
    provider.enabled = not provider.enabled
    await _reload(request, config)
    return {"name": name, "enabled": provider.enabled}


# ========== Models CRUD ==========

class CreateModelRequest(BaseModel):
    mode: str = "chain"
    description: str = ""
    adapters: list = []


class UpdateModelRequest(BaseModel):
    mode: Optional[str] = None
    description: Optional[str] = None
    adapters: Optional[list] = None


@router.get("/models")
async def list_models(request: Request):
    config = request.app.state.config
    result = {}
    for name, model in config.models.items():
        result[name] = model.model_dump()
    return result


@router.post("/models")
async def create_model(name: str, req: CreateModelRequest, request: Request):
    config = request.app.state.config
    if name in config.models:
        return JSONResponse(status_code=409, content={"error": {"message": f"Model '{name}' already exists"}})

    adapters = [ModelAdapterRef(**a) for a in req.adapters]
    model = ModelConfig(mode=req.mode, description=req.description, adapters=adapters)
    config.models[name] = model
    await _reload(request, config)
    return {"message": "Model created", "name": name}


@router.put("/models/{name}")
async def update_model(name: str, req: UpdateModelRequest, request: Request):
    config = request.app.state.config
    if name not in config.models:
        return JSONResponse(status_code=404, content={"error": {"message": "Model not found"}})
    model = config.models[name]
    for k, v in req.model_dump(exclude_none=True).items():
        if k == "adapters" and v is not None:
            model.adapters = [ModelAdapterRef(**a) for a in v]
        else:
            setattr(model, k, v)
    await _reload(request, config)
    return {"message": "Model updated"}


@router.delete("/models/{name}")
async def delete_model(name: str, request: Request):
    config = request.app.state.config
    if name not in config.models:
        return JSONResponse(status_code=404, content={"error": {"message": f"Model '{name}' not found"}})
    config.models.pop(name)
    await _reload(request, config)
    return {"message": "Model deleted"}


# ========== Health ==========

@router.get("/health")
async def health(request: Request):
    """健康检查"""
    chain_router = request.app.state.chain_router
    providers = chain_router.get_providers()
    provider_status = {}
    for name, provider in providers.items():
        provider_status[name] = {
            "enabled": provider.enabled,
            "base_url": provider.base_url,
        }
    return {
        "status": "healthy",
        "models": chain_router.list_models(),
        "providers": provider_status,
    }


async def _reload(request: Request, config):
    """重新加载配置并持久化"""
    from app.core.config import save_config
    from app.models.config_models import resolve_config_env

    config = resolve_config_env(config)
    request.app.state.config = config
    request.app.state.chain_router.reload_config(config)
    request.app.state.api_key_service.reload(config.api_keys)
    # Update middleware config by walking the middleware stack
    from app.core.middleware import GatewayMiddleware
    mw = request.app.middleware_stack
    while mw is not None:
        if isinstance(mw, GatewayMiddleware):
            mw.reload_config(config)
            break
        mw = getattr(mw, 'app', None)
    save_config(config, request.app.state.config_path)


# ========== 测试端点 ==========

@router.post("/providers/{name}/test")
async def test_provider(name: str, request: Request):
    """测试供应商连通性"""
    import time
    import httpx

    chain_router = request.app.state.chain_router
    providers = chain_router.get_providers()
    provider = providers.get(name)
    if not provider:
        return JSONResponse(status_code=404, content={"error": {"message": f"Provider '{name}' not found"}})
    if not provider.enabled:
        return {"success": False, "error": "Provider is disabled", "latency_ms": 0}

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Authorization": f"Bearer {provider.api_key}"}
            headers.update(provider.custom_headers)
            resp = await client.get(
                f"{provider.base_url}/models",
                headers=headers,
                timeout=8,
            )
        latency = (time.monotonic() - start) * 1000
        return {
            "success": 200 <= resp.status_code < 500,
            "status_code": resp.status_code,
            "latency_ms": round(latency, 0),
        }
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return {"success": False, "error": str(e), "latency_ms": round(latency, 0)}


@router.post("/models/{name}/test")
async def test_model(name: str, request: Request):
    """测试模型端到端调用"""
    import time

    chain_router = request.app.state.chain_router
    model_config = chain_router.get_model(name)
    if not model_config:
        return JSONResponse(status_code=404, content={"error": {"message": f"Model '{name}' not found"}})

    start = time.monotonic()
    try:
        result = await chain_router.execute_chat(
            model=name,
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            request_id="test",
            max_tokens=20,
        )
        latency = (time.monotonic() - start) * 1000

        if result.success:
            # Extract response preview
            preview = ""
            if result.body:
                if hasattr(result.body, "choices") and result.body.choices:
                    msg = result.body.choices[0].message
                    preview = getattr(msg, "content", "") or ""
                elif isinstance(result.body, dict):
                    choices = result.body.get("choices", [])
                    if choices:
                        preview = choices[0].get("message", {}).get("content", "")

            return {
                "success": True,
                "adapter_used": result.adapter_name,
                "model_name": result.model_name,
                "latency_ms": round(latency, 0),
                "usage": result.usage,
                "preview": preview[:200],
            }
        else:
            return {
                "success": False,
                "adapter_used": result.adapter_name,
                "latency_ms": round(latency, 0),
                "error": result.error,
            }
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return {"success": False, "latency_ms": round(latency, 0), "error": str(e)}
