from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.chain_router import ChainRouter
from app.core.config import load_config
from app.core.middleware import GatewayMiddleware
from app.core.request_queue import get_queue_manager
from app.models.config_models import resolve_config_env
from app.services.api_key_service import ApiKeyService
from app.services.conv_indexer import ConvIndexer, set_conv_indexer
from app.services.usage_tracker import UsageTracker
from app.utils.logging import setup_logging

from app.workspace import config_path, data_dir, resolve_workspace, web_dir as pkg_web_dir

# ========== 模块级加载配置 ==========
resolve_workspace()  # init from env var or default
CONFIG_PATH = str(config_path())
_initial_config = load_config(CONFIG_PATH)
_initial_config = resolve_config_env(_initial_config)

# 活跃请求计数（模块级共享）
_active_requests = {"count": 0}


# ========== lifespan（必须在 app 创建之前定义）==========


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ========== 启动 ==========
    print("=" * 50)
    print("  ModelSwitch Gateway Starting...")
    print("=" * 50)

    # 1. 加载配置
    config = load_config(CONFIG_PATH)
    config = resolve_config_env(config)
    app.state.config = config
    app.state.config_path = CONFIG_PATH
    app.state.active_requests = _active_requests

    # 2. 设置日志
    setup_logging(
        log_level=config.gateway.log_level,
        log_dir=config.gateway.log_dir,
        max_bytes=config.gateway.log_max_bytes,
        backup_count=config.gateway.log_backup_count,
    )
    logger = logging.getLogger("modelswitch")
    logger.info(
        f"Gateway 配置加载完成：{len(config.providers)} providers, {len(config.models)} models, {len(config.api_keys)} keys"
    )

    # 3. 初始化核心组件
    app.state.chain_router = ChainRouter(config)
    app.state.api_key_service = ApiKeyService(config.api_keys)

    # 3.5 初始化请求队列管理器
    queue_manager = get_queue_manager()
    for provider in config.providers:
        if provider.max_concurrent > 0:
            queue_manager.register_provider(
                provider_name=provider.name,
                max_concurrent=provider.max_concurrent,
                max_queue_size=provider.max_queue_size,
                queue_timeout=provider.queue_timeout,
            )
    await queue_manager.start()
    app.state.queue_manager = queue_manager

    # 4. 初始化用量统计
    usage_tracker = UsageTracker(
        db_path=config.gateway.usage_db,
        flush_interval=config.gateway.usage_flush_interval,
    )
    await usage_tracker.init()
    app.state.usage_tracker = usage_tracker

    # 4.5 初始化会话日志索引
    conv_indexer = ConvIndexer(db_path=str(data_dir() / "conv_index.db"))
    # Auto-rebuild if index is empty
    if conv_indexer.record_count() == 0:
        logger.info("会话索引为空，正在重建...")
        count = conv_indexer.rebuild_from_logs(config.gateway.log_dir)
        logger.info(f"会话索引重建完成：{count} 条记录")
    else:
        logger.info(f"会话索引已加载：{conv_indexer.record_count()} 条记录")
    app.state.conv_indexer = conv_indexer
    set_conv_indexer(conv_indexer)

    # 定时 flush 用量数据
    async def flush_loop():
        while True:
            await asyncio.sleep(config.gateway.usage_flush_interval)
            await usage_tracker.flush()

    flush_task = asyncio.create_task(flush_loop())

    # 5. 预热 httpx 连接池
    logger.info("预热 provider 连接...")
    async with httpx.AsyncClient(timeout=10) as client:
        for provider in config.providers:
            if not provider.enabled:
                continue
            try:
                headers = {"Authorization": f"Bearer {provider.api_key}"}
                headers.update(provider.custom_headers)
                resp = await client.get(
                    f"{provider.base_url}/models",
                    headers=headers,
                    timeout=5,
                )
                logger.info(f"  ✓ {provider.name} 连接成功 ({resp.status_code})")
            except Exception as e:
                logger.warning(f"  ✗ {provider.name} 连接失败: {e}")

    # 6. 配置热重载
    def reload_config():
        try:
            new_config = load_config(str(config_path()))
            new_config = resolve_config_env(new_config)
            app.state.config = new_config
            app.state.chain_router.reload_config(new_config)
            app.state.api_key_service.reload(new_config.api_keys)
            _update_middleware_config(app, new_config)
            # 同步队列管理器
            to_remove = queue_manager.sync_providers(new_config.providers)
            for name in to_remove:
                queue_manager.unregister_provider(name)
        except Exception as e:
            logging.getLogger("modelswitch").error(f"热重载失败: {e}")

    config_watcher = None
    try:
        from app.core.config_watcher import ConfigWatcher

        config_watcher = ConfigWatcher(CONFIG_PATH, reload_config)
        config_watcher.start()
    except ImportError:
        logger.warning("watchdog 未安装，配置热重载不可用")
    except Exception as e:
        logger.warning(f"配置热重载启动失败: {e}")

    # 7. 更新 Prometheus 指标
    from app.utils.metrics import ACTIVE_REQUESTS as ACTIVE_GAUGE

    async def update_active_gauge():
        while True:
            ACTIVE_GAUGE.set(_active_requests["count"])
            await asyncio.sleep(1)

    gauge_task = asyncio.create_task(update_active_gauge())

    logger.info(f"Gateway 启动完成，监听 {config.gateway.host}:{config.gateway.port}")

    yield

    # ========== 关闭 ==========
    logger.info("Gateway 正在关闭...")

    if config_watcher:
        config_watcher.stop()

    flush_task.cancel()
    gauge_task.cancel()

    # 停止请求队列管理器
    await queue_manager.stop()

    await usage_tracker.flush()
    await usage_tracker.close()
    conv_indexer.close()

    timeout = 30
    while _active_requests["count"] > 0 and timeout > 0:
        logger.info(f"等待 {_active_requests['count']} 个活跃请求完成... ({timeout}s)")
        await asyncio.sleep(1)
        timeout -= 1
    if _active_requests["count"] > 0:
        logger.warning(f"超时，强制关闭 {_active_requests['count']} 个活跃请求")

    logger.info("Gateway 已关闭")


def _update_middleware_config(app: FastAPI, config):
    """更新中间件中的 API Key 配置（热重载时调用）"""
    mw = app.middleware_stack
    while mw is not None:
        if isinstance(mw, GatewayMiddleware):
            mw.reload_config(config)
            return
        mw = getattr(mw, "app", None)


# ========== 创建 FastAPI 应用 ==========

app = FastAPI(
    title="ModelSwitch Gateway",
    description="LLM Gateway Proxy with multi-provider fallback",
    version="0.1.0",
    lifespan=lifespan,
)

# 添加网关中间件（纯 ASGI，无 BaseHTTPMiddleware）
app.add_middleware(
    GatewayMiddleware, config=_initial_config, active_requests_counter=_active_requests
)


# ========== 路由 ==========

from app.api.anthropic_routes import router as anthropic_router
from app.api.api_key_routes import router as api_key_router
from app.api.config_routes import router as config_router
from app.api.conversation_routes import router as conversation_router
from app.api.log_routes import router as log_router
from app.api.openai_routes import router as openai_router
from app.api.usage_routes import router as usage_router

app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(config_router)
app.include_router(api_key_router)
app.include_router(usage_router)
app.include_router(log_router)
app.include_router(conversation_router)


# ========== 前端静态文件 ==========

web_dir = pkg_web_dir()
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

    @app.get("/")
    async def index():
        return FileResponse(str(web_dir / "index.html"))


# ========== Prometheus 指标 ==========

try:
    from prometheus_client import make_asgi_app

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
except ImportError:
    pass


# ========== 全局异常处理 ==========

from fastapi.responses import JSONResponse

from app.core.exceptions import GatewayError


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError):
    path = request.url.path
    if "/v1/messages" in path or "/anthropic/" in path:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": exc.message},
            },
        )
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": "upstream_error",
                    "code": exc.detail.get("type", ""),
                }
            },
        )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logging.getLogger("modelswitch").error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "server_error"}},
    )


# ========== 直接运行 ==========

if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    uvicorn.run(
        "app.main:app",
        host=config.gateway.host,
        port=config.gateway.port,
        reload=False,
        log_level="warning",
    )
