"""Shared test fixtures for ModelSwitch tests."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Set workspace to a temp dir for test isolation
os.environ.setdefault("MODELSWITCH_WORKSPACE", tempfile.mkdtemp())

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.config_models import (
    ApiKeyConfig,
    GatewayConfig,
    GatewaySettings,
    ModelAdapterRef,
    ModelConfig,
    ProviderConfig,
)
from app.utils.logging import clear_log_buffer

# ========== Sample config factories ==========


def make_provider(
    name="test-provider",
    provider="openai",
    base_url="https://api.test.com/v1",
    api_key="sk-test-key",
    enabled=True,
    custom_headers=None,
):
    return ProviderConfig(
        name=name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        enabled=enabled,
        custom_headers=custom_headers or {},
    )


def make_model(name="test-model", mode="chain", adapter_names=None, model_names=None):
    adapter_names = adapter_names or ["test-provider"]
    model_names = model_names or ["test-upstream-model"]
    adapters = [
        ModelAdapterRef(adapter=a, model_name=m, priority=i + 1, timeout=60)
        for i, (a, m) in enumerate(zip(adapter_names, model_names))
    ]
    return name, ModelConfig(
        mode=mode, description=f"Test model {name}", adapters=adapters
    )


def make_api_key(
    key="sk-test-admin",
    name="admin",
    enabled=True,
    rate_limit=60,
    daily_limit=0,
    allowed_models=None,
    roles=None,
):
    return ApiKeyConfig(
        key=key,
        name=name,
        enabled=enabled,
        rate_limit=rate_limit,
        daily_limit=daily_limit,
        allowed_models=allowed_models or [],
        roles=roles or ["user"],
        created_at="2026-01-01T00:00:00",
        description="Test key",
    )


def make_config(providers=None, models=None, api_keys=None):
    p = providers or [make_provider()]
    model_dict = {}
    if models is None:
        name, m = make_model()
        model_dict[name] = m
    elif isinstance(models, list):
        for item in models:
            if isinstance(item, tuple):
                model_dict[item[0]] = item[1]
            else:
                n, mc = make_model(item)
                model_dict[n] = mc
    else:
        model_dict = models

    keys = api_keys or [make_api_key()]
    return GatewayConfig(
        gateway=GatewaySettings(log_level="WARNING"),
        providers=p,
        models=model_dict,
        api_keys=keys,
    )


# ========== Fixtures ==========


@pytest.fixture(autouse=True)
def reset_log_buffer():
    """Clear log buffer before each test for isolation."""
    clear_log_buffer()
    yield


@pytest.fixture
def sample_config():
    """A basic config with 2 providers, 2 models, 1 api key."""
    p1 = make_provider("provider-a", base_url="https://a.test.com/v1", api_key="sk-aaa")
    p2 = make_provider("provider-b", base_url="https://b.test.com/v1", api_key="sk-bbb")
    _, m_chain = make_model(
        "chain-model",
        "chain",
        adapter_names=["provider-a", "provider-b"],
        model_names=["upstream-a", "upstream-b"],
    )
    _, m_adapter = make_model(
        "direct-model",
        "adapter",
        adapter_names=["provider-a"],
        model_names=["upstream-a"],
    )
    key = make_api_key(roles=["admin"])
    return make_config(
        providers=[p1, p2],
        models={"chain-model": m_chain, "direct-model": m_adapter},
        api_keys=[key],
    )


@pytest.fixture
def config_with_expired_key():
    p = make_provider()
    name, m = make_model()
    expired_key = ApiKeyConfig(
        key="sk-expired",
        name="expired-user",
        enabled=True,
        expires_at="2020-01-01T00:00:00",
        created_at="2020-01-01T00:00:00",
    )
    return make_config(providers=[p], models={name: m}, api_keys=[expired_key])


@pytest.fixture
def config_with_disabled_key():
    p = make_provider()
    name, m = make_model()
    disabled_key = ApiKeyConfig(
        key="sk-disabled",
        name="disabled-user",
        enabled=False,
        created_at="2026-01-01T00:00:00",
    )
    return make_config(providers=[p], models={name: m}, api_keys=[disabled_key])


@pytest.fixture
def config_with_rate_limit():
    p = make_provider()
    name, m = make_model()
    limited_key = ApiKeyConfig(
        key="sk-limited",
        name="limited-user",
        enabled=True,
        rate_limit=2,
        daily_limit=0,
        created_at="2026-01-01T00:00:00",
    )
    return make_config(providers=[p], models={name: m}, api_keys=[limited_key])


@pytest.fixture
def config_with_daily_limit():
    p = make_provider()
    name, m = make_model()
    limited_key = ApiKeyConfig(
        key="sk-daily-limited",
        name="daily-limited",
        enabled=True,
        rate_limit=0,
        daily_limit=2,
        created_at="2026-01-01T00:00:00",
    )
    return make_config(providers=[p], models={name: m}, api_keys=[limited_key])


@pytest.fixture
def config_with_model_restriction():
    p = make_provider()
    name, m = make_model("allowed-model")
    restricted_key = ApiKeyConfig(
        key="sk-restricted",
        name="restricted-user",
        enabled=True,
        allowed_models=["allowed-model"],
        created_at="2026-01-01T00:00:00",
    )
    return make_config(providers=[p], models={name: m}, api_keys=[restricted_key])


@pytest.fixture
def tmp_config_file():
    """Write a config to a temp file and return the path. Cleans up after test."""
    import yaml

    cfg = make_config()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg.model_dump(mode="json"), f)
        path = f.name
    yield path
    os.unlink(path)


async def _make_shared_tracker():
    """Create a UsageTracker with a shared in-memory SQLite connection."""
    import aiosqlite

    from app.services.usage_tracker import UsageTracker

    db = await aiosqlite.connect(":memory:")
    tracker = UsageTracker(db_path=":memory:", flush_interval=9999)
    tracker._shared_db = db
    await tracker.init()
    return tracker


@pytest_asyncio.fixture
async def usage_db():
    """In-memory SQLite usage tracker with shared connection."""
    tracker = await _make_shared_tracker()
    yield tracker
    await tracker.close()


# ========== FastAPI test client fixture ==========


def _create_test_app(config):
    """Create a FastAPI app for testing with mocked chain router."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from app.core.chain_router import ChainRouter
    from app.core.middleware import GatewayMiddleware
    from app.services.api_key_service import ApiKeyService

    app = FastAPI()

    active_requests = {"count": 0}
    app.add_middleware(
        GatewayMiddleware, config=config, active_requests_counter=active_requests
    )
    app.state.config = config
    # Use a temp file for config persistence during tests
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.close()
    app.state.config_path = tmp.name
    app.state.active_requests = active_requests
    app.state.chain_router = ChainRouter(config)
    app.state.api_key_service = ApiKeyService(config.api_keys)

    # Register routes
    from app.api.anthropic_routes import router as anthropic_router
    from app.api.api_key_routes import router as api_key_router
    from app.api.config_routes import router as config_router
    from app.api.log_routes import router as log_router
    from app.api.openai_routes import router as openai_router
    from app.api.usage_routes import router as usage_router

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.include_router(config_router)
    app.include_router(api_key_router)
    app.include_router(usage_router)
    app.include_router(log_router)

    # Mount static files for frontend test
    from pathlib import Path as PPath

    from fastapi.responses import FileResponse

    web_dir = PPath(__file__).parent.parent / "app" / "web"
    if web_dir.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

        @app.get("/")
        async def index():
            return FileResponse(str(web_dir / "index.html"))

    return app


@pytest_asyncio.fixture
async def client(sample_config):
    """Async HTTP test client with full app."""
    app = _create_test_app(sample_config)

    tracker = await _make_shared_tracker()
    app.state.usage_tracker = tracker

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer sk-test-admin"},
    ) as c:
        yield c
    # Cleanup: close tracker before event loop closes
    await tracker.close()
    # Cleanup temp config file
    try:
        os.unlink(app.state.config_path)
    except OSError:
        pass


@pytest_asyncio.fixture
async def client_expired_key(config_with_expired_key):
    app = _create_test_app(config_with_expired_key)
    tracker = await _make_shared_tracker()
    app.state.usage_tracker = tracker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await tracker.close()
    try:
        os.unlink(app.state.config_path)
    except OSError:
        pass


@pytest_asyncio.fixture
async def client_with_model_restriction(config_with_model_restriction):
    """Client with API key that has model restrictions."""
    app = _create_test_app(config_with_model_restriction)
    tracker = await _make_shared_tracker()
    app.state.usage_tracker = tracker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await tracker.close()
    try:
        os.unlink(app.state.config_path)
    except OSError:
        pass


@pytest_asyncio.fixture
async def client_rate_limited(config_with_rate_limit):
    app = _create_test_app(config_with_rate_limit)
    tracker = await _make_shared_tracker()
    app.state.usage_tracker = tracker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await tracker.close()
    try:
        os.unlink(app.state.config_path)
    except OSError:
        pass
