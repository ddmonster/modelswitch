"""Tests for conversation_routes.py - conversation viewing API."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.config_models import (
    ApiKeyConfig,
    GatewayConfig,
    GatewaySettings,
    ModelAdapterRef,
    ModelConfig,
    ProviderConfig,
)

# ========== Config factories ==========


def make_provider(
    name="test-provider",
    provider="openai",
    base_url="https://api.test.com/v1",
    api_key="sk-test-key",
    enabled=True,
):
    return ProviderConfig(
        name=name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        enabled=enabled,
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
    roles=None,
):
    return ApiKeyConfig(
        key=key,
        name=name,
        enabled=enabled,
        rate_limit=rate_limit,
        daily_limit=daily_limit,
        allowed_models=[],
        roles=roles or ["user"],
        created_at="2026-01-01T00:00:00",
        description="Test key",
    )


def make_config(providers=None, models=None, api_keys=None, log_dir="logs"):
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
        gateway=GatewaySettings(log_level="WARNING", log_dir=log_dir),
        providers=p,
        models=model_dict,
        api_keys=keys,
    )


# ========== Cache reset fixture ==========


@pytest.fixture(autouse=True)
def reset_conversation_cache():
    """Reset the global cache before each test for isolation."""
    from app.api.conversation_routes import _cache

    _cache["mtime"] = 0.0
    _cache["records"] = []
    _cache["api_keys"] = []
    _cache["models"] = []
    yield


# ========== Test app factory ==========


def _create_test_app(config):
    """Create a FastAPI app for testing with conversation routes."""
    from fastapi import FastAPI

    from app.core.chain_router import ChainRouter
    from app.core.middleware import GatewayMiddleware
    from app.services.api_key_service import ApiKeyService

    app = FastAPI()

    active_requests = {"count": 0}
    app.add_middleware(
        GatewayMiddleware, config=config, active_requests_counter=active_requests
    )
    app.state.config = config
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.close()
    app.state.config_path = tmp.name
    app.state.active_requests = active_requests
    app.state.chain_router = ChainRouter(config)
    app.state.api_key_service = ApiKeyService(config.api_keys)

    # Register routes
    from app.api.conversation_routes import router as conversation_router

    app.include_router(conversation_router)

    return app


# ========== Fixtures ==========


@pytest_asyncio.fixture
async def client_with_temp_log_dir():
    """Create a client with a temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_config(log_dir=tmpdir, api_keys=[make_api_key(roles=["admin"])])
        app = _create_test_app(config)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer sk-test-admin"},
        ) as c:
            c._tmp_log_dir = tmpdir
            c._app = app
            yield c

        try:
            os.unlink(app.state.config_path)
        except OSError:
            pass


@pytest_asyncio.fixture
async def client_with_empty_log_dir():
    """Create a client with an empty log directory (no conversations file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_config(log_dir=tmpdir, api_keys=[make_api_key(roles=["admin"])])
        app = _create_test_app(config)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer sk-test-admin"},
        ) as c:
            c._tmp_log_dir = tmpdir
            c._app = app
            yield c

        try:
            os.unlink(app.state.config_path)
        except OSError:
            pass


def _write_conversations_file(log_dir: str, records: list[dict]):
    """Write a conversations.jsonl file with given records."""
    log_path = Path(log_dir) / "conversations.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return log_path


# ========== Test _output_preview directly ==========


class TestOutputPreview:
    """Tests for the _output_preview helper function."""

    def test_empty_output(self):
        from app.api.conversation_routes import _output_preview

        record = {}
        preview, has_tool_use = _output_preview(record)
        assert preview == ""
        assert has_tool_use is False

    def test_none_output(self):
        from app.api.conversation_routes import _output_preview

        record = {"output": None}
        preview, has_tool_use = _output_preview(record)
        assert preview == ""
        assert has_tool_use is False

    def test_non_list_output(self):
        from app.api.conversation_routes import _output_preview

        record = {"output": "not a list"}
        preview, has_tool_use = _output_preview(record)
        assert preview == ""
        assert has_tool_use is False

    def test_text_only(self):
        from app.api.conversation_routes import _output_preview

        record = {"output": [{"type": "text", "text": "Hello, this is a response."}]}
        preview, has_tool_use = _output_preview(record)
        assert preview == "Hello, this is a response."
        assert has_tool_use is False

    def test_text_truncated_to_100_chars(self):
        from app.api.conversation_routes import _output_preview

        long_text = "A" * 150
        record = {"output": [{"type": "text", "text": long_text}]}
        preview, has_tool_use = _output_preview(record)
        assert preview == "A" * 100
        assert has_tool_use is False

    def test_tool_use_only(self):
        from app.api.conversation_routes import _output_preview

        record = {
            "output": [
                {"type": "tool_use", "name": "get_weather", "input": {"city": "NYC"}}
            ]
        }
        preview, has_tool_use = _output_preview(record)
        assert preview == "[tool_use: get_weather]"
        assert has_tool_use is True

    def test_tool_use_without_name(self):
        from app.api.conversation_routes import _output_preview

        record = {"output": [{"type": "tool_use"}]}
        preview, has_tool_use = _output_preview(record)
        assert preview == "[tool_use: ]"
        assert has_tool_use is True

    def test_mixed_text_and_tool_use(self):
        from app.api.conversation_routes import _output_preview

        record = {
            "output": [
                {"type": "text", "text": "Let me check the weather."},
                {"type": "tool_use", "name": "get_weather", "input": {"city": "NYC"}},
                {"type": "text", "text": "The weather is sunny."},
            ]
        }
        preview, has_tool_use = _output_preview(record)
        assert preview == "Let me check the weather."
        assert has_tool_use is True

    def test_tool_use_before_text(self):
        from app.api.conversation_routes import _output_preview

        record = {
            "output": [
                {"type": "tool_use", "name": "search", "input": {"query": "test"}},
                {"type": "text", "text": "Here are the results."},
            ]
        }
        preview, has_tool_use = _output_preview(record)
        assert preview == "[tool_use: search]"
        assert has_tool_use is True

    def test_non_dict_block_skipped(self):
        from app.api.conversation_routes import _output_preview

        record = {
            "output": [
                "not a dict",
                {"type": "text", "text": "valid text"},
            ]
        }
        preview, has_tool_use = _output_preview(record)
        assert preview == "valid text"
        assert has_tool_use is False

    def test_unknown_type_skipped(self):
        from app.api.conversation_routes import _output_preview

        record = {
            "output": [
                {"type": "image", "url": "http://example.com/image.png"},
                {"type": "text", "text": "text after image"},
            ]
        }
        preview, has_tool_use = _output_preview(record)
        assert preview == "text after image"
        assert has_tool_use is False

    def test_multiple_tool_uses(self):
        from app.api.conversation_routes import _output_preview

        record = {
            "output": [
                {"type": "tool_use", "name": "search", "input": {}},
                {"type": "tool_use", "name": "get_weather", "input": {}},
            ]
        }
        preview, has_tool_use = _output_preview(record)
        assert preview == "[tool_use: search]"
        assert has_tool_use is True


# ========== Test _load_records ==========


class TestLoadRecords:
    """Tests for the _load_records helper function."""

    def test_no_file(self, tmp_path):
        from app.api.conversation_routes import _load_records

        log_path = tmp_path / "conversations.jsonl"
        records = _load_records(log_path)
        assert records == []

    def test_single_record(self, tmp_path):
        from app.api.conversation_routes import _load_records

        log_path = tmp_path / "conversations.jsonl"
        log_path.write_text('{"api_key": "sk-test", "model": "gpt-4"}\n')

        records = _load_records(log_path)
        assert len(records) == 1
        assert records[0]["api_key"] == "sk-test"
        assert records[0]["model"] == "gpt-4"
        assert records[0]["_line"] == 0

    def test_multiple_records(self, tmp_path):
        from app.api.conversation_routes import _cache, _load_records

        log_path = tmp_path / "conversations.jsonl"
        log_path.write_text(
            '{"api_key": "sk-1", "model": "gpt-4"}\n'
            '{"api_key": "sk-2", "model": "claude-3"}\n'
            '{"api_key": "sk-1", "model": "gpt-4"}\n'
        )

        records = _load_records(log_path)
        assert len(records) == 3
        assert records[0]["_line"] == 0
        assert records[1]["_line"] == 1
        assert records[2]["_line"] == 2
        assert _cache["api_keys"] == ["sk-1", "sk-2"]
        assert _cache["models"] == ["claude-3", "gpt-4"]

    def test_skip_empty_lines(self, tmp_path):
        from app.api.conversation_routes import _load_records

        log_path = tmp_path / "conversations.jsonl"
        log_path.write_text(
            '{"api_key": "sk-1", "model": "gpt-4"}\n'
            "\n"
            "   \n"
            '{"api_key": "sk-2", "model": "claude-3"}\n'
        )

        records = _load_records(log_path)
        assert len(records) == 2
        assert records[0]["_line"] == 0
        assert records[1]["_line"] == 3

    def test_skip_invalid_json(self, tmp_path):
        from app.api.conversation_routes import _load_records

        log_path = tmp_path / "conversations.jsonl"
        log_path.write_text(
            '{"api_key": "sk-1", "model": "gpt-4"}\n'
            "invalid json\n"
            '{"api_key": "sk-2", "model": "claude-3"}\n'
        )

        records = _load_records(log_path)
        assert len(records) == 2
        assert records[0]["api_key"] == "sk-1"
        assert records[1]["api_key"] == "sk-2"

    def test_cache_hit(self, tmp_path):
        import time

        from app.api.conversation_routes import _cache, _load_records

        log_path = tmp_path / "conversations.jsonl"
        log_path.write_text('{"api_key": "sk-test", "model": "gpt-4"}\n')

        # First load
        records1 = _load_records(log_path)
        assert len(records1) == 1

        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        log_path.write_text(
            '{"api_key": "sk-test", "model": "gpt-4"}\n'
            '{"api_key": "sk-test2", "model": "claude"}\n'
        )

        # Clear cache to force reload
        _cache["mtime"] = 0.0
        _cache["records"] = []
        records2 = _load_records(log_path)
        assert len(records2) == 2


# ========== Test _get_log_path ==========


class TestGetLogPath:
    """Tests for the _get_log_path helper function."""

    def test_default_log_dir(self):
        from app.api.conversation_routes import _get_log_path

        app_state = MagicMock()
        app_state.config = None

        path = _get_log_path(app_state)
        assert path == Path("logs") / "conversations.jsonl"

    def test_custom_log_dir(self):
        from app.api.conversation_routes import _get_log_path

        app_state = MagicMock()
        config = MagicMock()
        config.gateway = MagicMock()
        config.gateway.log_dir = "/var/log/gateway"
        app_state.config = config

        path = _get_log_path(app_state)
        assert path == Path("/var/log/gateway") / "conversations.jsonl"

    def test_config_without_gateway(self):
        from app.api.conversation_routes import _get_log_path

        app_state = MagicMock()
        config = MagicMock(spec=[])  # No gateway attribute
        app_state.config = config

        path = _get_log_path(app_state)
        assert path == Path("logs") / "conversations.jsonl"


# ========== Test list_conversations endpoint ==========


class TestListConversations:
    """Tests for GET /api/conversations."""

    @pytest.mark.asyncio
    async def test_list_empty_no_file(self, client_with_empty_log_dir):
        """Test listing when conversations file doesn't exist."""
        resp = await client_with_empty_log_dir.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["api_keys"] == []
        assert data["models"] == []
        assert data["limit"] == 50
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_with_data(self, client_with_temp_log_dir):
        """Test listing with actual conversation records."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
                "latency_ms": 100,
                "tokens_in": 10,
                "tokens_out": 20,
                "output": [{"type": "text", "text": "Hello"}],
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "claude-3",
                "adapter": "anthropic",
                "api_key": "sk-admin",
                "success": False,
                "latency_ms": 200,
                "tokens_in": 15,
                "tokens_out": 0,
                "output": [],
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        # Should be reversed (newest first)
        assert data["items"][0]["request_id"] == "req-2"
        assert data["items"][1]["request_id"] == "req-1"
        assert sorted(data["api_keys"]) == ["sk-admin", "sk-test"]
        assert sorted(data["models"]) == ["claude-3", "gpt-4"]

    @pytest.mark.asyncio
    async def test_filter_by_api_key(self, client_with_temp_log_dir):
        """Test filtering by api_key."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-user1",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-user2",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req-3",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-user1",
                "success": True,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?api_key=sk-user1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(item["api_key"] == "sk-user1" for item in data["items"])

    @pytest.mark.asyncio
    async def test_filter_by_model(self, client_with_temp_log_dir):
        """Test filtering by model."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "claude-3",
                "adapter": "anthropic",
                "api_key": "sk-test",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req-3",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?model=gpt-4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(item["model"] == "gpt-4" for item in data["items"])

    @pytest.mark.asyncio
    async def test_filter_by_success_true(self, client_with_temp_log_dir):
        """Test filtering by success=true."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": False,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?success=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_filter_by_success_false(self, client_with_temp_log_dir):
        """Test filtering by success=false."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": False,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?success=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["success"] is False

    @pytest.mark.asyncio
    async def test_filter_by_success_empty_string(self, client_with_temp_log_dir):
        """Test that success with empty string returns all records."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": False,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?success=")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_pagination_limit(self, client_with_temp_log_dir):
        """Test pagination with limit."""
        records = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "request_id": f"req-{i}",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            }
            for i in range(10)
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["items"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_pagination_offset(self, client_with_temp_log_dir):
        """Test pagination with offset."""
        records = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "request_id": f"req-{i}",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            }
            for i in range(10)
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?limit=3&offset=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["items"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 5
        # Items should be reversed (newest first), so offset 5 skips last 5
        # Original order: req-0, req-1, ..., req-9
        # Reversed: req-9, req-8, ..., req-0
        # Offset 5: req-4, req-3, req-2
        assert data["items"][0]["request_id"] == "req-4"

    @pytest.mark.asyncio
    async def test_combined_filters(self, client_with_temp_log_dir):
        """Test combining multiple filters."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-user1",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-user2",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req-3",
                "model": "claude-3",
                "adapter": "anthropic",
                "api_key": "sk-user1",
                "success": False,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get(
            "/api/conversations?api_key=sk-user1&success=true"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["request_id"] == "req-1"

    @pytest.mark.asyncio
    async def test_item_fields(self, client_with_temp_log_dir):
        """Test that items contain all expected fields."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
                "latency_ms": 150,
                "tokens_in": 10,
                "tokens_out": 20,
                "output": [{"type": "text", "text": "Hello world"}],
            }
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        item = data["items"][0]

        assert item["line"] == 0
        assert item["timestamp"] == "2024-01-01T10:00:00Z"
        assert item["request_id"] == "req-1"
        assert item["model"] == "gpt-4"
        assert item["adapter"] == "openai"
        assert item["api_key"] == "sk-test"
        assert item["success"] is True
        assert item["latency_ms"] == 150
        assert item["tokens_in"] == 10
        assert item["tokens_out"] == 20
        assert item["output_preview"] == "Hello world"
        assert item["has_tool_use"] is False

    @pytest.mark.asyncio
    async def test_item_defaults(self, client_with_temp_log_dir):
        """Test that items have sensible defaults for missing fields."""
        records = [{}]  # Empty record
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        item = data["items"][0]

        assert item["line"] == 0
        assert item["timestamp"] == ""
        assert item["request_id"] == ""
        assert item["model"] == ""
        assert item["adapter"] == ""
        assert item["api_key"] == ""
        assert item["success"] is True  # Default
        assert item["latency_ms"] == 0
        assert item["tokens_in"] == 0
        assert item["tokens_out"] == 0
        assert item["output_preview"] == ""
        assert item["has_tool_use"] is False


# ========== Test get_conversation_detail endpoint ==========


class TestGetConversationDetail:
    """Tests for GET /api/conversations/{line}."""

    @pytest.mark.asyncio
    async def test_get_detail_found(self, client_with_temp_log_dir):
        """Test getting a conversation detail by line number."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
                "messages": [{"role": "user", "content": "Hello"}],
                "output": [{"type": "text", "text": "Hi there!"}],
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "claude-3",
                "adapter": "anthropic",
                "api_key": "sk-admin",
                "success": False,
                "messages": [{"role": "user", "content": "Test"}],
                "output": [],
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-1"
        assert data["line"] == 0
        assert data["model"] == "gpt-4"
        assert data["messages"] == [{"role": "user", "content": "Hello"}]
        assert data["output"] == [{"type": "text", "text": "Hi there!"}]
        # Internal _line field should be excluded
        assert "_line" not in data

    @pytest.mark.asyncio
    async def test_get_detail_second_line(self, client_with_temp_log_dir):
        """Test getting the second conversation."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "claude-3",
                "adapter": "anthropic",
                "api_key": "sk-admin",
                "success": False,
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-2"
        assert data["line"] == 1

    @pytest.mark.asyncio
    async def test_get_detail_not_found(self, client_with_temp_log_dir):
        """Test getting a conversation that doesn't exist."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
            }
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations/999")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"] == "Record not found"

    @pytest.mark.asyncio
    async def test_get_detail_no_file(self, client_with_empty_log_dir):
        """Test getting detail when file doesn't exist."""
        resp = await client_with_empty_log_dir.get("/api/conversations/0")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"] == "Record not found"

    @pytest.mark.asyncio
    async def test_get_detail_internal_fields_excluded(self, client_with_temp_log_dir):
        """Test that internal _line field is excluded from response."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "_internal": "should be excluded",
            }
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations/0")
        assert resp.status_code == 200
        data = resp.json()
        assert "_internal" not in data
        assert data["line"] == 0  # line should be included


# ========== Test caching behavior ==========


class TestCaching:
    """Tests for the mtime-based caching in _load_records."""

    @pytest.mark.asyncio
    async def test_cache_returns_same_records(self, client_with_temp_log_dir):
        """Test that cache returns same records when file hasn't changed."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
            }
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        # First request
        resp1 = await client_with_temp_log_dir.get("/api/conversations")
        assert resp1.status_code == 200
        data1 = resp1.json()

        # Second request (should hit cache)
        resp2 = await client_with_temp_log_dir.get("/api/conversations")
        assert resp2.status_code == 200
        data2 = resp2.json()

        assert data1 == data2

    @pytest.mark.asyncio
    async def test_cache_reloads_on_mtime_change(self, client_with_temp_log_dir):
        """Test that cache reloads when file modification time changes."""
        import time

        records1 = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
            }
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records1)

        # First request
        resp1 = await client_with_temp_log_dir.get("/api/conversations")
        assert resp1.status_code == 200
        assert resp1.json()["total"] == 1

        # Wait and write new file (mtime will change)
        time.sleep(0.1)
        records2 = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
            },
            {
                "timestamp": "2024-01-01T11:00:00Z",
                "request_id": "req-2",
                "model": "claude-3",
            },
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records2)

        # Second request should see new data
        resp2 = await client_with_temp_log_dir.get("/api/conversations")
        assert resp2.status_code == 200
        assert resp2.json()["total"] == 2


# ========== Test edge cases ==========


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_limit_validation_min(self, client_with_empty_log_dir):
        """Test limit validation with minimum value."""
        resp = await client_with_empty_log_dir.get("/api/conversations?limit=1")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_limit_validation_max(self, client_with_empty_log_dir):
        """Test limit validation with maximum value."""
        resp = await client_with_empty_log_dir.get("/api/conversations?limit=200")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_limit_validation_below_min(self, client_with_empty_log_dir):
        """Test limit validation below minimum."""
        resp = await client_with_empty_log_dir.get("/api/conversations?limit=0")
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_limit_validation_above_max(self, client_with_empty_log_dir):
        """Test limit validation above maximum."""
        resp = await client_with_empty_log_dir.get("/api/conversations?limit=201")
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_offset_validation_min(self, client_with_empty_log_dir):
        """Test offset validation with minimum value."""
        resp = await client_with_empty_log_dir.get("/api/conversations?offset=0")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_offset_validation_negative(self, client_with_empty_log_dir):
        """Test offset validation with negative value."""
        resp = await client_with_empty_log_dir.get("/api/conversations?offset=-1")
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_offset_beyond_total(self, client_with_temp_log_dir):
        """Test offset beyond total records returns empty."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
            }
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        resp = await client_with_temp_log_dir.get("/api/conversations?offset=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_empty_lines_in_file(self, client_with_temp_log_dir):
        """Test handling of empty lines in JSONL file."""
        log_path = Path(client_with_temp_log_dir._tmp_log_dir) / "conversations.jsonl"
        with open(log_path, "w") as f:
            f.write('{"request_id": "req-1", "model": "gpt-4"}\n')
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace line
            f.write('{"request_id": "req-2", "model": "claude-3"}\n')

        resp = await client_with_temp_log_dir.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_malformed_json_lines(self, client_with_temp_log_dir):
        """Test handling of malformed JSON lines."""
        log_path = Path(client_with_temp_log_dir._tmp_log_dir) / "conversations.jsonl"
        with open(log_path, "w") as f:
            f.write('{"request_id": "req-1", "model": "gpt-4"}\n')
            f.write("not valid json\n")
            f.write('{"request_id": "req-2", "model": "claude-3"}\n')

        resp = await client_with_temp_log_dir.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_large_dataset_pagination(self, client_with_temp_log_dir):
        """Test pagination with a larger dataset."""
        records = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "request_id": f"req-{i}",
                "model": "gpt-4",
                "adapter": "openai",
                "api_key": "sk-test",
                "success": True,
            }
            for i in range(150)
        ]
        _write_conversations_file(client_with_temp_log_dir._tmp_log_dir, records)

        # First page
        resp1 = await client_with_temp_log_dir.get(
            "/api/conversations?limit=50&offset=0"
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["total"] == 150
        assert len(data1["items"]) == 50

        # Second page
        resp2 = await client_with_temp_log_dir.get(
            "/api/conversations?limit=50&offset=50"
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["total"] == 150
        assert len(data2["items"]) == 50

        # Third page
        resp3 = await client_with_temp_log_dir.get(
            "/api/conversations?limit=50&offset=100"
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["total"] == 150
        assert len(data3["items"]) == 50
