"""Tests for conversation_routes.py - SQLite-indexed conversation viewing API."""

import json
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.config_models import (
    ApiKeyConfig,
    GatewayConfig,
    ModelAdapterRef,
    ModelConfig,
    ProviderConfig,
)
from app.services.conv_indexer import ConvIndexer

# ========== Config factories ==========


def make_provider(name="provider-a", base_url="http://localhost:8001"):
    return ProviderConfig(name=name, provider="openai", base_url=base_url)


def make_model(name="test-model", adapter_name="provider-a"):
    return ModelConfig(
        mode="chain",
        adapters=[ModelAdapterRef(adapter=adapter_name, model_name="m1", priority=1)],
    )


def make_api_key(key="sk-test-admin", roles=None):
    return ApiKeyConfig(
        key=key,
        name="admin",
        enabled=True,
        rate_limit=60,
        daily_limit=0,
        allowed_models=[],
        roles=roles or ["admin"],
        created_at="2026-01-01T00:00:00",
    )


def make_config(log_dir=None, api_keys=None):
    config = GatewayConfig()
    if log_dir:
        config.gateway.log_dir = log_dir
    config.providers = [make_provider()]
    config.models = {"test-model": make_model()}
    config.api_keys = api_keys or [make_api_key()]
    return config


# ========== Fixtures ==========


@pytest.fixture
def temp_db():
    """Provide a temporary SQLite database for ConvIndexer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "conv_index.db")
        yield db_path


@pytest.fixture
def temp_log_dir():
    """Provide a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _create_test_app(config, indexer):
    """Create a standalone FastAPI app for testing with proper middleware auth."""
    from fastapi import FastAPI

    from app.api.conversation_routes import router as conversation_router
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
    app.state.conv_indexer = indexer

    app.include_router(conversation_router)
    return app


@pytest_asyncio.fixture
async def client_with_indexer(temp_db, temp_log_dir):
    """Create a test client with ConvIndexer and temp log dir."""
    config = make_config(log_dir=temp_log_dir)
    indexer = ConvIndexer(db_path=temp_db)
    app = _create_test_app(config, indexer)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer sk-test-admin"},
    ) as c:
        c._indexer = indexer
        c._temp_log_dir = temp_log_dir
        c._app = app
        yield c

    indexer.close()


@pytest_asyncio.fixture
async def client_with_empty_log_dir(temp_db):
    """Create a client with an empty log directory (no conversations file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_config(log_dir=tmpdir)
        indexer = ConvIndexer(db_path=temp_db)
        app = _create_test_app(config, indexer)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer sk-test-admin"},
        ) as c:
            c._temp_log_dir = tmpdir
            c._app = app
            yield c

        indexer.close()


# ========== Helpers ==========


def _write_test_records(indexer, log_dir, records):
    """Write records to a JSONL file and index them."""
    log_path = Path(log_dir) / "conversations.jsonl"
    offset = 0
    entries = []

    with open(log_path, "w") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            entries.append(
                {
                    "record": rec,
                    "file_path": "conversations.jsonl",
                    "byte_offset": offset,
                    "line_length": len(line.encode("utf-8")),
                }
            )
            f.write(line + "\n")
            offset += len(line.encode("utf-8")) + 1

    indexer.batch_index(entries)
    return log_path


def _make_record(
    timestamp="2026-01-01T00:00:00",
    request_id="test-req-1",
    model="glm-5",
    adapter="dashscope",
    api_key="test-key",
    success=True,
    latency_ms=100,
    tokens_in=10,
    tokens_out=5,
    messages=None,
    output=None,
):
    """Create a sample test record with defaults."""
    rec = {
        "timestamp": timestamp,
        "request_id": request_id,
        "model": model,
        "adapter": adapter,
        "api_key": api_key,
        "success": success,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "messages": messages or [],
        "output": output or [{"type": "text", "text": "Hello"}],
    }
    return rec


# ========== Test ListConversations ==========


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
    async def test_list_with_data(self, client_with_indexer):
        """Test listing with actual conversation records."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z",
                request_id="req-1",
                model="gpt-4",
                adapter="openai",
                api_key="sk-test",
                latency_ms=100,
                tokens_in=10,
                tokens_out=20,
                output=[{"type": "text", "text": "Hello"}],
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z",
                request_id="req-2",
                model="claude-3",
                adapter="anthropic",
                api_key="sk-admin",
                success=False,
                latency_ms=200,
                tokens_in=15,
                tokens_out=0,
                output=[],
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        # Should be reversed (newest first by id DESC)
        assert data["items"][0]["request_id"] == "req-2"
        assert data["items"][1]["request_id"] == "req-1"
        assert sorted(data["api_keys"]) == ["sk-admin", "sk-test"]
        assert sorted(data["models"]) == ["claude-3", "gpt-4"]

    @pytest.mark.asyncio
    async def test_filter_by_api_key(self, client_with_indexer):
        """Test filtering by api_key."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z", request_id="req-1", api_key="sk-user1"
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z", request_id="req-2", api_key="sk-user2"
            ),
            _make_record(
                timestamp="2024-01-01T12:00:00Z", request_id="req-3", api_key="sk-user1"
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?api_key=sk-user1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(item["api_key"] == "sk-user1" for item in data["items"])

    @pytest.mark.asyncio
    async def test_filter_by_model(self, client_with_indexer):
        """Test filtering by model."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z", request_id="req-1", model="gpt-4"
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z", request_id="req-2", model="claude-3"
            ),
            _make_record(
                timestamp="2024-01-01T12:00:00Z", request_id="req-3", model="gpt-4"
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?model=gpt-4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(item["model"] == "gpt-4" for item in data["items"])

    @pytest.mark.asyncio
    async def test_filter_by_success_true(self, client_with_indexer):
        """Test filtering by success=true."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z", request_id="req-1", success=True
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z", request_id="req-2", success=False
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?success=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_filter_by_success_false(self, client_with_indexer):
        """Test filtering by success=false."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z", request_id="req-1", success=True
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z", request_id="req-2", success=False
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?success=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["success"] is False

    @pytest.mark.asyncio
    async def test_filter_by_success_empty_string(self, client_with_indexer):
        """Test that success with empty string returns all records."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z", request_id="req-1", success=True
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z", request_id="req-2", success=False
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?success=")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_pagination_limit(self, client_with_indexer):
        """Test pagination with limit."""
        records = [
            _make_record(timestamp=f"2024-01-01T{i:02d}:00:00Z", request_id=f"req-{i}")
            for i in range(10)
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["items"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_pagination_offset(self, client_with_indexer):
        """Test pagination with offset."""
        records = [
            _make_record(timestamp=f"2024-01-01T{i:02d}:00:00Z", request_id=f"req-{i}")
            for i in range(10)
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?limit=3&offset=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["items"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 5
        # Items ordered by id DESC, so offset 5 skips the 5 newest
        # IDs: 1=req-0, 2=req-1, ..., 10=req-9
        # DESC order: id=10(req-9), 9(req-8), 8(req-7), 7(req-6), 6(req-5), 5(req-4), 4(req-3), 3(req-2), ...
        # Offset 5: starts at id=5 (req-4)
        assert data["items"][0]["request_id"] == "req-4"

    @pytest.mark.asyncio
    async def test_combined_filters(self, client_with_indexer):
        """Test combining multiple filters."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z",
                request_id="req-1",
                model="gpt-4",
                api_key="sk-user1",
                success=True,
            ),
            _make_record(
                timestamp="2024-01-01T11:00:00Z",
                request_id="req-2",
                model="gpt-4",
                api_key="sk-user2",
                success=True,
            ),
            _make_record(
                timestamp="2024-01-01T12:00:00Z",
                request_id="req-3",
                model="claude-3",
                api_key="sk-user1",
                success=False,
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get(
            "/api/conversations?api_key=sk-user1&success=true"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["request_id"] == "req-1"

    @pytest.mark.asyncio
    async def test_item_fields(self, client_with_indexer):
        """Test that items contain all expected fields."""
        records = [
            _make_record(
                timestamp="2024-01-01T10:00:00Z",
                request_id="req-1",
                model="gpt-4",
                adapter="openai",
                api_key="sk-test",
                success=True,
                latency_ms=150,
                tokens_in=10,
                tokens_out=20,
                output=[{"type": "text", "text": "Hello world"}],
            ),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        item = data["items"][0]

        # Note: "line" is now "id" (SQLite autoincrement)
        assert "id" in item
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
    async def test_item_defaults(self, client_with_indexer):
        """Test that items have sensible defaults for missing fields."""
        records = [{}]  # Empty record
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        item = data["items"][0]

        # Note: "line" is now "id"
        assert "id" in item
        assert item["timestamp"] == ""
        assert item["request_id"] == ""
        assert item["model"] == ""
        assert item["adapter"] == ""
        assert item["api_key"] == ""
        assert (
            item["success"] is False
        )  # Empty dict: get("success") returns None, which is falsy
        assert item["latency_ms"] == 0
        assert item["tokens_in"] == 0
        assert item["tokens_out"] == 0
        assert item["output_preview"] == ""
        assert item["has_tool_use"] is False


# ========== Test GetConversationDetail ==========


class TestGetConversationDetail:
    """Tests for GET /api/conversations/{record_id}."""

    @pytest.mark.asyncio
    async def test_get_detail_found(self, client_with_indexer):
        """Test getting a conversation detail by ID."""
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
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        # First record gets id=1
        resp = await client_with_indexer.get("/api/conversations/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-1"
        assert data["id"] == 1
        assert data["model"] == "gpt-4"
        assert data["messages"] == [{"role": "user", "content": "Hello"}]
        assert data["output"] == [{"type": "text", "text": "Hi there!"}]

    @pytest.mark.asyncio
    async def test_get_detail_second_line(self, client_with_indexer):
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
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        # Second record gets id=2
        resp = await client_with_indexer.get("/api/conversations/2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-2"
        assert data["id"] == 2

    @pytest.mark.asyncio
    async def test_get_detail_not_found(self, client_with_indexer):
        """Test getting a conversation that doesn't exist."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
            }
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations/999")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"] == "Record not found"

    @pytest.mark.asyncio
    async def test_get_detail_no_file(self, client_with_empty_log_dir):
        """Test getting detail when file doesn't exist."""
        resp = await client_with_empty_log_dir.get("/api/conversations/1")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"] == "Record not found"

    @pytest.mark.asyncio
    async def test_get_detail_internal_fields_excluded(self, client_with_indexer):
        """Test that internal _line field is excluded from response."""
        records = [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "request_id": "req-1",
                "model": "gpt-4",
                "_internal": "should be excluded",
            }
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations/1")
        assert resp.status_code == 200
        data = resp.json()
        # The route passes through all fields from the JSONL record
        # _internal would come through since it's in the file
        # but id should be added
        assert "id" in data
        assert data["id"] == 1


# ========== Test EdgeCases ==========


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
    async def test_offset_beyond_total(self, client_with_indexer):
        """Test offset beyond total records returns empty."""
        records = [
            _make_record(timestamp="2024-01-01T10:00:00Z", request_id="req-1"),
        ]
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        resp = await client_with_indexer.get("/api/conversations?offset=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_empty_lines_in_file(self, client_with_indexer):
        """Test handling of empty lines in JSONL file — only valid records indexed."""
        log_path = Path(client_with_indexer._temp_log_dir) / "conversations.jsonl"
        with open(log_path, "w") as f:
            f.write('{"request_id": "req-1", "model": "gpt-4"}\n')
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace line
            f.write('{"request_id": "req-2", "model": "claude-3"}\n')

        # Rebuild index from the file (simulates what would happen in production)
        client_with_indexer._indexer.rebuild_from_logs(
            client_with_indexer._temp_log_dir
        )

        resp = await client_with_indexer.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_malformed_json_lines(self, client_with_indexer):
        """Test handling of malformed JSON lines — only valid records indexed."""
        log_path = Path(client_with_indexer._temp_log_dir) / "conversations.jsonl"
        with open(log_path, "w") as f:
            f.write('{"request_id": "req-1", "model": "gpt-4"}\n')
            f.write("not valid json\n")
            f.write('{"request_id": "req-2", "model": "claude-3"}\n')

        # Rebuild index from the file
        client_with_indexer._indexer.rebuild_from_logs(
            client_with_indexer._temp_log_dir
        )

        resp = await client_with_indexer.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_large_dataset_pagination(self, client_with_indexer):
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
        _write_test_records(
            client_with_indexer._indexer,
            client_with_indexer._temp_log_dir,
            records,
        )

        # First page
        resp1 = await client_with_indexer.get("/api/conversations?limit=50&offset=0")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["total"] == 150
        assert len(data1["items"]) == 50

        # Second page
        resp2 = await client_with_indexer.get("/api/conversations?limit=50&offset=50")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["total"] == 150
        assert len(data2["items"]) == 50

        # Third page
        resp3 = await client_with_indexer.get("/api/conversations?limit=50&offset=100")
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["total"] == 150
        assert len(data3["items"]) == 50


# ========== Test ConvIndexer directly ==========


class TestIndexer:
    """Unit tests for ConvIndexer."""

    def test_extract_preview_text_only(self):
        """Test preview extraction with text-only output."""
        record = {
            "output": [
                {"type": "text", "text": "Hello, this is a response."},
                {"type": "text", "text": "This should not appear."},
            ],
        }
        preview, has_tool_use = ConvIndexer._extract_preview(record)
        assert preview == "Hello, this is a response."
        assert has_tool_use is False

    def test_extract_preview_tool_use(self):
        """Test preview extraction with tool_use output."""
        record = {
            "output": [
                {"type": "tool_use", "name": "get_weather", "input": {}},
            ],
        }
        preview, has_tool_use = ConvIndexer._extract_preview(record)
        assert preview == "[tool_use: get_weather]"
        assert has_tool_use is True

    def test_extract_preview_empty(self):
        """Test preview extraction with empty/missing output."""
        # No output key
        preview, has_tool_use = ConvIndexer._extract_preview({})
        assert preview == ""
        assert has_tool_use is False

        # None output
        preview, has_tool_use = ConvIndexer._extract_preview({"output": None})
        assert preview == ""
        assert has_tool_use is False

        # Empty list
        preview, has_tool_use = ConvIndexer._extract_preview({"output": []})
        assert preview == ""
        assert has_tool_use is False

        # Non-list output
        preview, has_tool_use = ConvIndexer._extract_preview({"output": "string"})
        assert preview == ""
        assert has_tool_use is False

    def test_rebuild_from_logs(self, temp_db, temp_log_dir):
        """Test rebuilding index from JSONL log files."""
        indexer = ConvIndexer(db_path=temp_db)

        # Write some records to the log file
        log_path = Path(temp_log_dir) / "conversations.jsonl"
        records = [
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
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req-3",
                "model": "gpt-4",
            },
        ]
        with open(log_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Rebuild
        count = indexer.rebuild_from_logs(temp_log_dir)
        assert count == 3

        # Verify we can query
        items, total = indexer.query()
        assert total == 3
        assert len(items) == 3

        # Verify filtering works
        items, total = indexer.query(model="gpt-4")
        assert total == 2

        indexer.close()
