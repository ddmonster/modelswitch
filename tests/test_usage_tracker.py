"""Unit tests for UsageTracker with in-memory SQLite."""
import pytest
import pytest_asyncio

from app.services.usage_tracker import UsageTracker


@pytest_asyncio.fixture
async def tracker():
    """Provide an in-memory UsageTracker with shared connection."""
    import aiosqlite
    db = await aiosqlite.connect(":memory:")
    t = UsageTracker(db_path=":memory:", flush_interval=9999)
    t._shared_db = db
    await t.init()
    yield t
    await t.close()


class TestInit:
    @pytest.mark.asyncio
    async def test_creates_tables(self, tracker):
        await tracker.init()
        # Verify by recording and querying
        await tracker.record(provider="p1", model="m1")
        await tracker.flush()
        result = await tracker.aggregate("provider")
        assert result["total"] == 1


class TestRecordAndFlush:
    @pytest.mark.asyncio
    async def test_record_and_flush_single(self, tracker):
        await tracker.init()
        await tracker.record(
            provider="dashscope", model="glm5", api_key_alias="admin",
            api_key_masked="sk-gat***", success=True,
            tokens_in=10, tokens_out=20, latency_ms=500.0, status_code=200,
        )
        assert len(tracker._pending_records) == 1
        await tracker.flush()
        assert len(tracker._pending_records) == 0

    @pytest.mark.asyncio
    async def test_flush_empty_is_noop(self, tracker):
        await tracker.init()
        await tracker.flush()  # should not raise

    @pytest.mark.asyncio
    async def test_flush_readds_on_failure(self, tracker):
        """If DB write fails, records should be preserved for retry."""
        await tracker.record(provider="p", model="m")
        # Close the shared DB connection to force a failure on next flush
        await tracker._shared_db.close()
        tracker._shared_db = None
        # Now flush should fail and re-add records
        await tracker.flush()
        assert len(tracker._pending_records) == 1


class TestAggregate:
    @pytest.mark.asyncio
    async def test_group_by_provider(self, tracker):
        await tracker.init()
        await tracker.record(provider="dashscope", model="glm5", success=True, tokens_in=10, tokens_out=20, latency_ms=100)
        await tracker.record(provider="dashscope", model="glm5", success=True, tokens_in=5, tokens_out=15, latency_ms=200)
        await tracker.record(provider="glm-plan", model="glm5", success=False, tokens_in=8, tokens_out=0, latency_ms=500, status_code=502)
        await tracker.flush()

        result = await tracker.aggregate("provider")
        assert result["group_by"] == "provider"
        assert result["total"] == 3
        assert len(result["groups"]) == 2

        ds = next(g for g in result["groups"] if g["name"] == "dashscope")
        assert ds["total_requests"] == 2
        assert ds["success_count"] == 2
        assert ds["fail_count"] == 0
        assert ds["tokens_in"] == 15

        gp = next(g for g in result["groups"] if g["name"] == "glm-plan")
        assert gp["total_requests"] == 1
        assert gp["fail_count"] == 1

    @pytest.mark.asyncio
    async def test_group_by_model(self, tracker):
        await tracker.init()
        await tracker.record(provider="p", model="glm5", success=True)
        await tracker.record(provider="p", model="gpt4o", success=True)
        await tracker.flush()

        result = await tracker.aggregate("model")
        assert len(result["groups"]) == 2

    @pytest.mark.asyncio
    async def test_group_by_api_key(self, tracker):
        await tracker.init()
        await tracker.record(provider="p", model="m", api_key_alias="alice", success=True)
        await tracker.record(provider="p", model="m", api_key_alias="bob", success=True)
        await tracker.flush()

        result = await tracker.aggregate("api_key")
        assert len(result["groups"]) == 2

    @pytest.mark.asyncio
    async def test_empty_result(self, tracker):
        await tracker.init()
        result = await tracker.aggregate("provider")
        assert result["total"] == 0
        assert result["groups"] == []

    @pytest.mark.asyncio
    async def test_date_filter(self, tracker):
        await tracker.init()
        await tracker.record(provider="p", model="m")
        await tracker.flush()

        # Today's data should be found
        from datetime import date
        today = date.today().isoformat()
        result = await tracker.aggregate("provider", date_from=today, date_to=today)
        assert result["total"] == 1

        # Far future date should find nothing
        result = await tracker.aggregate("provider", date_from="2099-01-01", date_to="2099-12-31")
        assert result["total"] == 0


class TestGetDetail:
    @pytest.mark.asyncio
    async def test_drill_down(self, tracker):
        await tracker.init()
        await tracker.record(provider="dashscope", model="glm5", api_key_alias="alice")
        await tracker.record(provider="dashscope", model="gpt4o", api_key_alias="alice")
        await tracker.record(provider="glm-plan", model="glm5", api_key_alias="bob")
        await tracker.flush()

        # Drill into dashscope -> group by model
        details = await tracker.get_detail("provider", "dashscope", "model")
        assert len(details) == 2
        names = {d["name"] for d in details}
        assert "glm5" in names
        assert "gpt4o" in names

    @pytest.mark.asyncio
    async def test_drill_down_by_api_key(self, tracker):
        await tracker.init()
        await tracker.record(provider="dashscope", model="glm5", api_key_alias="alice")
        await tracker.record(provider="dashscope", model="glm5", api_key_alias="bob")
        await tracker.flush()

        details = await tracker.get_detail("provider", "dashscope", "api_key")
        assert len(details) == 2
        names = {d["name"] for d in details}
        assert "alice" in names
        assert "bob" in names


class TestClose:
    @pytest.mark.asyncio
    async def test_close_flushes_pending(self, tracker):
        await tracker.init()
        await tracker.record(provider="p", model="m")
        assert len(tracker._pending_records) == 1
        await tracker.close()
        assert len(tracker._pending_records) == 0
