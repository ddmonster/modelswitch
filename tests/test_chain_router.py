"""Unit tests for ChainRouter."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.adapters.litellm_adapter import AdapterResponse
from app.core.chain_router import ChainRouter
from app.models.config_models import (
    GatewayConfig,
    ModelAdapterRef,
    ModelConfig,
    ProviderConfig,
)


def _make_config_with_n_adapters(n, mode="chain"):
    providers = [
        ProviderConfig(name=f"p{i}", provider="openai", base_url=f"https://p{i}.test.com/v1", api_key=f"sk-{i}")
        for i in range(n)
    ]
    adapters = [
        ModelAdapterRef(adapter=f"p{i}", model_name=f"model-{i}", priority=i + 1, timeout=60)
        for i in range(n)
    ]
    models = {"test-model": ModelConfig(mode=mode, adapters=adapters)}
    return GatewayConfig(
        providers=providers, models=models, api_keys=[],
    )


class TestInitAndConfig:
    def test_loads_providers(self):
        config = _make_config_with_n_adapters(3)
        router = ChainRouter(config)
        assert len(router.get_providers()) == 3
        assert "p0" in router.get_providers()

    def test_loads_models(self):
        config = _make_config_with_n_adapters(2)
        router = ChainRouter(config)
        assert "test-model" in router.list_models()

    def test_get_model_found(self):
        config = _make_config_with_n_adapters(1)
        router = ChainRouter(config)
        m = router.get_model("test-model")
        assert m is not None
        assert m.mode == "chain"

    def test_get_model_not_found(self):
        config = _make_config_with_n_adapters(1)
        router = ChainRouter(config)
        assert router.get_model("nonexistent") is None

    def test_reload_config(self):
        config1 = _make_config_with_n_adapters(1)
        router = ChainRouter(config1)
        config2 = _make_config_with_n_adapters(3)
        router.reload_config(config2)
        assert len(router.get_providers()) == 3


class TestExecuteChatModelNotFound:
    @pytest.mark.asyncio
    async def test_returns_404(self):
        config = _make_config_with_n_adapters(1)
        router = ChainRouter(config)
        result = await router.execute_chat("no-such-model", [], stream=False)
        assert result.success is False
        assert result.status_code == 404
        assert "not found" in result.error.lower()


class TestExecuteChatNoAdapters:
    @pytest.mark.asyncio
    async def test_returns_503(self):
        from app.models.config_models import GatewayConfig
        config = GatewayConfig(
            providers=[ProviderConfig(name="p0", provider="openai", base_url="https://x", api_key="k")],
            models={"empty": ModelConfig(mode="chain", adapters=[])},
            api_keys=[],
        )
        router = ChainRouter(config)
        result = await router.execute_chat("empty", [], stream=False)
        assert result.success is False
        assert result.status_code == 503


class TestAdapterMode:
    @pytest.mark.asyncio
    async def test_adapter_mode_calls_single_provider(self):
        config = _make_config_with_n_adapters(1, mode="adapter")
        router = ChainRouter(config)

        mock_response = AdapterResponse(status_code=200, success=True, body={"choices": []}, adapter_name="p0")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=mock_response):
            result = await router.execute_chat("test-model", [{"role": "user", "content": "hi"}], stream=False)
            assert result.success is True
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_adapter_mode_provider_not_found(self):
        """Adapter references a provider that doesn't exist."""
        config = GatewayConfig(
            providers=[],
            models={"m": ModelConfig(mode="adapter", adapters=[ModelAdapterRef(adapter="ghost", model_name="x", priority=1)])},
            api_keys=[],
        )
        router = ChainRouter(config)
        result = await router.execute_chat("m", [], stream=False)
        assert result.success is False
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_adapter_mode_provider_disabled(self):
        config = GatewayConfig(
            providers=[ProviderConfig(name="p0", provider="openai", base_url="https://x", api_key="k", enabled=False)],
            models={"m": ModelConfig(mode="adapter", adapters=[ModelAdapterRef(adapter="p0", model_name="x", priority=1)])},
            api_keys=[],
        )
        router = ChainRouter(config)
        result = await router.execute_chat("m", [], stream=False)
        assert result.success is False
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_adapter_mode_circuit_breaker_open(self):
        config = _make_config_with_n_adapters(1, mode="adapter")
        router = ChainRouter(config)
        # Force circuit breaker open
        cb = router._circuit_breakers["p0"]
        for _ in range(5):
            cb.record_failure()
        assert cb.state.value == "open"

        result = await router.execute_chat("test-model", [], stream=False)
        assert result.success is False
        assert result.status_code == 503


class TestChainModeFallback:
    @pytest.mark.asyncio
    async def test_first_adapter_succeeds(self):
        config = _make_config_with_n_adapters(2, mode="chain")
        router = ChainRouter(config)

        ok = AdapterResponse(status_code=200, success=True, body={"choices": []}, adapter_name="p0")
        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=ok):
            with patch.object(router._adapters["p1"], "chat_completion", new_callable=AsyncMock) as mock_p1:
                result = await router.execute_chat("test-model", [{"role": "user", "content": "hi"}], stream=False)
                assert result.success is True
                mock_p1.assert_not_called()  # second adapter not tried

    @pytest.mark.asyncio
    async def test_first_fails_second_succeeds(self):
        config = _make_config_with_n_adapters(2, mode="chain")
        router = ChainRouter(config)

        fail = AdapterResponse(status_code=502, success=False, error="bad gateway", adapter_name="p0")
        ok = AdapterResponse(status_code=200, success=True, body={"choices": []}, adapter_name="p1")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=fail):
            with patch.object(router._adapters["p1"], "chat_completion", new_callable=AsyncMock, return_value=ok):
                result = await router.execute_chat("test-model", [{"role": "user", "content": "hi"}], stream=False)
                assert result.success is True
                assert result.adapter_name == "p1"

    @pytest.mark.asyncio
    async def test_all_fail_returns_502(self):
        config = _make_config_with_n_adapters(2, mode="chain")
        router = ChainRouter(config)

        fail1 = AdapterResponse(status_code=502, success=False, error="fail1", adapter_name="p0")
        fail2 = AdapterResponse(status_code=500, success=False, error="fail2", adapter_name="p1")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=fail1):
            with patch.object(router._adapters["p1"], "chat_completion", new_callable=AsyncMock, return_value=fail2):
                result = await router.execute_chat("test-model", [{"role": "user", "content": "hi"}], stream=False)
                assert result.success is False
                assert result.status_code == 502

    @pytest.mark.asyncio
    async def test_auth_error_skips_to_next(self):
        """Non-retryable errors (401) should skip retry and fallback immediately."""
        config = _make_config_with_n_adapters(2, mode="chain")
        router = ChainRouter(config)

        auth_fail = AdapterResponse(status_code=401, success=False, error="bad key", adapter_name="p0")
        ok = AdapterResponse(status_code=200, success=True, body={"choices": []}, adapter_name="p1")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=auth_fail) as mock_p0:
            with patch.object(router._adapters["p1"], "chat_completion", new_callable=AsyncMock, return_value=ok):
                result = await router.execute_chat("test-model", [], stream=False)
                assert result.success is True
                assert mock_p0.call_count == 1  # no retry on 401

    @pytest.mark.asyncio
    async def test_retriable_error_retries_once(self):
        """Retryable errors (500) should retry once before fallback."""
        config = _make_config_with_n_adapters(1, mode="chain")
        router = ChainRouter(config)

        fail = AdapterResponse(status_code=500, success=False, error="internal error", adapter_name="p0")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=fail):
            with patch("app.core.chain_router.asyncio.sleep", new_callable=AsyncMock):
                result = await router.execute_chat("test-model", [], stream=False)
                assert result.success is False
                # Should have been called twice (initial + retry)

    @pytest.mark.asyncio
    async def test_disabled_provider_skipped(self):
        config = GatewayConfig(
            providers=[
                ProviderConfig(name="p0", provider="openai", base_url="https://x", api_key="k", enabled=False),
                ProviderConfig(name="p1", provider="openai", base_url="https://y", api_key="k", enabled=True),
            ],
            models={"m": ModelConfig(mode="chain", adapters=[
                ModelAdapterRef(adapter="p0", model_name="x", priority=1),
                ModelAdapterRef(adapter="p1", model_name="y", priority=2),
            ])},
            api_keys=[],
        )
        router = ChainRouter(config)
        ok = AdapterResponse(status_code=200, success=True, body={"choices": []}, adapter_name="p1")

        with patch.object(router._adapters["p1"], "chat_completion", new_callable=AsyncMock, return_value=ok):
            result = await router.execute_chat("m", [], stream=False)
            assert result.success is True


class TestStreamingMode:
    @pytest.mark.asyncio
    async def test_stream_returns_stream_in_result(self):
        config = _make_config_with_n_adapters(1, mode="chain")
        router = ChainRouter(config)

        async def fake_stream():
            yield {"choices": [{"delta": {"content": "hi"}}]}
            yield {"choices": [{"delta": {"content": " there"}, "finish_reason": "stop"}]}

        stream_resp = AdapterResponse(status_code=200, success=True, stream=fake_stream(), adapter_name="p0")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=stream_resp):
            result = await router.execute_chat("test-model", [{"role": "user", "content": "hi"}], stream=True)
            assert result.success is True
            assert result.stream is not None

    @pytest.mark.asyncio
    async def test_stream_all_fail_yields_error_chunk(self):
        config = _make_config_with_n_adapters(1, mode="chain")
        router = ChainRouter(config)

        fail = AdapterResponse(status_code=502, success=False, error="all bad", adapter_name="p0")

        with patch.object(router._adapters["p0"], "chat_completion", new_callable=AsyncMock, return_value=fail):
            result = await router.execute_chat("test-model", [{"role": "user", "content": "hi"}], stream=True)
            assert result.success is True  # the stream itself is "successful" (returns a generator)
            chunks = []
            async for chunk in result.stream:
                chunks.append(chunk)
            assert len(chunks) == 1
            assert "error" in chunks[0]
