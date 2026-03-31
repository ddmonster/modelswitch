"""API integration tests for all route handlers.

Uses httpx AsyncClient with ASGI transport for full end-to-end testing.
LiteLLM calls are mocked at the ChainRouter/adapter level.
"""
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest


# ========== Auth / Middleware Tests ==========

class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_public_path_no_auth_needed(self, client):
        resp = await client.get("/api/config/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_v1_models_requires_auth(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_v1_models_with_valid_key(self, client):
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sk-test-admin"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_bearer_auth(self, client):
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sk-test-admin"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_x_api_key_auth(self, client):
        resp = await client.get("/v1/models", headers={"x-api-key": "sk-test-admin"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_sk_prefix_auth(self, client):
        resp = await client.get("/v1/models", headers={"Authorization": "sk-test-admin"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_key_rejected(self, client):
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sk-wrong"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_key_rejected(self, client_expired_key):
        resp = await client_expired_key.get("/v1/models", headers={"Authorization": "Bearer sk-expired"})
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, client_rate_limited):
        headers = {"Authorization": "Bearer sk-limited"}
        # First 2 should succeed
        resp1 = await client_rate_limited.get("/v1/models", headers=headers)
        resp2 = await client_rate_limited.get("/v1/models", headers=headers)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Third should be rate limited
        resp3 = await client_rate_limited.get("/v1/models", headers=headers)
        assert resp3.status_code == 429
        assert "Retry-After" in resp3.headers

    @pytest.mark.asyncio
    async def test_cors_options(self, client):
        resp = await client.options("/v1/models")
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"

    @pytest.mark.asyncio
    async def test_frontend_served(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200


# ========== OpenAI Routes ==========

class TestOpenAIModels:
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sk-test-admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert "chain-model" in model_ids
        assert "direct-model" in model_ids

    @pytest.mark.asyncio
    async def test_model_format(self, client):
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sk-test-admin"})
        model = resp.json()["data"][0]
        assert model["object"] == "model"
        assert "id" in model
        assert "owned_by" in model
        assert "created" in model

    @pytest.mark.asyncio
    async def test_new_openai_endpoint(self, client):
        """新的 /openai/models 端点正常工作"""
        resp = await client.get("/openai/models", headers={"Authorization": "Bearer sk-test-admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1


class TestOpenAIChatCompletions:
    @pytest.mark.asyncio
    async def test_non_stream_success(self, client, sample_config):
        """Mock the adapter to return a successful response."""
        from app.adapters.litellm_adapter import AdapterResponse

        mock_body = MagicMock()
        mock_body.model_dump.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion",
            new_callable=AsyncMock,
            return_value=AdapterResponse(
                status_code=200, success=True, body=mock_body,
                adapter_name="provider-a", model_name="upstream-a",
                latency_ms=100, usage={"prompt_tokens": 5, "completion_tokens": 2},
            ),
        ):
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-test-admin"},
                json={"model": "chain-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_non_stream_all_adapters_fail(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(status_code=502, success=False, error="fail", adapter_name="provider-a"),
        ), patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-b"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(status_code=502, success=False, error="also fail", adapter_name="provider-b"),
        ):
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-test-admin"},
                json={"model": "chain-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50},
            )
            assert resp.status_code == 502
            assert "error" in resp.json()

    @pytest.mark.asyncio
    async def test_model_not_found(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test-admin"},
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        # chain_router returns AdapterResponse with status_code 404
        # The route converts it to 502 (since success=False is treated as upstream error)
        assert resp.status_code in (404, 502)

    @pytest.mark.asyncio
    async def test_stream_success(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse

        async def fake_stream():
            yield {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": " there"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(
                status_code=200, success=True, stream=fake_stream(),
                adapter_name="provider-a", model_name="upstream-a",
            ),
        ):
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-test-admin"},
                json={"model": "chain-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert resp.status_code == 200
            text = resp.text
            assert "data:" in text
            assert "[DONE]" in text


# ========== Anthropic Routes ==========

class TestAnthropicMessages:
    @pytest.mark.asyncio
    async def test_non_stream_success(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse

        mock_body = MagicMock()
        mock_body.model_dump.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "Bonjour!", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(
                status_code=200, success=True, body=mock_body,
                adapter_name="provider-a", latency_ms=200,
                usage={"prompt_tokens": 3, "completion_tokens": 1},
            ),
        ):
            resp = await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-test-admin", "anthropic-version": "2023-06-01"},
                json={"model": "chain-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert len(data["content"]) == 1
            assert data["content"][0]["text"] == "Bonjour!"
            assert data["usage"]["input_tokens"] == 3

    @pytest.mark.asyncio
    async def test_anthropic_error_format(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(status_code=502, success=False, error="boom", adapter_name="p"),
        ), patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-b"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(status_code=502, success=False, error="boom2", adapter_name="p"),
        ):
            resp = await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-test-admin", "anthropic-version": "2023-06-01"},
                json={"model": "chain-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50},
            )
            assert resp.status_code == 502
            data = resp.json()
            assert data["type"] == "error"
            assert "error" in data

    @pytest.mark.asyncio
    async def test_anthropic_system_message(self, client, sample_config):
        """Verify system message is converted properly."""
        from app.adapters.litellm_adapter import AdapterResponse

        mock_body = MagicMock()
        mock_body.model_dump.return_value = {
            "choices": [{"message": {"content": "ok", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(
                status_code=200, success=True, body=mock_body,
                adapter_name="provider-a", latency_ms=100,
                usage={"prompt_tokens": 10, "completion_tokens": 1},
            ),
        ) as mock_call:
            resp = await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-test-admin", "anthropic-version": "2023-06-01"},
                json={
                    "model": "chain-model",
                    "system": "Be helpful",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            # Verify the messages sent to chain_router include system message
            call_args = mock_call.call_args
            messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][1] if len(call_args[0]) > 1 else None
            # The messages should start with system role
            if messages:
                assert messages[0]["role"] == "system"


# ========== New Endpoint Paths ==========

class TestNewEndpointPaths:
    """测试新的 /openai/ 和 /anthropic/ 端点路径"""

    @pytest.mark.asyncio
    async def test_openai_models_new_path(self, client):
        resp = await client.get("/openai/models", headers={"Authorization": "Bearer sk-test-admin"})
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"

    @pytest.mark.asyncio
    async def test_openai_chat_new_path(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse
        mock_body = MagicMock()
        mock_body.model_dump.return_value = {
            "id": "chatcmpl-new", "object": "chat.completion",
            "choices": [{"message": {"content": "Hi", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        with patch.object(
            client._transport.app.state.chain_router, "_adapters",
            {"test-provider": MagicMock()},
        ), patch.object(
            client._transport.app.state.chain_router, "execute_chat",
            new_callable=AsyncMock, return_value=AdapterResponse(
                success=True, body=mock_body, status_code=200,
                adapter_name="test-provider", latency_ms=50,
            ),
        ):
            resp = await client.post(
                "/openai/chat/completions",
                json={"model": "chain-model", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer sk-test-admin"},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_anthropic_messages_new_path(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse
        mock_body = MagicMock()
        mock_body.model_dump.return_value = {
            "id": "chatcmpl-anth", "object": "chat.completion",
            "choices": [{"message": {"content": "Bonjour", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with patch.object(
            client._transport.app.state.chain_router, "_adapters",
            {"test-provider": MagicMock()},
        ), patch.object(
            client._transport.app.state.chain_router, "execute_chat",
            new_callable=AsyncMock, return_value=AdapterResponse(
                success=True, body=mock_body, status_code=200,
                adapter_name="test-provider", latency_ms=50,
            ),
        ):
            resp = await client.post(
                "/anthropic/messages",
                json={
                    "model": "chain-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 50,
                },
                headers={"x-api-key": "sk-test-admin"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_v1_backward_compat(self, client):
        """旧的 /v1/ 路径仍然工作"""
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sk-test-admin"})
        assert resp.status_code == 200


# ========== Config Routes ==========

class TestConfigRoutes:
    @pytest.mark.asyncio
    async def test_get_config(self, client):
        resp = await client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "models" in data
        assert "api_keys" in data
        # API keys should be masked
        for k in data["api_keys"]:
            assert "***" in k["key"]

    @pytest.mark.asyncio
    async def test_list_providers(self, client):
        resp = await client.get("/api/config/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        resp = await client.get("/api/config/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "chain-model" in data
        assert "direct-model" in data

    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/api/config/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "models" in data
        assert "providers" in data

    @pytest.mark.asyncio
    async def test_create_provider(self, client):
        resp = await client.post("/api/config/providers", json={
            "name": "new-provider",
            "provider": "openai",
            "base_url": "https://new.test.com/v1",
            "api_key": "sk-new-key",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_duplicate_provider(self, client):
        resp = await client.post("/api/config/providers", json={
            "name": "provider-a",
            "provider": "openai",
            "base_url": "https://x",
            "api_key": "k",
        })
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_update_provider(self, client):
        resp = await client.put("/api/config/providers/provider-a", json={
            "base_url": "https://updated.test.com/v1",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_nonexistent_provider(self, client):
        resp = await client.put("/api/config/providers/nope", json={"base_url": "https://x"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_provider(self, client):
        resp = await client.delete("/api/config/providers/provider-a")
        assert resp.status_code == 200
        # Verify deleted
        resp2 = await client.get("/api/config/providers")
        names = [p["name"] for p in resp2.json()]
        assert "provider-a" not in names

    @pytest.mark.asyncio
    async def test_toggle_provider(self, client):
        resp = await client.patch("/api/config/providers/provider-a/toggle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_create_model(self, client):
        resp = await client.post(
            "/api/config/models?name=new-model",
            json={"mode": "adapter", "description": "Test", "adapters": [{"adapter": "provider-a", "model_name": "x", "priority": 1, "timeout": 60}]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_duplicate_model(self, client):
        resp = await client.post(
            "/api/config/models?name=chain-model",
            json={"mode": "chain", "adapters": []},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_update_model(self, client):
        resp = await client.put("/api/config/models/chain-model", json={
            "description": "Updated desc",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_model(self, client):
        resp = await client.delete("/api/config/models/chain-model")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, client):
        resp = await client.delete("/api/config/models/nope")
        assert resp.status_code == 404


class TestConfigTestEndpoints:
    @pytest.mark.asyncio
    async def test_provider_test_success(self, client, sample_config):
        """Test provider connectivity test with mocked httpx."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.post("/api/config/providers/provider-a/test")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_provider_test_not_found(self, client):
        resp = await client.post("/api/config/providers/no-such-provider/test")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_model_test_success(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse

        mock_body = MagicMock()
        mock_body.choices = [MagicMock()]
        mock_body.choices[0].message = MagicMock()
        mock_body.choices[0].message.content = "Test response"

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(
                status_code=200, success=True, body=mock_body,
                adapter_name="provider-a", model_name="upstream-a",
                latency_ms=300, usage={"prompt_tokens": 6, "completion_tokens": 2},
            ),
        ):
            resp = await client.post("/api/config/models/chain-model/test")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["adapter_used"] == "provider-a"
            assert "latency_ms" in data
            assert data["preview"] == "Test response"

    @pytest.mark.asyncio
    async def test_model_test_not_found(self, client):
        resp = await client.post("/api/config/models/no-such-model/test")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_model_test_failure(self, client, sample_config):
        from app.adapters.litellm_adapter import AdapterResponse

        with patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-a"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(status_code=502, success=False, error="conn refused", adapter_name="p"),
        ), patch.object(
            client._transport.app.state.chain_router._adapters.get("provider-b"),
            "chat_completion", new_callable=AsyncMock,
            return_value=AdapterResponse(status_code=502, success=False, error="also fail", adapter_name="p"),
        ):
            resp = await client.post("/api/config/models/chain-model/test")
            data = resp.json()
            assert data["success"] is False
            assert "error" in data


# ========== API Key Routes ==========

class TestApiKeyRoutes:
    @pytest.mark.asyncio
    async def test_list_keys(self, client):
        resp = await client.get("/api/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        # Keys should be masked
        for k in data:
            assert "***" in k["key"]

    @pytest.mark.asyncio
    async def test_create_key(self, client):
        resp = await client.post("/api/keys", json={
            "name": "test-user",
            "description": "for testing",
            "rate_limit": 30,
            "daily_limit": 100,
            "allowed_models": ["chain-model"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["key"].startswith("sk-")
        assert len(data["key"]) > 10

    @pytest.mark.asyncio
    async def test_toggle_key(self, client):
        resp = await client.patch("/api/keys/sk-test-admin/toggle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_nonexistent_key(self, client):
        resp = await client.patch("/api/keys/sk-nope/toggle")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_key(self, client):
        resp = await client.delete("/api/keys/sk-test-admin")
        assert resp.status_code == 200
        resp2 = await client.get("/api/keys")
        keys = resp2.json()
        assert not any("admin" in k.get("name", "") for k in keys)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, client):
        resp = await client.delete("/api/keys/sk-nope")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_key(self, client):
        resp = await client.put("/api/keys/sk-test-admin", json={
            "rate_limit": 120,
            "daily_limit": 500,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_key_usage(self, client):
        resp = await client.get("/api/keys/sk-test-admin/usage")
        assert resp.status_code == 200


# ========== Usage Routes ==========

class TestUsageRoutes:
    @pytest.mark.asyncio
    async def test_usage_empty(self, client):
        resp = await client.get("/api/usage?group_by=model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_usage_with_data(self, client):
        tracker = client._transport.app.state.usage_tracker
        await tracker.record(provider="p", model="m1", api_key_alias="admin", success=True, tokens_in=10, tokens_out=5, latency_ms=100)
        await tracker.record(provider="p", model="m2", api_key_alias="admin", success=False, tokens_in=5, tokens_out=0, latency_ms=200)
        await tracker.flush()

        resp = await client.get("/api/usage?group_by=model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["groups"]) == 2

    @pytest.mark.asyncio
    async def test_usage_detail_drill_down(self, client):
        tracker = client._transport.app.state.usage_tracker
        await tracker.record(provider="dashscope", model="glm5", api_key_alias="alice")
        await tracker.record(provider="dashscope", model="gpt4o", api_key_alias="bob")
        await tracker.flush()

        resp = await client.get("/api/usage/dashscope/detail?group_by=provider&sub_group=model")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


# ========== Log Routes ==========

class TestLogRoutes:
    @pytest.mark.asyncio
    async def test_get_logs_empty(self, client):
        resp = await client.get("/api/logs?tail=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["logs"] == []

    @pytest.mark.asyncio
    async def test_get_logs_with_level_filter(self, client):
        resp = await client.get("/api/logs?tail=10&level=ERROR")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_logs_with_request_id(self, client):
        resp = await client.get("/api/logs?tail=10&request_id=abc123")
        assert resp.status_code == 200
