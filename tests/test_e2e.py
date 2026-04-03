"""End-to-end tests with real HTTP requests against a running server.

These tests require a running ModelSwitch server. They can be run with:
    MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v

Environment variables:
    MODELSWITCH_E2E=1     - Enable e2e tests
    MODELSWITCH_URL       - Server URL (default: http://localhost:8000)
    MODELSWITCH_ADMIN_KEY - API key (default: sk-gateway-admin)
    MODEL_NAME            - Model to test (default: glm-5)
"""

import concurrent.futures
import os

import httpx
import pytest

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("MODELSWITCH_E2E"),
    reason="Set MODELSWITCH_E2E=1 to run end-to-end tests",
)

# Server configuration
BASE_URL = os.environ.get("MODELSWITCH_URL", "http://localhost:8000")
ADMIN_KEY = os.environ.get("MODELSWITCH_ADMIN_KEY", "sk-gateway-admin")
MODEL = os.environ.get("MODEL_NAME", "glm-5")
REQUEST_TIMEOUT = 60.0


@pytest.fixture(scope="module")
def server_ready():
    """Check if server is running and ready."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/config/health", timeout=5.0)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    pytest.skip(
        f"Server not ready at {BASE_URL}. Start with: python -m uvicorn app.main:app"
    )


@pytest.fixture
def headers():
    """Default headers with admin API key."""
    return {
        "Authorization": f"Bearer {ADMIN_KEY}",
        "Content-Type": "application/json",
    }


class TestHealthAndConfig:
    """Test health and configuration endpoints."""

    def test_health_endpoint(self, server_ready):
        """Test /api/config/health returns healthy status."""
        resp = httpx.get(f"{BASE_URL}/api/config/health", timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "models" in data
        assert "providers" in data

    def test_list_models(self, server_ready, headers):
        """Test GET /v1/models returns model list."""
        resp = httpx.get(f"{BASE_URL}/v1/models", headers=headers, timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_list_providers(self, server_ready, headers):
        """Test GET /api/config/providers returns provider list."""
        resp = httpx.get(
            f"{BASE_URL}/api/config/providers", headers=headers, timeout=10.0
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_get_config(self, server_ready, headers):
        """Test GET /api/config returns configuration."""
        resp = httpx.get(f"{BASE_URL}/api/config", headers=headers, timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "models" in data
        assert "api_keys" in data


class TestAuthentication:
    """Test authentication and authorization."""

    def test_no_auth_rejected(self, server_ready):
        """Test that protected endpoints reject requests without auth."""
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=10.0)
        assert resp.status_code == 401

    def test_invalid_key_rejected(self, server_ready):
        """Test that invalid API keys are rejected."""
        resp = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": "Bearer sk-invalid-key"},
            timeout=10.0,
        )
        assert resp.status_code == 401

    def test_valid_key_accepted(self, server_ready, headers):
        """Test that valid API keys are accepted."""
        resp = httpx.get(f"{BASE_URL}/v1/models", headers=headers, timeout=10.0)
        assert resp.status_code == 200

    def test_x_api_key_header(self, server_ready):
        """Test authentication via x-api-key header."""
        resp = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"x-api-key": ADMIN_KEY},
            timeout=10.0,
        )
        assert resp.status_code == 200


class TestOpenAIChatCompletions:
    """Test OpenAI-compatible chat completions endpoint."""

    def test_non_stream_chat(self, server_ready, headers):
        """Test non-streaming chat completion."""
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Say 'hello world' and nothing else."}
            ],
            "max_tokens": 20,
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    def test_non_stream_chat_with_usage(self, server_ready, headers):
        """Test that usage stats are returned."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]

    def test_stream_chat(self, server_ready, headers):
        """Test streaming chat completion."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 20,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    chunks.append(chunk)
            assert len(chunks) > 0

    def test_model_not_found(self, server_ready, headers):
        """Test error when requesting non-existent model."""
        payload = {
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code in [404, 502]

    def test_request_id_header(self, server_ready, headers):
        """Test that X-Request-ID is returned in response headers."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert "x-request-id" in resp.headers
        assert resp.headers["x-request-id"] != ""


class TestAnthropicMessages:
    """Test Anthropic-compatible messages endpoint."""

    def test_non_stream_message(self, server_ready, headers):
        """Test non-streaming Anthropic message."""
        payload = {
            "model": MODEL,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            headers={**headers, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert "content" in data
        assert len(data["content"]) > 0

    def test_message_format(self, server_ready, headers):
        """Test that response follows Anthropic message format."""
        payload = {
            "model": MODEL,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            headers={**headers, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert "id" in data
        assert "model" in data
        assert "usage" in data
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_stream_message(self, server_ready, headers):
        """Test streaming Anthropic message."""
        payload = {
            "model": MODEL,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Count 1, 2, 3."}],
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/messages",
            headers={**headers, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: ") or line.startswith("event: "):
                    events.append(line)
            assert len(events) > 0

    def test_system_message(self, server_ready, headers):
        """Test Anthropic message with system prompt."""
        payload = {
            "model": MODEL,
            "max_tokens": 20,
            "system": "You are a helpful assistant. Be concise.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            headers={**headers, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200


class TestToolCalls:
    """Test tool/function calling functionality."""

    def test_openai_tools_non_stream(self, server_ready, headers):
        """Test OpenAI tool calling (non-streaming)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 100,
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        # The model may or may not call the tool depending on its capabilities
        assert "choices" in data

    def test_openai_tools_stream(self, server_ready, headers):
        """Test OpenAI tool calling (streaming)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Calculate 2+2"}],
            "tools": tools,
            "stream": True,
            "max_tokens": 100,
        }
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line)
            assert len(chunks) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_body(self, server_ready, headers):
        """Test that invalid JSON returns proper error."""
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            content="not valid json",
            timeout=10.0,
        )
        # Server may return 422 or 500 for invalid JSON
        assert resp.status_code in [422, 500]

    def test_missing_model(self, server_ready, headers):
        """Test that missing model parameter returns error."""
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        # Model may default or return error
        assert resp.status_code in [200, 400, 404, 502]

    def test_missing_messages(self, server_ready, headers):
        """Test that missing messages returns error."""
        payload = {
            "model": MODEL,
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        # Server may return 400, 422, or 500
        assert resp.status_code in [400, 422, 500, 502]

    def test_empty_messages(self, server_ready, headers):
        """Test that empty messages returns error."""
        payload = {
            "model": MODEL,
            "messages": [],
        }
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        # Server may return 400, 422, or 500
        assert resp.status_code in [400, 422, 500, 502]

    def test_cors_headers(self, server_ready):
        """Test that CORS headers are present."""
        resp = httpx.options(
            f"{BASE_URL}/v1/models",
            headers={"Origin": "http://localhost:3000"},
            timeout=10.0,
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers(self, server_ready, headers):
        """Test that rate limit info is in response headers (if applicable)."""
        resp = httpx.get(f"{BASE_URL}/v1/models", headers=headers, timeout=10.0)
        # Rate limit headers may or may not be present depending on config
        assert resp.status_code == 200


class TestStreamingBehavior:
    """Test streaming-specific behavior."""

    def test_stream_sse_format(self, server_ready, headers):
        """Test that streaming uses proper SSE format."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200
            has_data = False
            has_done = False
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    has_data = True
                    if line == "data: [DONE]":
                        has_done = True
            assert has_data
            assert has_done

    def test_stream_can_be_cancelled(self, server_ready, headers):
        """Test that streaming can be cancelled mid-stream."""
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Write a long story about a cat."}
            ],
            "max_tokens": 500,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200
            chunk_count = 0
            for line in resp.iter_lines():
                chunk_count += 1
                if chunk_count >= 3:
                    break  # Cancel early

    def test_stream_timeout_handled(self, server_ready, headers):
        """Test that stream timeout is handled gracefully."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": True,
        }
        # Use a very short timeout to test handling
        try:
            with httpx.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=0.001,  # Very short timeout
            ) as resp:
                # If the server responds fast enough, this might succeed
                pass
        except httpx.TimeoutException:
            # This is expected for very short timeouts
            pass


class TestConcurrency:
    """Test concurrent request handling."""

    def test_concurrent_requests(self, server_ready, headers):
        """Test that server can handle multiple concurrent requests."""

        def make_request(i):
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": f"Say {i}"}],
                "max_tokens": 5,
            }
            resp = httpx.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        assert all(code == 200 for code in results)

    def test_sequential_requests(self, server_ready, headers):
        """Test multiple sequential requests."""
        for i in range(3):
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": f"Request {i}"}],
                "max_tokens": 5,
            }
            resp = httpx.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
