"""Comprehensive tests for OpenAIAdapter."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError, APITimeoutError

from app.adapters.openai_adapter import _OPENAI_STANDARD_PARAMS, OpenAIAdapter
from app.core.request_queue import reset_queue_manager
from app.models.config_models import ProviderConfig

# ========== Test Fixtures ==========


@pytest.fixture(autouse=True)
def reset_queue():
    """Reset queue manager before and after each test."""
    reset_queue_manager()
    yield
    reset_queue_manager()


@pytest.fixture
def provider_config():
    """Create a basic provider config for testing."""
    return ProviderConfig(
        name="test-openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
        max_concurrent=0,  # No queue by default
    )


@pytest.fixture
def provider_config_with_queue():
    """Create a provider config with queue enabled."""
    return ProviderConfig(
        name="test-openai-queued",
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
        max_concurrent=5,
        max_queue_size=10,
        queue_timeout=60.0,
    )


# ========== Mock Classes for OpenAI SDK ==========


@dataclass
class MockUsage:
    """Mock usage object."""

    prompt_tokens: int = 10
    completion_tokens: int = 20


@dataclass
class MockChoice:
    """Mock choice object."""

    index: int = 0
    finish_reason: str = "stop"
    message: Dict[str, Any] = field(
        default_factory=lambda: {"role": "assistant", "content": "Hello!"}
    )
    delta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockResponse:
    """Mock OpenAI chat completion response."""

    id: str = "chatcmpl-123"
    model: str = "gpt-4"
    choices: List[MockChoice] = field(default_factory=lambda: [MockChoice()])
    usage: Optional[MockUsage] = field(default_factory=MockUsage)


@dataclass
class MockStreamChunk:
    """Mock streaming chunk."""

    id: str = "chatcmpl-123"
    model: str = "gpt-4"
    choices: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[MockUsage] = None


# ========== Test OpenAIAdapter.__init__ ==========


class TestOpenAIAdapterInit:
    """Tests for OpenAIAdapter initialization."""

    def test_init_without_queue(self, provider_config):
        """Test initialization without queue (max_concurrent = 0)."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)

            mock_client_cls.assert_called_once_with(
                api_key="sk-test-key",
                base_url="https://api.openai.com/v1",
            )
            assert adapter.name == "test-openai"
            assert adapter.provider == provider_config
            assert adapter._use_queue is False

    def test_init_with_queue(self, provider_config_with_queue):
        """Test initialization with queue enabled (max_concurrent > 0)."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config_with_queue)

            mock_client_cls.assert_called_once_with(
                api_key="sk-test-key",
                base_url="https://api.openai.com/v1",
            )
            assert adapter.name == "test-openai-queued"
            assert adapter._use_queue is True

            # Verify queue was registered
            from app.core.request_queue import get_queue_manager

            queue_manager = get_queue_manager()
            queue = queue_manager.get_queue("test-openai-queued")
            assert queue is not None
            assert queue.max_concurrent == 5
            assert queue.max_queue_size == 10
            assert queue.queue_timeout == 60.0

    def test_init_with_custom_headers(self, provider_config):
        """Test initialization with custom headers."""
        provider_config.custom_headers = {"X-Custom-Header": "test-value"}

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            assert adapter.provider.custom_headers == {"X-Custom-Header": "test-value"}


# ========== Test OpenAIAdapter.chat_completion (Queue Routing) ==========


class TestChatCompletionQueueRouting:
    """Tests for queue routing in chat_completion."""

    @pytest.mark.asyncio
    async def test_uses_queue_when_enabled(self, provider_config_with_queue):
        """Test that queue is used when max_concurrent > 0."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MockResponse()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config_with_queue)

            # Start the queue
            from app.core.request_queue import get_queue_manager

            queue_manager = get_queue_manager()
            await queue_manager.start()

            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

            assert result.success is True
            assert result.status_code == 200

            await queue_manager.stop()

    @pytest.mark.asyncio
    async def test_direct_call_when_queue_disabled(self, provider_config):
        """Test that _do_chat_completion is called directly when queue disabled."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MockResponse()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)

            # Patch _do_chat_completion to track calls
            original_method = adapter._do_chat_completion
            call_count = [0]

            async def tracked_call(*args, **kwargs):
                call_count[0] += 1
                return await original_method(*args, **kwargs)

            adapter._do_chat_completion = tracked_call

            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

            assert call_count[0] == 1
            assert result.success is True


# ========== Test _do_chat_completion Non-Stream ==========


class TestDoChatCompletionNonStream:
    """Tests for non-streaming _do_chat_completion."""

    @pytest.mark.asyncio
    async def test_non_stream_success(self, provider_config):
        """Test successful non-stream response."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is True
        assert result.status_code == 200
        assert result.adapter_name == "test-openai"
        assert result.model_name == "gpt-4"
        assert result.body is not None
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_non_stream_without_usage(self, provider_config):
        """Test non-stream response without usage."""
        mock_response = MockResponse(usage=None)

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is True
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_non_stream_with_standard_params(self, provider_config):
        """Test non-stream with standard OpenAI params."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
            )

        assert result.success is True
        # Verify standard params were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9
        assert "extra_body" not in call_kwargs

    @pytest.mark.asyncio
    async def test_non_stream_with_extra_body(self, provider_config):
        """Test non-stream with non-standard params (extra_body)."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                top_k=50,  # Non-standard param
                custom_param="value",
            )

        assert result.success is True
        # Verify extra_body was created with non-standard params
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["top_k"] == 50
        assert call_kwargs["extra_body"]["custom_param"] == "value"

    @pytest.mark.asyncio
    async def test_non_stream_with_custom_headers(self, provider_config):
        """Test non-stream with custom headers."""
        provider_config.custom_headers = {"X-Custom": "test"}
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is True
        # Verify extra_headers was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "extra_headers" in call_kwargs
        assert call_kwargs["extra_headers"]["X-Custom"] == "test"

    @pytest.mark.asyncio
    async def test_non_stream_with_timeout_param(self, provider_config):
        """Test non-stream with custom timeout."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                timeout=120,
            )

        assert result.success is True
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 120

    @pytest.mark.asyncio
    async def test_non_stream_with_request_id(self, provider_config):
        """Test non-stream with request_id in logs."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                request_id="test-req-123",
            )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_non_stream_with_tools(self, provider_config):
        """Test non-stream with tools parameter."""
        mock_response = MockResponse()
        tools = [{"type": "function", "function": {"name": "test"}}]

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=tools,
            )

        assert result.success is True
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] == tools


# ========== Test _do_chat_completion Stream ==========


class TestDoChatCompletionStream:
    """Tests for streaming _do_chat_completion."""

    @pytest.mark.asyncio
    async def test_stream_success(self, provider_config):
        """Test successful stream response."""
        chunks = [
            MockStreamChunk(
                choices=[{"delta": {"content": "Hello"}}],
            ),
            MockStreamChunk(
                choices=[{"delta": {"content": " world"}}],
            ),
            MockStreamChunk(
                choices=[{"delta": {}, "finish_reason": "stop"}],
                usage=MockUsage(prompt_tokens=5, completion_tokens=10),
            ),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        assert result.status_code == 200
        assert result.stream is not None

        # Consume stream
        collected = []
        async for chunk in result.stream:
            collected.append(chunk)

        assert len(collected) == 3
        # Verify usage was captured
        assert result.usage == {"prompt_tokens": 5, "completion_tokens": 10}

    @pytest.mark.asyncio
    async def test_stream_auto_includes_stream_options(self, provider_config):
        """Test that stream_options is automatically added for streaming."""
        chunks = [MockStreamChunk(choices=[{"delta": {"content": "Hi"}}])]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        # Verify stream_options was added
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "stream_options" in call_kwargs
        assert call_kwargs["stream_options"]["include_usage"] is True

        # Consume stream to avoid warning
        async for _ in result.stream:
            pass

    @pytest.mark.asyncio
    async def test_stream_preserves_existing_stream_options(self, provider_config):
        """Test that existing stream_options is preserved."""
        chunks = [MockStreamChunk(choices=[{"delta": {"content": "Hi"}}])]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                stream_options={"include_usage": False},
            )

        # Verify stream_options was not overwritten
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream_options"]["include_usage"] is False

        # Consume stream
        async for _ in result.stream:
            pass

    @pytest.mark.asyncio
    async def test_stream_with_usage_zero_tokens(self, provider_config):
        """Test stream with usage containing zero tokens."""
        chunks = [
            MockStreamChunk(choices=[{"delta": {"content": "Hi"}}]),
            MockStreamChunk(
                choices=[{"finish_reason": "stop"}],
                usage=MockUsage(prompt_tokens=0, completion_tokens=0),
            ),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        async for _ in result.stream:
            pass

        # Verify usage with zeros is captured
        assert result.usage == {"prompt_tokens": 0, "completion_tokens": 0}

    @pytest.mark.asyncio
    async def test_stream_without_usage(self, provider_config):
        """Test stream without usage in final chunk."""
        chunks = [
            MockStreamChunk(choices=[{"delta": {"content": "Hi"}}]),
            MockStreamChunk(choices=[{"finish_reason": "stop"}]),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        async for _ in result.stream:
            pass

        # Usage is estimated from streamed content when provider doesn't return it
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] >= 1  # "Hi" = 2 chars -> ~1 token

    @pytest.mark.asyncio
    async def test_stream_error_in_generator(self, provider_config):
        """Test error handling in stream generator."""

        async def async_iter():
            yield MockStreamChunk(choices=[{"delta": {"content": "Hi"}}])
            raise RuntimeError("Stream error")

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        assert result.stream is not None

        # Consume stream and expect error
        with pytest.raises(RuntimeError, match="Stream error"):
            async for _ in result.stream:
                pass

    @pytest.mark.asyncio
    async def test_stream_with_request_id(self, provider_config):
        """Test stream with request_id for logging."""
        chunks = [
            MockStreamChunk(choices=[{"delta": {"content": "Hi"}}]),
        ]

        async def async_iter():
            for chunk in chunks:
                yield chunk

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = async_iter()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                request_id="req-123",
            )

        # Consume stream
        async for _ in result.stream:
            pass

        assert result.success is True


# ========== Test Error Handling ==========


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_api_timeout_error(self, provider_config):
        """Test APITimeoutError handling."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APITimeoutError("Timeout")
            )
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 504
        assert "timed out" in result.error.lower()
        assert result.adapter_name == "test-openai"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_api_status_error(self, provider_config):
        """Test APIStatusError handling."""
        error = APIStatusError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        error.status_code = 429

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=error)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 429
        assert "Rate limit" in result.error
        assert result.adapter_name == "test-openai"

    @pytest.mark.asyncio
    async def test_generic_exception(self, provider_config):
        """Test generic exception handling."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=ValueError("Something went wrong")
            )
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 502
        assert "Something went wrong" in result.error
        assert result.adapter_name == "test-openai"

    @pytest.mark.asyncio
    async def test_timeout_error_with_request_id(self, provider_config):
        """Test timeout error with request_id for logging."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APITimeoutError("Timeout")
            )
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                request_id="req-456",
            )

        assert result.success is False
        assert result.status_code == 504

    @pytest.mark.asyncio
    async def test_api_status_error_500(self, provider_config):
        """Test APIStatusError with 500 status code."""
        error = APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500),
            body={"error": {"message": "Internal server error"}},
        )
        error.status_code = 500

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=error)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_connection_error(self, provider_config):
        """Test connection error handling."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=ConnectionError("Failed to connect")
            )
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 502
        assert "Failed to connect" in result.error


# ========== Test Queue Integration ==========


class TestQueueIntegration:
    """Tests for queue integration."""

    @pytest.mark.asyncio
    async def test_queue_execution(self, provider_config_with_queue):
        """Test that queue properly executes requests."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config_with_queue)

            # Start the queue
            from app.core.request_queue import get_queue_manager

            queue_manager = get_queue_manager()
            await queue_manager.start()

            result = await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

            assert result.success is True
            assert result.status_code == 200

            # Verify the API was called
            mock_client.chat.completions.create.assert_called_once()

            await queue_manager.stop()

    @pytest.mark.asyncio
    async def test_queue_with_concurrent_requests(self, provider_config_with_queue):
        """Test queue handles concurrent requests properly."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config_with_queue)

            # Start the queue
            from app.core.request_queue import get_queue_manager

            queue_manager = get_queue_manager()
            await queue_manager.start()

            # Make multiple concurrent requests
            tasks = [
                adapter.chat_completion(
                    model_name="gpt-4",
                    messages=[{"role": "user", "content": f"Hi {i}"}],
                    stream=False,
                )
                for i in range(3)
            ]

            results = await asyncio.gather(*tasks)

            assert all(r.success for r in results)
            assert mock_client.chat.completions.create.call_count == 3

            await queue_manager.stop()


# ========== Test Standard Params ==========


class TestStandardParams:
    """Tests for standard OpenAI params handling."""

    def test_standard_params_set(self):
        """Test that _OPENAI_STANDARD_PARAMS contains expected params."""
        expected_params = {
            # 核心参数
            "max_tokens", "temperature", "top_p", "stop", "seed", "n",
            # 工具调用
            "tools", "tool_choice",
            # 格式控制
            "response_format", "stream_options", "max_completion_tokens",
            # 惩罚参数
            "frequency_penalty", "presence_penalty",
            # 日志概率
            "logprobs", "top_logprobs",
            # 其他标准参数
            "logit_bias", "metadata",
            # 用户标识
            "user",
        }
        assert _OPENAI_STANDARD_PARAMS == expected_params

    @pytest.mark.asyncio
    async def test_all_standard_params_passed(self, provider_config):
        """Test that all standard params are passed through."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                max_tokens=100,
                temperature=0.5,
                top_p=0.9,
                stop=["END"],
                tools=[{"type": "function", "function": {"name": "test"}}],
                tool_choice="auto",
                response_format={"type": "json"},
                seed=42,
                n=2,
                frequency_penalty=0.5,
                presence_penalty=0.3,
                logprobs=True,
                top_logprobs=5,
                stream_options={"include_usage": True},
                max_completion_tokens=200,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # Verify all standard params were passed
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["tools"] == [
            {"type": "function", "function": {"name": "test"}}
        ]
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["response_format"] == {"type": "json"}
        assert call_kwargs["seed"] == 42
        assert call_kwargs["n"] == 2
        assert call_kwargs["frequency_penalty"] == 0.5
        assert call_kwargs["presence_penalty"] == 0.3
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 5
        assert call_kwargs["stream_options"] == {"include_usage": True}
        assert call_kwargs["max_completion_tokens"] == 200

        # No extra_body should be present
        assert "extra_body" not in call_kwargs


# ========== Test Mixed Params ==========


class TestMixedParams:
    """Tests for mixed standard and non-standard params."""

    @pytest.mark.asyncio
    async def test_mixed_params(self, provider_config):
        """Test that standard params pass through and non-standard go to extra_body."""
        mock_response = MockResponse()

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter(provider_config)
            await adapter.chat_completion(
                model_name="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                temperature=0.7,  # Standard
                top_k=50,  # Non-standard (DashScope param)
                custom_field="value",  # Non-standard
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # Standard param should be at top level
        assert call_kwargs["temperature"] == 0.7

        # Non-standard params should be in extra_body
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["top_k"] == 50
        assert call_kwargs["extra_body"]["custom_field"] == "value"
