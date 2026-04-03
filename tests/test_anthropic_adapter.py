"""Comprehensive tests for AnthropicAdapter and related conversion functions."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError, APITimeoutError

from app.adapters.anthropic_adapter import (
    AnthropicAdapter,
    _anthropic_event_to_openai_chunk,
    _anthropic_response_to_openai,
    _make_chunk,
    _openai_to_anthropic_messages,
)
from app.models.config_models import ProviderConfig

# ========== Test Fixtures ==========


@pytest.fixture
def provider_config():
    """Create a basic provider config for testing."""
    return ProviderConfig(
        name="test-anthropic",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="sk-test-key",
    )


# ========== Mock Classes for Anthropic SDK ==========


@dataclass
class MockUsage:
    input_tokens: int = 10
    output_tokens: int = 20


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = "Hello, I am Claude."


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_123"
    name: str = "get_weather"
    input: dict = field(default_factory=lambda: {"city": "San Francisco"})


@dataclass
class MockAnthropicResponse:
    """Mock Anthropic messages.create response."""

    id: str = "msg_123"
    model: str = "claude-3-sonnet"
    content: list = field(default_factory=lambda: [MockTextBlock()])
    stop_reason: str = "end_turn"
    usage: MockUsage = field(default_factory=MockUsage)


@dataclass
class MockTextDelta:
    type: str = "text_delta"
    text: str = "Hello"


@dataclass
class MockInputJsonDelta:
    type: str = "input_json_delta"
    partial_json: str = '{"city":'


@dataclass
class MockContentBlock:
    """Mock content block for streaming."""

    type: str = "text"
    text: str = ""
    id: str = ""
    name: str = ""


@dataclass
class MockContentBlockStartEvent:
    type: str = "content_block_start"
    index: int = 0
    content_block: Any = None


@dataclass
class MockContentBlockDeltaEvent:
    type: str = "content_block_delta"
    index: int = 0
    delta: Any = None


@dataclass
class MockMessageDelta:
    stop_reason: str = "end_turn"


@dataclass
class MockMessageDeltaEvent:
    type: str = "message_delta"
    delta: Any = None
    usage: Any = None


@dataclass
class MockMessageStartEvent:
    type: str = "message_start"
    message: Any = None


# ========== Test AnthropicAdapter.__init__ ==========


class TestAnthropicAdapterInit:
    """Tests for AnthropicAdapter initialization."""

    def test_init_with_base_url(self, provider_config):
        """Test initialization with base_url configured."""
        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            adapter = AnthropicAdapter(provider_config)
            mock_client_cls.assert_called_once_with(
                api_key="sk-test-key",
                base_url="https://api.anthropic.com",
            )
            assert adapter.name == "test-anthropic"
            assert adapter.provider == provider_config

    def test_init_without_base_url(self):
        """Test initialization without base_url."""
        config = ProviderConfig(
            name="test-no-base",
            provider="anthropic",
            base_url="",  # Empty base_url
            api_key="sk-test-key",
        )
        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            adapter = AnthropicAdapter(config)
            # Empty string is falsy, so base_url should not be passed
            call_kwargs = mock_client_cls.call_args[1]
            assert "base_url" not in call_kwargs or call_kwargs.get("base_url") == ""
            assert adapter.name == "test-no-base"


# ========== Test AnthropicAdapter.chat_completion ==========


class TestChatCompletionNonStream:
    """Tests for non-streaming chat completion."""

    @pytest.mark.asyncio
    async def test_non_stream_text_success(self, provider_config):
        """Test successful non-stream text response."""
        mock_response = MockAnthropicResponse(
            content=[MockTextBlock(text="Hello, world!")],
            stop_reason="end_turn",
        )

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is True
        assert result.status_code == 200
        assert result.adapter_name == "test-anthropic"
        assert result.model_name == "claude-3-sonnet"
        assert result.body is not None
        assert result.body["choices"][0]["message"]["content"] == "Hello, world!"
        assert result.body["choices"][0]["finish_reason"] == "stop"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_non_stream_tool_use_success(self, provider_config):
        """Test successful non-stream tool_use response."""
        mock_response = MockAnthropicResponse(
            content=[MockToolUseBlock()],
            stop_reason="tool_use",
        )

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "What's the weather?"}],
                stream=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                tool_choice="auto",
            )

        assert result.success is True
        assert result.status_code == 200
        assert result.body is not None
        assert "tool_calls" in result.body["choices"][0]["message"]
        tool_calls = result.body["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "toolu_123"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert result.body["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_non_stream_with_system_message(self, provider_config):
        """Test non-stream with system message."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                stream=False,
            )

        # Verify system was passed to API
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert call_kwargs["system"] == [{"type": "text", "text": "You are helpful."}]

    @pytest.mark.asyncio
    async def test_non_stream_with_all_params(self, provider_config):
        """Test non-stream with all optional parameters."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                stop=["END", "STOP"],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["stop_sequences"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_non_stream_with_tool_choice_required(self, provider_config):
        """Test non-stream with tool_choice='required'."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=[
                    {"type": "function", "function": {"name": "test", "parameters": {}}}
                ],
                tool_choice="required",
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "any"}

    @pytest.mark.asyncio
    async def test_non_stream_with_tool_choice_named(self, provider_config):
        """Test non-stream with named tool_choice."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=[
                    {"type": "function", "function": {"name": "test", "parameters": {}}}
                ],
                tool_choice={"function": {"name": "get_weather"}},
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "get_weather"}

    @pytest.mark.asyncio
    async def test_non_stream_timeout_error(self, provider_config):
        """Test non-stream timeout error handling."""
        from anthropic import APITimeoutError as AnthropicTimeout

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=AnthropicTimeout("Timeout")
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 504
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_non_stream_api_status_error(self, provider_config):
        """Test non-stream API status error handling."""
        from anthropic import APIStatusError as AnthropicStatusError

        mock_response = MagicMock()
        mock_response.status_code = 400

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=AnthropicStatusError(
                    "Bad request", response=mock_response, body={"error": "bad"}
                )
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 400
        assert "Bad request" in result.error

    @pytest.mark.asyncio
    async def test_non_stream_generic_exception(self, provider_config):
        """Test non-stream generic exception handling."""
        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=ValueError("Unexpected error")
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.success is False
        assert result.status_code == 502
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_non_stream_timeout_param(self, provider_config):
        """Test that timeout parameter is passed correctly."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                timeout=120,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == 120

    @pytest.mark.asyncio
    async def test_non_stream_without_tools(self, provider_config):
        """Test non-stream without tools parameter."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


class TestChatCompletionStream:
    """Tests for streaming chat completion."""

    @pytest.mark.asyncio
    async def test_stream_text_success(self, provider_config):
        """Test successful streaming text response."""
        events = [
            MockContentBlockStartEvent(content_block=MockContentBlock(type="text")),
            MockContentBlockDeltaEvent(delta=MockTextDelta(text="Hello")),
            MockContentBlockDeltaEvent(delta=MockTextDelta(text=" world")),
            MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="end_turn")),
        ]

        async def async_iter():
            for e in events:
                yield e

        mock_stream = async_iter()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_stream)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        assert result.status_code == 200
        assert result.stream is not None

        chunks = []
        async for chunk in result.stream:
            chunks.append(chunk)

        assert len(chunks) == 4
        # First chunk: content block start
        assert chunks[0]["choices"][0]["delta"]["content"] == ""
        # Text deltas
        assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[2]["choices"][0]["delta"]["content"] == " world"
        # Finish
        assert chunks[3]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_tool_use_success(self, provider_config):
        """Test successful streaming tool_use response."""
        events = [
            MockContentBlockStartEvent(
                index=0,
                content_block=MockContentBlock(
                    type="tool_use", id="toolu_123", name="get_weather"
                ),
            ),
            MockContentBlockDeltaEvent(
                index=0, delta=MockInputJsonDelta(partial_json='{"city": ')
            ),
            MockContentBlockDeltaEvent(
                index=0, delta=MockInputJsonDelta(partial_json='"SF"}')
            ),
            MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="tool_use")),
        ]

        async def async_iter():
            for e in events:
                yield e

        mock_stream = async_iter()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_stream)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        assert result.stream is not None

        chunks = []
        async for chunk in result.stream:
            chunks.append(chunk)

        assert len(chunks) == 4
        # Tool use start
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_123"
        assert (
            chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "get_weather"
        )
        # JSON deltas
        assert (
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '{"city": '
        )
        assert (
            chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '"SF"}'
        )
        # Finish
        assert chunks[3]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_error(self, provider_config):
        """Test streaming error handling."""

        async def async_iter():
            yield MockContentBlockStartEvent(
                content_block=MockContentBlock(type="text")
            )
            raise ValueError("Stream error")

        mock_stream = async_iter()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_stream)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        assert result.stream is not None

        chunks = []
        with pytest.raises(ValueError, match="Stream error"):
            async for chunk in result.stream:
                chunks.append(chunk)

        # Got one chunk before error
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_unknown_event_skipped(self, provider_config):
        """Test that unknown event types are skipped."""
        events = [
            MockMessageStartEvent(message=MagicMock()),  # Unknown event, returns None
            MockContentBlockDeltaEvent(delta=MockTextDelta(text="Hello")),
        ]

        async def async_iter():
            for e in events:
                yield e

        mock_stream = async_iter()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_stream)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        chunks = []
        async for chunk in result.stream:
            chunks.append(chunk)

        # Only the delta event should produce a chunk
        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"


class TestChatCompletionEdgeCases:
    """Tests for edge cases in chat_completion."""

    @pytest.mark.asyncio
    async def test_default_max_tokens(self, provider_config):
        """Test that default max_tokens is 4096."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_request_id_in_logs(self, provider_config):
        """Test that request_id appears in debug logs."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with patch("app.adapters.anthropic_adapter.logger") as mock_logger:
                adapter = AnthropicAdapter(provider_config)
                await adapter.chat_completion(
                    model_name="claude-3-sonnet",
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=False,
                    request_id="test-req-123",
                )

            # Check that logger.debug was called with request_id
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("test-req-123" in str(c) for c in debug_calls)


# ========== Test _openai_to_anthropic_messages ==========


class TestOpenaiToAnthropicMessages:
    """Tests for _openai_to_anthropic_messages conversion."""

    def test_simple_user_message(self):
        """Test simple user message conversion."""
        messages = [{"role": "user", "content": "Hello"}]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert system is None
        assert anth_msgs == [{"role": "user", "content": "Hello"}]

    def test_system_message(self):
        """Test system message extraction."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert system == [{"type": "text", "text": "You are helpful."}]
        assert anth_msgs == [{"role": "user", "content": "Hi"}]

    def test_multiple_system_messages(self):
        """Test multiple system messages are combined."""
        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hi"},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert system == [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        assert len(anth_msgs) == 1

    def test_assistant_message_simple(self):
        """Test simple assistant message."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert anth_msgs == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

    def test_assistant_message_empty_content(self):
        """Test assistant message with empty content."""
        messages = [
            {"role": "assistant", "content": ""},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert anth_msgs == [{"role": "assistant", "content": ""}]

    def test_assistant_with_tool_calls(self):
        """Test assistant message with tool_calls."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert len(anth_msgs) == 1
        assert anth_msgs[0]["role"] == "assistant"
        content = anth_msgs[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["id"] == "call_123"
        assert content[0]["name"] == "get_weather"
        assert content[0]["input"] == {"city": "SF"}

    def test_assistant_with_tool_calls_and_text(self):
        """Test assistant message with both text and tool_calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{}",
                        },
                    }
                ],
            },
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        content = anth_msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me check."
        assert content[1]["type"] == "tool_use"

    def test_assistant_with_invalid_json_arguments(self):
        """Test assistant message with invalid JSON in arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": "not valid json",
                        },
                    }
                ],
            },
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        content = anth_msgs[0]["content"]
        assert content[0]["input"] == {}  # Falls back to empty dict

    def test_tool_message(self):
        """Test tool role message conversion."""
        messages = [
            {"role": "tool", "tool_call_id": "toolu_123", "content": "Result: 72°F"},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert anth_msgs == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Result: 72°F",
                    }
                ],
            }
        ]

    def test_tool_message_empty_content(self):
        """Test tool message with empty content."""
        messages = [
            {"role": "tool", "tool_call_id": "toolu_123", "content": ""},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert anth_msgs[0]["content"][0]["content"] == ""

    def test_multiple_tool_results(self):
        """Test multiple tool result messages."""
        messages = [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "Result 1"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "Result 2"},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert len(anth_msgs) == 2
        assert anth_msgs[0]["content"][0]["tool_use_id"] == "toolu_1"
        assert anth_msgs[1]["content"][0]["tool_use_id"] == "toolu_2"

    def test_object_style_messages(self):
        """Test messages as objects with attributes instead of dicts."""

        class MockMessage:
            def __init__(self, role, content, tool_calls=None, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_call_id = tool_call_id

        class MockToolCall:
            def __init__(self, id, function):
                self.id = id
                self.function = function

        class MockFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        messages = [
            MockMessage("system", "You are helpful"),
            MockMessage("user", "Hi"),
            MockMessage(
                "assistant",
                "",
                tool_calls=[MockToolCall("call_123", MockFunction("test", '{"x":1}'))],
            ),
            MockMessage("tool", "result", tool_call_id="call_123"),
        ]

        system, anth_msgs = _openai_to_anthropic_messages(messages)

        assert system == [{"type": "text", "text": "You are helpful"}]
        assert len(anth_msgs) == 3
        assert anth_msgs[0]["role"] == "user"
        assert anth_msgs[1]["role"] == "assistant"
        assert anth_msgs[1]["content"][0]["type"] == "tool_use"
        assert anth_msgs[2]["role"] == "user"
        assert anth_msgs[2]["content"][0]["type"] == "tool_result"

    def test_empty_messages(self):
        """Test with empty message list."""
        system, anth_msgs = _openai_to_anthropic_messages([])
        assert system is None
        assert anth_msgs == []

    def test_unknown_role_ignored(self):
        """Test that messages with unknown roles are handled."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "unknown", "content": "???"},
        ]
        system, anth_msgs = _openai_to_anthropic_messages(messages)

        # Unknown roles get empty role string, content still added
        assert len(anth_msgs) == 1  # Only user message


# ========== Test _anthropic_response_to_openai ==========


class TestAnthropicResponseToOpenai:
    """Tests for _anthropic_response_to_openai conversion."""

    def test_text_response(self):
        """Test simple text response conversion."""
        response = MockAnthropicResponse(
            id="msg_123",
            model="claude-3-sonnet",
            content=[MockTextBlock(text="Hello, world!")],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=10, output_tokens=5),
        )

        result = _anthropic_response_to_openai(response)

        assert result["id"] == "msg_123"
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-3-sonnet"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello, world!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_use_response(self):
        """Test tool_use response conversion."""
        response = MockAnthropicResponse(
            id="msg_456",
            model="claude-3-sonnet",
            content=[
                MockToolUseBlock(
                    id="toolu_123",
                    name="get_weather",
                    input={"city": "San Francisco", "unit": "celsius"},
                )
            ],
            stop_reason="tool_use",
            usage=MockUsage(input_tokens=20, output_tokens=10),
        )

        result = _anthropic_response_to_openai(response)

        assert result["id"] == "msg_456"
        message = result["choices"][0]["message"]
        assert message["content"] is None
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        tc = message["tool_calls"][0]
        assert tc["id"] == "toolu_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {
            "city": "San Francisco",
            "unit": "celsius",
        }
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_mixed_text_and_tool_use(self):
        """Test response with both text and tool_use."""
        response = MockAnthropicResponse(
            content=[
                MockTextBlock(text="Let me check that for you."),
                MockToolUseBlock(
                    id="toolu_123", name="search", input={"query": "test"}
                ),
            ],
            stop_reason="tool_use",
        )

        result = _anthropic_response_to_openai(response)

        message = result["choices"][0]["message"]
        # Text is captured, tool_calls is present
        assert message["content"] == "Let me check that for you."
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1

    def test_stop_reason_max_tokens(self):
        """Test max_tokens stop_reason maps to 'length'."""
        response = MockAnthropicResponse(
            content=[MockTextBlock(text="Truncated")],
            stop_reason="max_tokens",
        )

        result = _anthropic_response_to_openai(response)

        assert result["choices"][0]["finish_reason"] == "length"

    def test_stop_reason_unknown(self):
        """Test unknown stop_reason defaults to 'stop'."""
        response = MockAnthropicResponse(
            content=[MockTextBlock(text="Done")],
            stop_reason="unknown_reason",
        )

        result = _anthropic_response_to_openai(response)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_multiple_tool_uses(self):
        """Test response with multiple tool_use blocks."""
        response = MockAnthropicResponse(
            content=[
                MockToolUseBlock(id="toolu_1", name="func1", input={"a": 1}),
                MockToolUseBlock(id="toolu_2", name="func2", input={"b": 2}),
            ],
            stop_reason="tool_use",
        )

        result = _anthropic_response_to_openai(response)

        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "toolu_1"
        assert tool_calls[1]["id"] == "toolu_2"

    def test_empty_content(self):
        """Test response with empty content list."""
        response = MockAnthropicResponse(
            content=[],
            stop_reason="end_turn",
        )

        result = _anthropic_response_to_openai(response)

        assert result["choices"][0]["message"]["content"] is None
        assert result["choices"][0]["finish_reason"] == "stop"


# ========== Test _anthropic_event_to_openai_chunk ==========


class TestAnthropicEventToOpenaiChunk:
    """Tests for _anthropic_event_to_openai_chunk conversion."""

    def test_content_block_start_text(self):
        """Test content_block_start with text block."""
        event = MockContentBlockStartEvent(content_block=MockContentBlock(type="text"))

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", -1)

        assert result is not None
        assert result["model"] == "claude-3-sonnet"
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == ""
        assert result["choices"][0]["finish_reason"] is None

    def test_content_block_start_tool_use(self):
        """Test content_block_start with tool_use block."""
        event = MockContentBlockStartEvent(
            content_block=MockContentBlock(
                type="tool_use", id="toolu_123", name="get_weather"
            )
        )

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", -1)

        assert result is not None
        assert result["_tc_index_bump"] == 0  # New index
        delta = result["choices"][0]["delta"]
        assert "tool_calls" in delta
        tc = delta["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["id"] == "toolu_123"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == ""

    def test_content_block_start_tool_use_increment_index(self):
        """Test tool_use increments from current index."""
        event = MockContentBlockStartEvent(
            content_block=MockContentBlock(
                type="tool_use", id="toolu_456", name="search"
            )
        )

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 2)

        assert result["_tc_index_bump"] == 3
        assert result["choices"][0]["delta"]["tool_calls"][0]["index"] == 3

    def test_content_block_delta_text(self):
        """Test content_block_delta with text_delta."""
        event = MockContentBlockDeltaEvent(delta=MockTextDelta(text="Hello"))

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_content_block_delta_input_json(self):
        """Test content_block_delta with input_json_delta."""
        event = MockContentBlockDeltaEvent(
            delta=MockInputJsonDelta(partial_json='{"city": "SF"')
        )

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 1)

        assert result is not None
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["index"] == 1  # max(current_tc_index, 0)
        assert tc["function"]["arguments"] == '{"city": "SF"'

    def test_content_block_delta_input_json_negative_index(self):
        """Test input_json_delta with negative current index uses 0."""
        event = MockContentBlockDeltaEvent(
            delta=MockInputJsonDelta(partial_json='{"x"')
        )

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", -5)

        assert result["choices"][0]["delta"]["tool_calls"][0]["index"] == 0

    def test_message_delta_end_turn(self):
        """Test message_delta with end_turn stop_reason."""
        event = MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="end_turn"))

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_message_delta_max_tokens(self):
        """Test message_delta with max_tokens stop_reason."""
        event = MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="max_tokens"))

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result["choices"][0]["finish_reason"] == "length"

    def test_message_delta_tool_use(self):
        """Test message_delta with tool_use stop_reason."""
        event = MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="tool_use"))

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_message_delta_unknown_stop_reason(self):
        """Test message_delta with unknown stop_reason defaults to stop."""
        event = MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="unknown"))

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_unknown_event_type_returns_none(self):
        """Test that unknown event types return None."""
        event = MockMessageStartEvent(message=MagicMock())

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result is None

    def test_content_block_start_unknown_type_returns_none(self):
        """Test content_block_start with unknown block type returns None."""

        @dataclass
        class UnknownBlock:
            type: str = "unknown"

        event = MockContentBlockStartEvent(content_block=UnknownBlock())

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result is None

    def test_content_block_delta_unknown_type_returns_none(self):
        """Test content_block_delta with unknown delta type returns None."""

        @dataclass
        class UnknownDelta:
            type: str = "unknown"

        event = MockContentBlockDeltaEvent(delta=UnknownDelta())

        result = _anthropic_event_to_openai_chunk(event, "claude-3-sonnet", 0)

        assert result is None


# ========== Test _make_chunk ==========


class TestMakeChunk:
    """Tests for _make_chunk helper function."""

    def test_make_chunk_content_only(self):
        """Test chunk with content only."""
        result = _make_chunk("claude-3-sonnet", content="Hello")

        assert result["object"] == "chat.completion.chunk"
        assert result["model"] == "claude-3-sonnet"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["index"] == 0
        assert result["choices"][0]["delta"]["content"] == "Hello"
        assert result["choices"][0]["finish_reason"] is None

    def test_make_chunk_tool_call_only(self):
        """Test chunk with tool_call only."""
        tool_call = {
            "index": 0,
            "id": "call_123",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        }
        result = _make_chunk("claude-3-sonnet", tool_call=tool_call)

        assert "tool_calls" in result["choices"][0]["delta"]
        assert result["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_123"

    def test_make_chunk_finish_reason_only(self):
        """Test chunk with finish_reason only."""
        result = _make_chunk("claude-3-sonnet", finish_reason="stop")

        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["delta"] == {}

    def test_make_chunk_all_params(self):
        """Test chunk with all parameters."""
        tool_call = {"index": 0, "function": {"arguments": '{"x":1}'}}
        result = _make_chunk(
            model="claude-3-sonnet",
            content="Partial",
            tool_call=tool_call,
            finish_reason="tool_calls",
        )

        assert result["choices"][0]["delta"]["content"] == "Partial"
        assert "tool_calls" in result["choices"][0]["delta"]
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_make_chunk_empty_content(self):
        """Test chunk with empty string content."""
        result = _make_chunk("claude-3-sonnet", content="")

        assert result["choices"][0]["delta"]["content"] == ""


# ========== Test Timeout Error Conversion ==========


class TestTimeoutErrorConversion:
    """Tests for timeout error conversion from Anthropic to OpenAI format."""

    @pytest.mark.asyncio
    async def test_anthropic_timeout_converted_to_openai_timeout(self, provider_config):
        """Test that Anthropic APITimeoutError is converted to OpenAI APITimeoutError."""
        from anthropic import APITimeoutError as AnthropicTimeout

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=AnthropicTimeout("Timeout")
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        # Should be caught and converted to 504 Gateway Timeout
        assert result.status_code == 504
        assert result.success is False


# ========== Test API Status Error Conversion ==========


class TestAPIStatusErrorConversion:
    """Tests for API status error conversion."""

    @pytest.mark.asyncio
    async def test_anthropic_status_error_converted(self, provider_config):
        """Test that Anthropic APIStatusError is converted to OpenAI APIStatusError."""
        from anthropic import APIStatusError as AnthropicStatusError

        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=AnthropicStatusError(
                    "Rate limited", response=mock_response, body={"error": "rate_limit"}
                )
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.status_code == 429
        assert result.success is False
        assert "Rate limited" in result.error


# ========== Test Stream Timeout Error ==========


class TestStreamTimeoutError:
    """Tests for timeout errors during streaming."""

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self, provider_config):
        """Test timeout error during stream creation."""
        from anthropic import APITimeoutError as AnthropicTimeout

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=AnthropicTimeout("Stream timeout")
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is False
        assert result.status_code == 504


# ========== Test Tool Conversion Edge Cases ==========


class TestToolConversionEdgeCases:
    """Tests for tool conversion edge cases."""

    @pytest.mark.asyncio
    async def test_tools_without_description(self, provider_config):
        """Test tools without description field."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            # No description
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "test_func"
        assert call_kwargs["tools"][0]["description"] == ""

    @pytest.mark.asyncio
    async def test_tools_without_parameters(self, provider_config):
        """Test tools without parameters field."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "description": "A test function",
                            # No parameters
                        },
                    }
                ],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"][0]["input_schema"] == {}

    @pytest.mark.asyncio
    async def test_tools_without_function_key(self, provider_config):
        """Test tools without function key in tool dict."""
        mock_response = MockAnthropicResponse()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=[
                    {"type": "function"}  # Missing function key
                ],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        # Should have empty values
        assert call_kwargs["tools"][0]["name"] == ""
        assert call_kwargs["tools"][0]["description"] == ""
        assert call_kwargs["tools"][0]["input_schema"] == {}

    # ========== Test Stream Status Error ==========

    @pytest.mark.asyncio
    async def test_stream_multiple_tool_uses(self, provider_config):
        """Test streaming with multiple tool_use blocks to cover tool_call_index bump."""
        events = [
            # First tool_use
            MockContentBlockStartEvent(
                index=0,
                content_block=MockContentBlock(
                    type="tool_use", id="toolu_1", name="func1"
                ),
            ),
            MockContentBlockDeltaEvent(
                index=0, delta=MockInputJsonDelta(partial_json='{"a":')
            ),
            MockContentBlockDeltaEvent(
                index=0, delta=MockInputJsonDelta(partial_json="1}")
            ),
            # Second tool_use
            MockContentBlockStartEvent(
                index=1,
                content_block=MockContentBlock(
                    type="tool_use", id="toolu_2", name="func2"
                ),
            ),
            MockContentBlockDeltaEvent(
                index=1, delta=MockInputJsonDelta(partial_json='{"b":')
            ),
            MockContentBlockDeltaEvent(
                index=1, delta=MockInputJsonDelta(partial_json="2}")
            ),
            MockMessageDeltaEvent(delta=MockMessageDelta(stop_reason="tool_use")),
        ]

        async def async_iter():
            for e in events:
                yield e

        mock_stream = async_iter()

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_stream)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is True
        assert result.stream is not None

        chunks = []
        async for chunk in result.stream:
            chunks.append(chunk)

        # Should have 7 chunks: 2 tool_use starts + 4 input_json deltas + 1 finish
        assert len(chunks) == 7
        # First tool_use start should have index 0
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_1"
        # Second tool_use should have index 1 (after bug fix: now correctly tracks tool_call_index)
        assert chunks[3]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
        assert chunks[3]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_2"
        # Finish reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamStatusError:
    """Tests for API status errors during streaming."""

    @pytest.mark.asyncio
    async def test_stream_api_status_error(self, provider_config):
        """Test API status error during stream creation."""
        from anthropic import APIStatusError as AnthropicStatusError

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=AnthropicStatusError(
                    "Internal error", response=mock_response, body={}
                )
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is False
        assert result.status_code == 500


# ========== Test Stream Generic Error ==========


class TestStreamGenericError:
    """Tests for generic errors during streaming."""

    @pytest.mark.asyncio
    async def test_stream_generic_error(self, provider_config):
        """Test generic error during stream creation."""
        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=RuntimeError("Something went wrong")
            )
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter(provider_config)
            result = await adapter.chat_completion(
                model_name="claude-3-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        assert result.success is False
        assert result.status_code == 502
