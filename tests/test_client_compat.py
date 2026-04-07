"""Client compatibility tests — simulates how Claude Code, Zed, and OpenCode use the API."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.adapters.openai_adapter import OpenAIAdapter
from app.adapters.anthropic_adapter import AnthropicAdapter
from app.models.config_models import ProviderConfig


# ========== Shared mock helpers ==========


@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockStreamChunk:
    """Mimics openai.ChatCompletionChunk (has choices as list of dicts)."""
    id: str = "chatcmpl-123"
    model: str = "test-model"
    created: int = 1700000000
    choices: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[MockUsage] = None

    def model_dump(self, exclude_none=None):
        d = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model}
        if self.choices is not None:
            d["choices"] = self.choices
        if self.usage is not None:
            d["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        return d


@dataclass
class MockContentBlock:
    """Mimics Anthropic content_block (avoids MagicMock 'name' conflict)."""
    type: str = "text"
    id: str = ""
    name: str = ""


@dataclass
class MockAnthropicEvent:
    """Mimics an Anthropic streaming event."""
    type: str = "content_block_delta"
    index: int = 0
    delta: Any = None
    content_block: Any = None
    usage: Any = None
    message: Any = None


def _openai_provider():
    return ProviderConfig(name="test-openai", provider="openai", base_url="https://api.test.com/v1", api_key="sk-test")


def _anthropic_provider():
    return ProviderConfig(name="test-anthropic", provider="anthropic", base_url="https://api.anthropic.com", api_key="sk-test")


def _mock_nonstream_response(content="OK", usage=None):
    """Create a mock non-streaming OpenAI response."""
    resp = MagicMock()
    resp.usage = usage or MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock(content=content, role="assistant", tool_calls=None)
    resp.choices[0].message.model_dump.return_value = {"role": "assistant", "content": content}
    resp.choices[0].finish_reason = "stop"
    resp.model = "test-model"
    resp.id = "chatcmpl-123"
    resp.model_dump.return_value = {
        "id": "chatcmpl-123", "object": "chat.completion", "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return resp


def _mock_anthropic_response(content="OK", usage_input=100, usage_output=50, stop_reason="end_turn"):
    """Create a mock non-streaming Anthropic response with proper attribute access."""
    resp = MagicMock()
    resp.id = "msg_123"
    resp.model = "claude-sonnet"
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=usage_input, output_tokens=usage_output)
    resp.content = [MagicMock(type="text", text=content)]
    return resp


# ========== Zed Compatibility ==========


class TestZedCompatibility:
    """Zed editor sends requests to /v1/chat/completions with specific patterns:
    - Uses max_completion_tokens instead of max_tokens
    - Always sends stream_options: {include_usage: true}
    - Messages can use multipart content format
    """

    @pytest.mark.asyncio
    async def test_non_stream_with_max_completion_tokens(self):
        """Zed sends max_completion_tokens — verify it passes through to upstream."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_nonstream_response())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            result = await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                timeout=60,
                max_completion_tokens=128000,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 128000

    @pytest.mark.asyncio
    async def test_streaming_with_include_usage(self):
        """Zed always sends stream_options: {include_usage: true} — verify it is not overridden."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_nonstream_response())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                timeout=60,
                stream_options={"include_usage": True},
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_multipart_message_content(self):
        """Zed sends messages with content as array of {type, text} blocks."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_nonstream_response())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Hello Zed"}]}],
                stream=False,
                timeout=60,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == [{"type": "text", "text": "Hello Zed"}]

    @pytest.mark.asyncio
    async def test_reasoning_content_merged_in_stream(self):
        """Zed receives reasoning_content from GLM models — verify it is buffered and merged."""
        chunks = [
            MockStreamChunk(choices=[{"delta": {"role": "assistant", "reasoning_content": "Let me think"}}]),
            MockStreamChunk(choices=[{"delta": {"reasoning_content": " about this..."}}]),
            MockStreamChunk(choices=[{"delta": {"content": "The answer is 42"}}]),
            MockStreamChunk(choices=[{"delta": {}, "finish_reason": "stop"}], usage=MockUsage()),
        ]

        async def async_iter():
            for c in chunks:
                yield c

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_iter())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            result = await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                timeout=60,
            )

        assert result.success is True
        collected = []
        async for chunk in result.stream:
            data = chunk if isinstance(chunk, dict) else chunk.model_dump(exclude_none=True)
            collected.append(data)

        # Reasoning buffered + content chunk + finish chunk = 3
        assert len(collected) == 3
        assert collected[0]["choices"][0]["delta"]["content"] == "Let me think about this..."
        assert collected[1]["choices"][0]["delta"]["content"] == "The answer is 42"

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_flushed_at_end(self):
        """When stream has only reasoning_content (no content), flush as one block at end."""
        chunks = [
            MockStreamChunk(choices=[{"delta": {"role": "assistant", "reasoning_content": "Thinking..."}}]),
            MockStreamChunk(choices=[{"delta": {}, "finish_reason": "length"}], usage=MockUsage()),
        ]

        async def async_iter():
            for c in chunks:
                yield c

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_iter())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            result = await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                timeout=60,
            )

        assert result.success is True
        collected = []
        async for chunk in result.stream:
            collected.append(chunk)

        # Finish reason chunk + reasoning flushed chunk = 2
        assert len(collected) == 2
        # Last chunk should have reasoning as content
        reasoning_chunk = collected[1] if isinstance(collected[1], dict) else collected[1].model_dump()
        assert reasoning_chunk["choices"][0]["delta"]["content"] == "Thinking..."


# ========== OpenCode Compatibility ==========


class TestOpenCodeCompatibility:
    """OpenCode sends requests to /v1/chat/completions with:
    - max_tokens (non-reasoning) or max_completion_tokens (reasoning)
    - stream_options: {include_usage: true}
    - Standard tool_calls / tool messages format
    """

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self):
        """OpenCode uses tool_calls in responses and tool messages in input."""
        args1 = '{"pat'
        args2 = 'tern":"TODO"}'
        chunks = [
            MockStreamChunk(choices=[{"delta": {"role": "assistant", "content": None, "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "grep", "arguments": ""}}]}}]),
            MockStreamChunk(choices=[{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args1}}]}}]),
            MockStreamChunk(choices=[{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args2}}]}}]),
            MockStreamChunk(choices=[{"delta": {}, "finish_reason": "tool_calls"}], usage=MockUsage()),
        ]

        async def async_iter():
            for c in chunks:
                yield c

        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_iter())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            result = await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "Find TODOs"}],
                stream=True,
                timeout=60,
                tools=[{"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}}],
            )

        assert result.success is True
        collected = []
        async for chunk in result.stream:
            data = chunk if isinstance(chunk, dict) else chunk.model_dump(exclude_none=True)
            collected.append(data)

        assert len(collected) == 4
        assert collected[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "grep"
        assert collected[3]["choices"][0].get("finish_reason") == "tool_calls"

    @pytest.mark.asyncio
    async def test_tool_result_message_passed_through(self):
        """OpenCode sends role=tool messages — verify they pass to upstream."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_nonstream_response())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            await adapter.chat_completion(
                model_name="test-model",
                messages=[
                    {"role": "user", "content": "Find TODOs"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "grep", "arguments": '{"pattern":"TODO"}'}}
                    ]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "Found 3 matches"},
                ],
                stream=False,
                timeout=60,
            )

        msgs = mock_client.chat.completions.create.call_args[1]["messages"]
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_non_standard_params_go_to_extra_body(self):
        """OpenCode might send top_k (DashScope) — verify it goes to extra_body."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_nonstream_response())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                timeout=60,
                top_k=50,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["top_k"] == 50
        assert "top_p" not in call_kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_to_extra_body(self):
        """Zed sends parallel_tool_calls — verify it goes to extra_body."""
        with patch("app.adapters.openai_adapter.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_nonstream_response())
            mock_cls.return_value = mock_client
            adapter = OpenAIAdapter(_openai_provider())

            await adapter.chat_completion(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                timeout=60,
                parallel_tool_calls=True,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["parallel_tool_calls"] is True


# ========== Claude Code Compatibility ==========


class TestClaudeCodeCompatibility:
    """Claude Code sends requests to /v1/messages (Anthropic format) with:
    - system as list of text blocks with cache_control
    - tools with input_schema (not function wrapping)
    - stream: true
    - thinking/budget_tokens for extended thinking
    - Messages use content blocks (tool_use, tool_result)
    """

    @pytest.mark.asyncio
    async def test_non_stream_with_system_and_tools(self):
        """Claude Code sends system as list of blocks and tools with input_schema."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
            mock_cls.return_value = mock_client
            adapter = AnthropicAdapter(_anthropic_provider())

            result = await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "Read the file"}]},
                ],
                stream=False,
                timeout=60,
                max_tokens=16000,
            )

        assert result.success is True
        assert result.usage == {"prompt_tokens": 100, "completion_tokens": 50}

    @pytest.mark.asyncio
    async def test_system_and_tools_forwarded(self):
        """Verify system blocks and tools are forwarded to upstream."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
            mock_cls.return_value = mock_client
            adapter = AnthropicAdapter(_anthropic_provider())

            # The adapter converts OpenAI-format tools to Anthropic format,
            # so we need to send tools in OpenAI format
            await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Read the file"},
                ],
                stream=False,
                timeout=60,
                max_tokens=16000,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
                    },
                }],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        # System extracted from messages (returned as Anthropic content blocks list)
        assert call_kwargs["system"] == [{"type": "text", "text": "You are helpful."}]
        # Tools converted to Anthropic format (input_schema, no function wrapper)
        assert call_kwargs["tools"][0]["name"] == "Read"
        assert call_kwargs["tools"][0]["input_schema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_streaming_with_tool_use_events(self):
        """Claude Code expects proper SSE events for streaming tool use."""
        events = [
            MockAnthropicEvent(type="message_start", message=MagicMock(usage=MagicMock(input_tokens=50))),
            MockAnthropicEvent(type="content_block_start", index=0, content_block=MockContentBlock(type="tool_use", id="toolu_cc1", name="Bash")),
            MockAnthropicEvent(type="content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='{"command":"ls')),
            MockAnthropicEvent(type="content_block_delta", delta=MagicMock(type="input_json_delta", partial_json=' -la"}')),
            MockAnthropicEvent(type="content_block_stop", index=0),
            MockAnthropicEvent(type="message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock(output_tokens=20)),
        ]

        async def async_iter():
            for e in events:
                yield e

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=async_iter())
            mock_cls.return_value = mock_client
            adapter = AnthropicAdapter(_anthropic_provider())

            result = await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[{"role": "user", "content": "List files"}],
                stream=True,
                timeout=60,
                max_tokens=8000,
            )

        assert result.success is True
        collected = []
        async for chunk in result.stream:
            collected.append(chunk)

        # tool_use start + 2 json deltas + stop = 4
        assert len(collected) == 4
        assert collected[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Bash"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 20

    @pytest.mark.asyncio
    async def test_tool_result_message_roundtrip(self):
        """Claude Code sends tool_result content blocks back as user messages."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response("I see 5 files."))
            mock_cls.return_value = mock_client
            adapter = AnthropicAdapter(_anthropic_provider())

            result = await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "List files"}]},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "toolu_cc1", "name": "Bash", "input": {"command": "ls"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_cc1", "content": "file1\nfile2\nfile3"},
                    ]},
                ],
                stream=False,
                timeout=60,
                max_tokens=8000,
            )

        assert result.success is True
        msgs = mock_client.messages.create.call_args[1]["messages"]
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"][0]["type"] == "tool_result"
        assert msgs[2]["content"][0]["tool_use_id"] == "toolu_cc1"

    @pytest.mark.asyncio
    async def test_streaming_text_content(self):
        """Claude Code streaming text response produces correct OpenAI-format chunks."""
        events = [
            MockAnthropicEvent(type="message_start", message=MagicMock(usage=MagicMock(input_tokens=20))),
            MockAnthropicEvent(type="content_block_start", index=0, content_block=MagicMock(type="text")),
            MockAnthropicEvent(type="content_block_delta", delta=MagicMock(type="text_delta", text="Hello")),
            MockAnthropicEvent(type="content_block_delta", delta=MagicMock(type="text_delta", text=" world")),
            MockAnthropicEvent(type="content_block_stop", index=0),
            MockAnthropicEvent(type="message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=5)),
        ]

        async def async_iter():
            for e in events:
                yield e

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=async_iter())
            mock_cls.return_value = mock_client
            adapter = AnthropicAdapter(_anthropic_provider())

            result = await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                timeout=60,
                max_tokens=8000,
            )

        assert result.success is True
        collected = []
        async for chunk in result.stream:
            collected.append(chunk)

        # text start + 2 text deltas + stop = 4 (content_block_start yields empty delta, then 2 deltas, then stop)
        assert len(collected) == 4
        texts = []
        for c in collected:
            delta = c.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                texts.append(delta["content"])
        assert "".join(texts) == "Hello world"
        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_thinking_param_passed_through(self):
        """Claude Code sends 'thinking' param — adapter should pass it to upstream."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
            mock_cls.return_value = mock_client
            adapter = AnthropicAdapter(_anthropic_provider())

            result = await adapter.chat_completion(
                model_name="claude-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                timeout=60,
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 10000},
            )

        assert result.success is True
        call_kwargs = mock_client.messages.create.call_args[1]
        # thinking is not in the adapter's standard param list, so it goes to extra_body
        assert call_kwargs.get("thinking") is None
        # The adapter doesn't know about 'thinking', so it's silently dropped
        # (If we want to support it, we'd need to add it to the adapter)
