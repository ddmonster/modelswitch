"""Unit tests for message_converter protocol conversion."""

import json

import pytest

from app.utils.message_converter import (
    anthropic_to_openai_messages,
    openai_stream_to_anthropic,
)


class TestAnthropicToOpenaiMessages:
    def test_simple_user_message(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            }
        )
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 100
        assert result["model"] == "claude-3"

    def test_system_string(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are helpful.",
            }
        )
        assert result["messages"][0] == {
            "role": "system",
            "content": "You are helpful.",
        }
        assert result["messages"][1] == {"role": "user", "content": "hi"}

    def test_system_list_of_blocks(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "system": [
                    {"type": "text", "text": "Part 1."},
                    {"type": "text", "text": "Part 2."},
                ],
            }
        )
        assert result["messages"][0]["role"] == "system"
        assert "Part 1." in result["messages"][0]["content"]
        assert "Part 2." in result["messages"][0]["content"]

    def test_system_list_with_non_text_blocks_ignored(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "system": [
                    {"type": "image", "source": {"data": "abc"}},
                    {"type": "text", "text": "Only this"},
                ],
            }
        )
        assert result["messages"][0]["content"] == "Only this"

    def test_user_content_list_with_text_and_image(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {
                                "type": "image",
                                "source": {
                                    "media_type": "image/png",
                                    "data": "base64data",
                                },
                            },
                        ],
                    }
                ],
            }
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64,base64data" == content[1]["image_url"]["url"]

    def test_user_single_text_block_simplified_to_string(self):
        """Single text block in list should be simplified to string content.
        This ensures compatibility with providers that don't support array content (GLM/BigModel).
        """
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            }
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Content should be a string, not a list
        assert isinstance(msgs[0]["content"], str)
        assert msgs[0]["content"] == "Hello"

    def test_user_single_text_block_with_cache_control_preserved(self):
        """Single text block with cache_control should preserve the marker."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            }
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)
        assert msgs[0]["content"] == "Hello"
        assert msgs[0]["cache_control"] == {"type": "ephemeral"}

    def test_user_multiple_text_blocks_joined_to_string(self):
        """Multiple text blocks (no images) should be joined into string for provider compatibility.

        Some providers (GLM/BigModel) don't support array content format and return
        INVALID_ARGUMENT: content is not repeating, cannot start list.

        This fix ensures multiple text-only blocks are joined into a single string.
        """
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Part one"},
                            {"type": "text", "text": "Part two"},
                            {"type": "text", "text": "Part three"},
                        ],
                    }
                ],
            }
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Content should be a string, not a list
        assert isinstance(msgs[0]["content"], str)
        # All text parts should be joined with spaces
        assert msgs[0]["content"] == "Part one Part two Part three"

    def test_user_multiple_text_blocks_with_cache_control_preserved(self):
        """Multiple text blocks with cache_control should preserve marker from first block."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "First",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": "Second"},
                        ],
                    }
                ],
            }
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0]["content"], str)
        assert msgs[0]["content"] == "First Second"
        # cache_control from first block should be preserved
        assert msgs[0]["cache_control"] == {"type": "ephemeral"}

    def test_assistant_string_content(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "assistant", "content": "I said this"}],
            }
        )
        assert result["messages"] == [{"role": "assistant", "content": "I said this"}]

    def test_assistant_list_content(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Part A"},
                            {"type": "text", "text": "Part B"},
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert "Part A" in msg["content"]
        assert "Part B" in msg["content"]

    def test_empty_messages(self):
        result, tool_name_mapping = anthropic_to_openai_messages({"messages": []})
        assert result["messages"] == []

    def test_no_system(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert result["messages"] == [{"role": "user", "content": "hi"}]

    def test_extra_params_passed_through(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 50,
                "stop_sequences": ["END"],
                "stream": True,
            }
        )
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stop"] == ["END"]
        assert result["stream"] is True

    def test_top_k_not_passed_through(self):
        """P1 fix: top_k is Anthropic-only parameter, not supported by OpenAI providers."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "top_k": 50,  # Anthropic-only parameter
            }
        )
        assert result["temperature"] == 0.7
        assert "top_k" not in result  # Should be filtered out

    def test_thinking_param_converts_to_reasoning_effort(self):
        """thinking.type=enabled should set reasoning_effort and use budget_tokens as max_tokens."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "test",
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "max_tokens": 8192,
            }
        )
        assert result["reasoning_effort"] == "high"
        assert result["max_tokens"] == 10000  # budget_tokens overrides max_tokens

    def test_thinking_param_disabled_ignored(self):
        """thinking.type=disabled should not affect conversion."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "test",
                "thinking": {"type": "disabled"},
                "max_tokens": 8192,
            }
        )
        assert "reasoning_effort" not in result
        assert result["max_tokens"] == 8192

    def test_assistant_thinking_blocks_in_history(self):
        """Prior assistant thinking blocks should be merged into text content."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me reason..."},
                            {"type": "text", "text": "The answer is 42"},
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert "Let me reason" in msg["content"]
        assert "The answer is 42" in msg["content"]

    # ========== Tools 转换 ==========

    def test_tools_conversion(self):
        """Anthropic tools 格式转为 OpenAI 格式"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            }
        )
        assert result["tools"] is not None
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather"
        assert "properties" in tool["function"]["parameters"]

    def test_no_tools_returns_none(self):
        """无 tools 时返回 None"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert result["tools"] is None
        assert result["tool_choice"] is None

    def test_tool_choice_auto(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "auto"},
            }
        )
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "any"},
            }
        )
        assert result["tool_choice"] == "required"

    def test_tool_choice_none(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "none"},
            }
        )
        assert result["tool_choice"] == "none"

    def test_tool_choice_named(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "tool", "name": "get_weather"},
            }
        )
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_choice_string(self):
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "auto",
            }
        )
        assert result["tool_choice"] == "auto"

    # ========== tool_use / tool_result 块转换 ==========

    def test_assistant_tool_use_blocks(self):
        """assistant 的 tool_use 块转为 OpenAI tool_calls"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Let me check"},
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "SF"},
                            },
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "toolu_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "SF"}

    def test_assistant_tool_use_only(self):
        """assistant 只有 tool_use 无文本"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_abc",
                                "name": "run",
                                "input": {},
                            },
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1

    def test_user_tool_result_string_content(self):
        """user 的 tool_result 块（字符串 content）转为 OpenAI role:tool"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "fn",
                                "input": {},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "72°F",
                            },
                        ],
                    },
                ],
            }
        )
        # assistant msg
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["tool_calls"][0]["id"] == "toolu_1"
        # tool result msg
        tool_msg = result["messages"][1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_1"
        assert tool_msg["content"] == "72°F"

    def test_user_tool_result_list_content(self):
        """tool_result 的 content 为列表时提取文本"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_2",
                                "content": [
                                    {"type": "text", "text": "result A"},
                                    {"type": "text", "text": "result B"},
                                ],
                            },
                        ],
                    }
                ],
            }
        )
        tool_msg = result["messages"][0]
        assert tool_msg["role"] == "tool"
        assert "result A" in tool_msg["content"]
        assert "result B" in tool_msg["content"]

    def test_user_tool_result_with_is_error(self):
        """P2 fix: tool_result with is_error=True should prefix content with error indicator."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_3",
                                "content": "Tool execution failed",
                                "is_error": True,
                            },
                        ],
                    }
                ],
            }
        )
        tool_msg = result["messages"][0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_3"
        assert "[ERROR]" in tool_msg["content"]
        assert "Tool execution failed" in tool_msg["content"]

    def test_user_tool_result_without_is_error(self):
        """P2 fix: tool_result with is_error=False (or missing) should not have error prefix."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_4",
                                "content": "Success result",
                                "is_error": False,
                            },
                        ],
                    }
                ],
            }
        )
        tool_msg = result["messages"][0]
        assert "[ERROR]" not in tool_msg["content"]
        assert tool_msg["content"] == "Success result"

    def test_user_empty_text_block_filtered(self):
        """Empty text blocks in user content should be filtered out to prevent API Error 400."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ""},  # Empty - should be filtered
                            {"type": "text", "text": "   "},  # Whitespace-only - should be filtered
                            {"type": "text", "text": "Hello"},  # Valid - should be kept
                        ],
                    }
                ],
            }
        )
        # Only valid text should be in the result
        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str)
        assert msgs[0]["content"] == "Hello"

    def test_assistant_empty_text_block_filtered(self):
        """Empty text blocks in assistant content should be filtered out."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": ""},  # Empty - should be filtered
                            {"type": "text", "text": "  \n  "},  # Whitespace-only - should be filtered
                            {"type": "text", "text": "Valid text"},  # Valid - should be kept
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Valid text"

    def test_assistant_empty_thinking_block_filtered(self):
        """Empty thinking blocks in assistant content should be filtered out."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": ""},  # Empty - should be filtered
                            {"type": "thinking", "thinking": "   "},  # Whitespace-only - should be filtered
                            {"type": "thinking", "thinking": "Let me think"},  # Valid - should be kept
                            {"type": "text", "text": "Answer"},
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert "Let me think" in msg["content"]
        assert "Answer" in msg["content"]
        # Empty thinking blocks should not appear in content

    def test_user_mixed_text_and_tool_result(self):
        """user 消息同时有 text 和 tool_result 块"""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "thanks"},
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_3",
                                "content": "done",
                            },
                        ],
                    }
                ],
            }
        )
        # 应该有 2 条消息：一条 tool（先处理 tool_result），一条 user（text）
        assert len(result["messages"]) == 2
        roles = [m["role"] for m in result["messages"]]
        assert "user" in roles
        assert "tool" in roles


class TestOpenaiStreamToAnthropic:
    @pytest.mark.asyncio
    async def test_full_stream_with_content(self):
        """Test normal stream: message_start -> content -> stop."""

        class Chunk:
            def __init__(self, data):
                self._data = data

            def model_dump(self, **kwargs):
                return self._data

        async def fake_openai_stream():
            yield Chunk(
                {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
            )
            yield Chunk(
                {"choices": [{"delta": {"content": " there"}, "finish_reason": None}]}
            )
            yield Chunk(
                {"choices": [{"delta": {"content": None}, "finish_reason": "stop"}]}
            )

        events = []
        async for event in openai_stream_to_anthropic(
            fake_openai_stream(), "test-model"
        ):
            if isinstance(event, bytes):
                events.append(event.decode())
            else:
                events.append(str(event))

        text = "".join(events)
        assert "message_start" in text
        assert "content_block_start" in text
        assert "content_block_delta" in text
        assert "content_block_stop" in text
        assert "message_delta" in text
        assert "message_stop" in text
        assert "Hi" in text
        assert "there" in text
        assert "end_turn" in text

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Stream with no chunks should still emit message_stop."""

        async def empty_stream():
            return
            yield  # make it a generator

        events = []
        async for event in openai_stream_to_anthropic(empty_stream(), "test-model"):
            events.append(event)

        # Empty stream: no events emitted (no message_start sent)
        assert events == []

    @pytest.mark.asyncio
    async def test_stream_with_length_finish(self):
        """finish_reason='length' maps to 'max_tokens'."""

        class Chunk:
            def __init__(self, data):
                self._data = data

            def model_dump(self, **kwargs):
                return self._data

        async def fake_stream():
            yield Chunk(
                {"choices": [{"delta": {"content": "cut"}, "finish_reason": None}]}
            )
            yield Chunk(
                {"choices": [{"delta": {"content": None}, "finish_reason": "length"}]}
            )

        events = []
        async for event in openai_stream_to_anthropic(fake_stream(), "m"):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        assert "max_tokens" in text

    @pytest.mark.asyncio
    async def test_stream_with_empty_choices(self):
        """Chunks with empty choices list should be skipped."""

        class Chunk:
            def __init__(self, data):
                self._data = data

            def model_dump(self, **kwargs):
                return self._data

        async def fake_stream():
            yield Chunk({"choices": []})
            yield Chunk(
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}
            )
            yield Chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})

        events = []
        async for event in openai_stream_to_anthropic(fake_stream(), "m"):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        assert "ok" in text
        assert "message_stop" in text

    @pytest.mark.asyncio
    async def test_dict_chunks(self):
        """Chunks that are already dicts."""

        async def fake_stream():
            yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = []
        async for event in openai_stream_to_anthropic(fake_stream(), "m"):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        assert "hello" in text

    # ========== Streaming tool_calls ==========

    @pytest.mark.asyncio
    async def test_stream_single_tool_call(self):
        """单个 tool_call 流式转换"""

        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city":'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"SF"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        events = []
        async for event in openai_stream_to_anthropic(fake_stream(), "m"):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        # 应有 tool_use content_block_start
        assert '"tool_use"' in text
        assert "get_weather" in text
        assert "call_1" in text
        # 应有 input_json_delta
        assert "input_json_delta" in text
        # finish_reason 应映射为 tool_use
        assert (
            "tool_use" in text.split("stop_reason")[-1]
            if "stop_reason" in text
            else True
        )

    @pytest.mark.asyncio
    async def test_stream_multiple_tool_calls(self):
        """多个 tool_call 追踪索引"""

        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {"name": "fn1", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"x":1}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {"name": "fn2", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "function": {"arguments": '{"y":2}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        events = []
        async for event in openai_stream_to_anthropic(fake_stream(), "m"):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        assert "fn1" in text
        assert "fn2" in text
        assert "call_a" in text
        assert "call_b" in text
        # 两个 tool_use 块
        assert text.count('"tool_use"') >= 2

    @pytest.mark.asyncio
    async def test_stream_text_then_tool_call(self):
        """先文本后 tool_call"""

        async def fake_stream():
            yield {"choices": [{"delta": {"content": "Let me"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": " check"}, "finish_reason": None}]}
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"q":"test"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        events = []
        async for event in openai_stream_to_anthropic(fake_stream(), "m"):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        assert "Let me" in text
        assert "check" in text
        assert '"tool_use"' in text
        assert "search" in text
        # 文本块应在 index 0，tool 块应在 index 1
        assert '"index": 0' in text
        assert '"index": 1' in text

    @pytest.mark.asyncio
    async def test_reasoning_content_emits_thinking_block(self):
        """reasoning_content in delta should emit thinking type content blocks when thinking_enabled=True."""

        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Let me think",
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {"reasoning_content": " about this..."},
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {"delta": {"content": "The answer is 42"}, "finish_reason": None}
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "test-model", thinking_enabled=True
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        # thinking block emitted
        assert '"type": "thinking"' in text
        assert '"thinking_delta"' in text
        assert "Let me think" in text
        assert "about this..." in text
        # text block emitted at index 1 (after thinking at index 0)
        assert '"type": "text_delta"' in text
        assert "The answer is 42" in text
        # correct block ordering: thinking at 0, text at 1
        lines = [l for l in text.split("\n") if l.startswith("data: ")]
        block_starts = [l for l in lines if "content_block_start" in l]
        assert len(block_starts) == 2  # thinking + text
        assert '"index": 0' in block_starts[0]  # thinking first
        assert '"index": 1' in block_starts[1]  # text second

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_emits_thinking(self):
        """Stream with only reasoning_content (no content) emits thinking block when thinking_enabled=True.

        C5 fix: Also emits an empty text block to prevent Anthropic SDK client errors when
        clients expect at least one text block in every message.
        """

        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Thinking...",
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "m", thinking_enabled=True
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        assert '"type": "thinking"' in text
        assert "Thinking..." in text
        # C5 fix: Empty text block should also be emitted to prevent client SDK errors
        assert '"type": "text"' in text
        assert '"text": ""' in text

    @pytest.mark.asyncio
    async def test_thinking_then_tool_call_indexing(self):
        """thinking(0) + tool(1), no text block — requires thinking_enabled=True."""

        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {"reasoning_content": "Need to search"},
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"q":"x"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "m", thinking_enabled=True
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        # C5 fix: thinking(0) + empty_text(1) + tool_use(2)
        # Empty text block inserted BEFORE tool_use for Anthropic protocol compliance
        # (Anthropic expects: thinking -> text -> tool_use order)
        lines = [l for l in text.split("\n") if l.startswith("data: ")]
        block_starts = [l for l in lines if "content_block_start" in l]
        assert len(block_starts) == 3
        assert '"index": 0' in block_starts[0]
        assert '"type": "thinking"' in block_starts[0]
        # C5 fix: Empty text block at index 1 (before tool_use)
        assert '"index": 1' in block_starts[1]
        assert '"type": "text"' in block_starts[1]
        assert '"text": ""' in block_starts[1]
        # tool_use at index 2 (after empty text)
        assert '"index": 2' in block_starts[2]
        assert '"type": "tool_use"' in block_starts[2]

    @pytest.mark.asyncio
    async def test_tool_use_only_without_text(self):
        """C5 fix: tool_use without thinking or text needs empty text block inserted."""

        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"q":"test"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "m", thinking_enabled=False
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        lines = [l for l in text.split("\n") if l.startswith("data: ")]
        block_starts = [l for l in lines if "content_block_start" in l]
        # C5 fix: empty text(0) + tool_use(1)
        assert len(block_starts) == 2
        assert '"index": 0' in block_starts[0]
        assert '"type": "text"' in block_starts[0]
        assert '"text": ""' in block_starts[0]
        assert '"index": 1' in block_starts[1]
        assert '"type": "tool_use"' in block_starts[1]

    @pytest.mark.asyncio
    async def test_thinking_block_without_signature(self):
        """P9 fix: thinking block from OpenAI provider should not have signature field."""
        async def fake_stream():
            yield {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Let me think...",
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {"content": "The answer"},
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "m", thinking_enabled=True
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        # Thinking block should be present
        assert '"type": "thinking"' in text
        # Content block should NOT contain signature field (OpenAI providers don't provide it)
        # Check that the content_block for thinking doesn't include "signature"
        import json
        # Parse the content_block_start event for thinking
        lines = [l for l in text.split("\n") if "content_block_start" in l]
        for line in lines:
            if '"type": "thinking"' in line:
                # Extract the data
                data_str = line.split("data: ")[1]
                data = json.loads(data_str)
                content_block = data.get("content_block", {})
                # Signature should be absent for OpenAI providers
                assert "signature" not in content_block


class TestConvertOpenaiToAnthropicResponse:
    """测试 convert_openai_to_anthropic_response 的 tool_calls 转换"""

    def _call(self, resp_data, model="test", thinking_enabled=False):
        from app.utils.message_converter import convert_openai_to_anthropic_response

        return convert_openai_to_anthropic_response(
            resp_data, model, thinking_enabled=thinking_enabled
        )

    def test_text_only_response(self):
        resp = self._call(
            {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        assert resp["content"] == [{"type": "text", "text": "Hello"}]
        assert resp["stop_reason"] == "end_turn"

    def test_tool_calls_response(self):
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"SF"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }
        )
        # 应有 tool_use content block
        tool_blocks = [b for b in resp["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "call_abc"
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "SF"}
        assert resp["stop_reason"] == "tool_use"

    def test_tool_calls_with_text(self):
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Checking now",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "fn", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
        text_blocks = [b for b in resp["content"] if b["type"] == "text"]
        tool_blocks = [b for b in resp["content"] if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert len(tool_blocks) == 1
        assert resp["stop_reason"] == "tool_use"

    def test_length_finish(self):
        resp = self._call(
            {
                "choices": [
                    {"message": {"content": "cut off"}, "finish_reason": "length"}
                ],
            }
        )
        assert resp["stop_reason"] == "max_tokens"

    def test_no_choices(self):
        resp = self._call({"choices": [], "usage": {}})
        # C3/C4 fix: 空响应应保证至少一个 text 块
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == ""
        assert resp["stop_reason"] == "end_turn"

    def test_reasoning_content_emits_thinking_block(self):
        """thinking_enabled=True 时，reasoning_content 应生成 thinking 块。"""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The answer is 42",
                            "reasoning_content": "Let me think about this...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            thinking_enabled=True,
        )
        assert resp["content"][0]["type"] == "thinking"
        assert resp["content"][0]["thinking"] == "Let me think about this..."
        assert resp["content"][1]["type"] == "text"
        assert resp["content"][1]["text"] == "The answer is 42"

    def test_reasoning_content_only(self):
        """thinking_enabled=True 时，纯 reasoning 应生成 thinking 块。"""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "Deep thoughts",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            thinking_enabled=True,
        )
        thinking_blocks = [b for b in resp["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Deep thoughts"

    def test_reasoning_content_merged_when_thinking_disabled(self):
        """C2/C3 fix: thinking_enabled=False 时，reasoning_content 合并到 text 块。"""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The answer is 42",
                            "reasoning_content": "Let me think...",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        # 不应有 thinking 块
        thinking_blocks = [b for b in resp["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) == 0
        # reasoning 应合并到 text 块
        text_blocks = [b for b in resp["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert "Let me think..." in text_blocks[0]["text"]
        assert "The answer is 42" in text_blocks[0]["text"]

    def test_reasoning_only_merged_when_thinking_disabled(self):
        """C3 fix: thinking_enabled=False 且无 content 时，reasoning 合并为 text 块（不产生空 content）。"""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "Deep thoughts only",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        thinking_blocks = [b for b in resp["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) == 0
        text_blocks = [b for b in resp["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Deep thoughts only"

    def test_empty_content_gets_fallback_text_block(self):
        """M4 fix: content=None 且无 reasoning/tool_calls 时，应有空 text 块。"""
        resp = self._call(
            {
                "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            }
        )
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"

    def test_whitespace_content_filtered(self):
        """Whitespace-only content should be filtered out to prevent API Error 400."""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {"content": "   \n\t  "},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        # Should have fallback empty text block (no other content)
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"

    def test_whitespace_reasoning_filtered(self):
        """Whitespace-only reasoning should be filtered out."""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "   \n  ",  # Whitespace-only
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            thinking_enabled=True,
        )
        # Should only have text block, no thinking block (filtered)
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == "Answer"

    def test_empty_reasoning_filtered(self):
        """Empty reasoning_content should be filtered out."""
        resp = self._call(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hello",
                            "reasoning_content": "",  # Empty
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            thinking_enabled=True,
        )
        # Should only have text block
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"


# ========== New Features Tests ==========


class TestOSeriesModels:
    """Tests for o-series model handling."""

    def test_o_series_detection(self):
        """o-series models should be detected correctly."""
        from app.utils.message_converter import is_openai_o_series

        assert is_openai_o_series("o1") is True
        assert is_openai_o_series("o1-preview") is True
        assert is_openai_o_series("o1-mini") is True
        assert is_openai_o_series("o3") is True
        assert is_openai_o_series("o3-mini") is True
        assert is_openai_o_series("o4-mini") is True
        assert is_openai_o_series("gpt-4o") is False
        assert is_openai_o_series("claude-3") is False

    def test_o_series_uses_max_completion_tokens(self):
        """o-series models should use max_completion_tokens instead of max_tokens."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "o3-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
            }
        )
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 4096

    def test_non_o_series_keeps_max_tokens(self):
        """Non o-series models should keep max_tokens."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
            }
        )
        assert result["max_tokens"] == 1024
        assert "max_completion_tokens" not in result

    def test_o_series_with_budget_tokens(self):
        """o-series with thinking.budget_tokens should use max_completion_tokens."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "o3",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled", "budget_tokens": 10000},
            }
        )
        assert result["max_completion_tokens"] == 10000


class TestReasoningEffortResolution:
    """Tests for reasoning_effort resolution."""

    def test_output_config_effort_high(self):
        """output_config.effort should map to reasoning_effort."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"output_config": {"effort": "high"}}
        assert resolve_reasoning_effort(body) == "high"

    def test_output_config_effort_max(self):
        """output_config.effort=max should map to xhigh."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"output_config": {"effort": "max"}}
        assert resolve_reasoning_effort(body) == "xhigh"

    def test_output_config_takes_priority(self):
        """output_config.effort should take priority over thinking."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {
            "output_config": {"effort": "low"},
            "thinking": {"type": "adaptive"},
        }
        assert resolve_reasoning_effort(body) == "low"

    def test_thinking_adaptive_maps_medium(self):
        """thinking.type=adaptive should map to medium (LiteLLM-aligned behavior)."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "adaptive"}}
        assert resolve_reasoning_effort(body) == "medium"

    def test_budget_small_maps_low(self):
        """Small budget_tokens should map to low effort."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 2000}}
        assert resolve_reasoning_effort(body) == "low"

    def test_budget_medium_maps_medium(self):
        """Medium budget_tokens should map to medium effort."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
        assert resolve_reasoning_effort(body) == "medium"

    def test_budget_large_maps_high(self):
        """Large budget_tokens should map to high effort."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 20000}}
        assert resolve_reasoning_effort(body) == "high"

    def test_gpt5_with_output_config(self):
        """GPT-5+ models should use reasoning_effort from output_config."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"effort": "medium"},
            }
        )
        assert result["reasoning_effort"] == "medium"


class TestCacheControlPreservation:
    """Tests for cache_control preservation."""

    def test_cache_control_on_system(self):
        """cache_control on system blocks should be preserved."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "system": [
                    {"type": "text", "text": "Be helpful", "cache_control": {"type": "ephemeral"}},
                ],
            }
        )
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_on_tools(self):
        """cache_control on tools should be preserved."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "test",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        )
        assert result["tools"][0]["cache_control"] == {"type": "ephemeral"}

    def test_batchtool_filtered(self):
        """BatchTool should be filtered from tools."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"name": "BatchTool", "type": "BatchTool"},
                    {"name": "real_tool", "input_schema": {"type": "object"}},
                ],
            }
        )
        # After filtering BatchTool, only real_tool remains
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "real_tool"


class TestRefusalHandling:
    """Tests for refusal block handling in responses."""

    def test_refusal_in_content_parts(self):
        """Refusal in content parts array should convert to text."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "refusal", "refusal": "I can't help"},
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            "test",
        )
        text_blocks = [b for b in resp["content"] if b["type"] == "text"]
        assert len(text_blocks) == 2
        assert "I can't help" in [b["text"] for b in text_blocks]

    def test_message_level_refusal(self):
        """Message-level refusal should convert to text."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {"content": None, "refusal": "Content blocked"},
                        "finish_reason": "stop",
                    }
                ],
            },
            "test",
        )
        text_blocks = [b for b in resp["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Content blocked"

    def test_content_filter_finish_reason(self):
        """content_filter finish_reason should map to end_turn."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {"content": "Blocked"},
                        "finish_reason": "content_filter",
                    }
                ],
            },
            "test",
        )
        assert resp["stop_reason"] == "end_turn"


class TestLegacyFunctionCall:
    """Tests for legacy function_call format."""

    def test_legacy_function_call(self):
        """Legacy function_call should convert to tool_use."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
                            },
                        },
                        "finish_reason": "function_call",
                    }
                ],
            },
            "test",
        )
        assert resp["content"][0]["type"] == "text"  # C5 fix: empty text block before tool_use
        assert resp["content"][0]["text"] == ""
        assert resp["content"][1]["type"] == "tool_use"  # C5 fix: tool_use at index 1
        assert resp["content"][1]["name"] == "get_weather"
        assert resp["content"][1]["input"]["city"] == "Tokyo"
        assert resp["stop_reason"] == "tool_use"


class TestCacheTokenMapping:
    """Tests for cache token mapping in responses."""

    def test_cached_tokens_mapped(self):
        """prompt_tokens_details.cached_tokens should map to cache_read_input_tokens."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
            "test",
        )
        assert resp["usage"]["input_tokens"] == 100
        assert resp["usage"]["output_tokens"] == 50
        assert resp["usage"]["cache_read_input_tokens"] == 80

    def test_direct_cache_fields(self):
        """Direct cache fields should be preserved."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cache_read_input_tokens": 60,
                    "cache_creation_input_tokens": 20,
                },
            },
            "test",
        )
        assert resp["usage"]["cache_read_input_tokens"] == 60
        assert resp["usage"]["cache_creation_input_tokens"] == 20


class TestPreserveThinkingBlocks:
    """Tests for preserve_thinking_blocks feature."""

    # ========== Request conversion tests ==========

    def test_assistant_thinking_preserved_in_request(self):
        """When preserve_thinking_blocks=True, thinking should be emitted as reasoning_content."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me reason..."},
                            {"type": "text", "text": "The answer is 42"},
                        ],
                    }
                ],
                "thinking": {"type": "enabled"},
            },
            preserve_thinking_blocks=True,
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["reasoning_content"] == "Let me reason..."
        assert msg["content"] == "The answer is 42"

    def test_assistant_thinking_preserved_backward_compatible(self):
        """When preserve_thinking_blocks=False (default), thinking should be merged into text."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me reason..."},
                            {"type": "text", "text": "The answer is 42"},
                        ],
                    }
                ],
            },
            preserve_thinking_blocks=False,
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        # thinking should be merged into content
        assert "Let me reason" in msg["content"]
        assert "The answer is 42" in msg["content"]
        assert "reasoning_content" not in msg

    def test_assistant_thinking_only_preserved(self):
        """When preserve_thinking_blocks=True and only thinking content."""
        result, tool_name_mapping = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Just thinking..."},
                        ],
                    }
                ],
                "thinking": {"type": "enabled"},
            },
            preserve_thinking_blocks=True,
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["reasoning_content"] == "Just thinking..."
        assert msg["content"] is None

    # ========== Response conversion tests ==========

    def test_response_thinking_preserved_with_flag(self):
        """When preserve_thinking_blocks=True, reasoning should be separate thinking block."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The answer is 42",
                            "reasoning_content": "Let me think...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            "test",
            preserve_thinking_blocks=True,
        )
        # Should have thinking block first, then text block
        assert resp["content"][0]["type"] == "thinking"
        assert resp["content"][0]["thinking"] == "Let me think..."
        assert resp["content"][1]["type"] == "text"
        assert resp["content"][1]["text"] == "The answer is 42"

    def test_response_thinking_preserved_backward_compatible(self):
        """When preserve_thinking_blocks=False, reasoning should be merged into text."""
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The answer is 42",
                            "reasoning_content": "Let me think...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            "test",
            preserve_thinking_blocks=False,
        )
        # Should have single text block with merged content
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "text"
        assert "Let me think..." in resp["content"][0]["text"]
        assert "The answer is 42" in resp["content"][0]["text"]

    def test_response_reasoning_only_preserved(self):
        """When preserve_thinking_blocks=True and only reasoning.

        C5 fix: Also emits an empty text block to prevent Anthropic SDK client errors.
        """
        from app.utils.message_converter import convert_openai_to_anthropic_response

        resp = convert_openai_to_anthropic_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "Deep thoughts",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            "test",
            preserve_thinking_blocks=True,
        )
        assert resp["content"][0]["type"] == "thinking"
        assert resp["content"][0]["thinking"] == "Deep thoughts"
        # C5 fix: Empty text block should also be present
        assert len(resp["content"]) == 2
        assert resp["content"][1]["type"] == "text"
        assert resp["content"][1]["text"] == ""

    # ========== Streaming tests ==========

    @pytest.mark.asyncio
    async def test_stream_thinking_preserved(self):
        """When preserve_thinking_blocks=True, reasoning should emit thinking_delta."""
        async def fake_stream():
            yield {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]}
            yield {"choices": [{"delta": {"content": "Answer"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "m", preserve_thinking_blocks=True
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        # Should have thinking block and text block
        assert '"type": "thinking"' in text
        assert '"type": "thinking_delta"' in text
        assert '"type": "text"' in text
        assert '"type": "text_delta"' in text

    @pytest.mark.asyncio
    async def test_stream_thinking_preserved_backward_compatible(self):
        """When preserve_thinking_blocks=False, reasoning should be merged into content."""
        async def fake_stream():
            yield {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]}
            yield {"choices": [{"delta": {"content": "Answer"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = []
        async for event in openai_stream_to_anthropic(
            fake_stream(), "m", preserve_thinking_blocks=False
        ):
            events.append(event if isinstance(event, bytes) else event.encode())

        text = b"".join(events).decode()
        # Should NOT have thinking block - reasoning merged into text
        assert '"type": "thinking"' not in text
        assert '"type": "thinking_delta"' not in text
        # Text should contain both reasoning and content
        assert "Thinking..." in text
        assert "Answer" in text


class TestToolNameTruncation:
    """Tests for tool name truncation (OpenAI 64-char limit)."""

    def test_short_tool_name_not_truncated(self):
        """Tool names under 64 chars should not be truncated."""
        from app.utils.message_converter import truncate_tool_name

        assert truncate_tool_name("get_weather") == "get_weather"
        assert truncate_tool_name("short_tool") == "short_tool"

    def test_long_tool_name_truncated(self):
        """Tool names over 64 chars should be truncated with hash."""
        from app.utils.message_converter import truncate_tool_name, OPENAI_MAX_TOOL_NAME_LENGTH

        long_name = "this_is_a_very_long_tool_name_that_exceeds_the_openai_limit_of_64_characters"
        truncated = truncate_tool_name(long_name)
        assert len(truncated) == OPENAI_MAX_TOOL_NAME_LENGTH
        assert truncated != long_name
        # Should contain underscore separator for hash
        assert "_" in truncated[-9:]  # hash is last 8 chars + underscore

    def test_tool_name_mapping_created(self):
        """Long tool names should be added to mapping."""
        from app.utils.message_converter import create_tool_name_mapping, truncate_tool_name

        long_name = "very_long_tool_name_exceeding_sixty_four_characters_for_testing_purposes"
        tools = [{"name": "short"}, {"name": long_name}]
        mapping = create_tool_name_mapping(tools)
        # Only long name should be in mapping
        assert "short" not in mapping
        truncated = truncate_tool_name(long_name)
        assert mapping[truncated] == long_name

    def test_tools_with_long_names_converted(self):
        """Tools with long names should be truncated in conversion."""
        long_name = "this_is_a_very_long_tool_name_that_exceeds_sixty_four_characters_limit"
        result, mapping = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"name": "short_tool", "input_schema": {"type": "object"}},
                    {"name": long_name, "input_schema": {"type": "object"}},
                ],
            }
        )
        assert len(result["tools"]) == 2
        # First tool should not be truncated
        assert result["tools"][0]["function"]["name"] == "short_tool"
        # Second tool should be truncated
        truncated_name = result["tools"][1]["function"]["name"]
        assert len(truncated_name) <= 64
        assert truncated_name != long_name
        # Mapping should restore original name
        assert mapping[truncated_name] == long_name


class TestMetadataUserId:
    """Tests for metadata.user_id -> user field mapping."""

    def test_metadata_user_id_mapped(self):
        """metadata.user_id should be mapped to user field."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": "test_user_123"},
            }
        )
        assert result["user"] == "test_user_123"

    def test_metadata_user_id_truncated(self):
        """metadata.user_id should be truncated to 64 chars."""
        long_id = "a" * 100  # 100 characters
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": long_id},
            }
        )
        assert result["user"] == long_id[:64]
        assert len(result["user"]) == 64

    def test_no_metadata_user_id_not_mapped(self):
        """No metadata.user_id should not add user field."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert "user" not in result


class TestOutputFormat:
    """Tests for output_format/output_config structured output support."""

    def test_output_format_json_schema(self):
        """output_format json_schema should be mapped to response_format."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_format": {"type": "json_schema", "schema": schema},
            }
        )
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"
        assert result["response_format"]["json_schema"]["schema"] == schema
        assert result["response_format"]["json_schema"]["strict"] is True

    def test_output_config_format_json_schema(self):
        """output_config.format json_schema should be mapped to response_format."""
        schema = {"type": "object", "properties": {"result": {"type": "number"}}}
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }
        )
        assert "response_format" in result
        assert result["response_format"]["json_schema"]["schema"] == schema


class TestReasoningEffortThresholds:
    """Tests for LiteLLM-aligned reasoning effort thresholds."""

    def test_budget_minimal(self):
        """Budget < 2000 should map to minimal."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 1000}}
        assert resolve_reasoning_effort(body) == "minimal"

    def test_budget_low(self):
        """Budget 2000-5000 should map to low."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 3000}}
        assert resolve_reasoning_effort(body) == "low"

    def test_budget_medium_threshold(self):
        """Budget 5000-10000 should map to medium."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 7000}}
        assert resolve_reasoning_effort(body) == "medium"

    def test_budget_high_threshold(self):
        """Budget >= 10000 should map to high."""
        from app.utils.message_converter import resolve_reasoning_effort

        body = {"thinking": {"type": "enabled", "budget_tokens": 10000}}
        assert resolve_reasoning_effort(body) == "high"

    def test_reasoning_summary_detailed(self):
        """Enabled thinking should default to detailed summary."""
        from app.utils.message_converter import resolve_reasoning_summary

        body = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
        assert resolve_reasoning_summary(body) == "detailed"

    def test_reasoning_summary_explicit(self):
        """Explicit summary should be preserved."""
        from app.utils.message_converter import resolve_reasoning_summary

        body = {"thinking": {"type": "enabled", "summary": "auto"}}
        assert resolve_reasoning_summary(body) == "auto"


class TestClaudeCode2xCompatibility:
    """Tests for Claude Code 2.x edge case handling.

    Claude Code 2.x uses structured content blocks + tool use by default.
    This test class ensures modelswitch handles these cases properly without
    breaking compatibility with older clients that send plain string content.
    """

    def test_assistant_mixed_text_and_tool_use(self):
        """Claude Code 2.x sends assistant messages with text + tool_use mixed."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I'll check the repository"},
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "bash",
                                "input": {"cmd": "git log --oneline"},
                            },
                        ],
                    }
                ],
            }
        )
        # Should have single assistant message with content + tool_calls
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I'll check the repository"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "bash"

    def test_assistant_only_tool_use(self):
        """Claude Code 2.x may send assistant messages with only tool_use."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_abc",
                                "name": "read_file",
                                "input": {"path": "/src/main.py"},
                            },
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1

    def test_user_message_with_tool_use_edge_case(self):
        """Claude Code 2.x may send tool_use in user messages in agent workflows.

        We handle this by creating a synthetic assistant message with tool_calls
        to maintain proper conversation flow in OpenAI format.
        """
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please run this command"},
                            {
                                "type": "tool_use",
                                "id": "toolu_xyz",
                                "name": "bash",
                                "input": {"cmd": "ls -la"},
                            },
                        ],
                    }
                ],
            }
        )
        # Should have synthetic assistant message with tool_calls + user message
        assert len(result["messages"]) == 2
        # First message: synthetic assistant with tool_calls
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["tool_calls"][0]["function"]["name"] == "bash"
        # Second message: user with text
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Please run this command"

    def test_full_tool_workflow_claude_code_2x(self):
        """Full Claude Code 2.x workflow: assistant tool_use -> user tool_result."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Checking the git history"},
                            {
                                "type": "tool_use",
                                "id": "toolu_001",
                                "name": "bash",
                                "input": {"cmd": "git log --oneline -10"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_001",
                                "content": "abc123 First commit\ndef456 Second commit",
                            },
                        ],
                    },
                ],
            }
        )
        # Message 0: assistant with content + tool_calls
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["content"] == "Checking the git history"
        assert len(result["messages"][0]["tool_calls"]) == 1
        # Message 1: tool result
        assert result["messages"][1]["role"] == "tool"
        assert result["messages"][1]["tool_call_id"] == "toolu_001"

    def test_backward_compatible_plain_string_content(self):
        """Older clients sending plain string content should still work."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well!"},
                ],
            }
        )
        # Should work exactly as before
        assert result["messages"][0] == {"role": "user", "content": "Hello, how are you?"}
        assert result["messages"][1] == {"role": "assistant", "content": "I'm doing well!"}

    def test_assistant_with_redacted_thinking(self):
        """Claude Code 2.x may include redacted_thinking blocks."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "thinking": "[REDACTED]"},
                            {"type": "text", "text": "Here's my answer"},
                        ],
                    }
                ],
            }
        )
        # redacted_thinking should be skipped, only text preserved
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Here's my answer"
        assert "tool_calls" not in msg

    def test_multiple_tool_use_blocks_in_assistant(self):
        """Claude Code 2.x may call multiple tools in one message."""
        result, _ = anthropic_to_openai_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I'll run both commands"},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "bash",
                                "input": {"cmd": "git status"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_2",
                                "name": "read_file",
                                "input": {"path": "README.md"},
                            },
                        ],
                    }
                ],
            }
        )
        msg = result["messages"][0]
        assert msg["content"] == "I'll run both commands"
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "bash"
        assert msg["tool_calls"][1]["function"]["name"] == "read_file"
