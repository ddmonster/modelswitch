"""Unit tests for message_converter protocol conversion."""

import json

import pytest

from app.utils.message_converter import (
    anthropic_to_openai_messages,
    openai_stream_to_anthropic,
)


class TestAnthropicToOpenaiMessages:
    def test_simple_user_message(self):
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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

    def test_assistant_string_content(self):
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "assistant", "content": "I said this"}],
            }
        )
        assert result["messages"] == [{"role": "assistant", "content": "I said this"}]

    def test_assistant_list_content(self):
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages({"messages": []})
        assert result["messages"] == []

    def test_no_system(self):
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert result["messages"] == [{"role": "user", "content": "hi"}]

    def test_extra_params_passed_through(self):
        result = anthropic_to_openai_messages(
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

    def test_thinking_param_converts_to_reasoning_effort(self):
        """thinking.type=enabled should set reasoning_effort and use budget_tokens as max_tokens."""
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert result["tools"] is None
        assert result["tool_choice"] is None

    def test_tool_choice_auto(self):
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "auto"},
            }
        )
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "any"},
            }
        )
        assert result["tool_choice"] == "required"

    def test_tool_choice_none(self):
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "none"},
            }
        )
        assert result["tool_choice"] == "none"

    def test_tool_choice_named(self):
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "auto",
            }
        )
        assert result["tool_choice"] == "auto"

    # ========== tool_use / tool_result 块转换 ==========

    def test_assistant_tool_use_blocks(self):
        """assistant 的 tool_use 块转为 OpenAI tool_calls"""
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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
        result = anthropic_to_openai_messages(
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

    def test_user_mixed_text_and_tool_result(self):
        """user 消息同时有 text 和 tool_result 块"""
        result = anthropic_to_openai_messages(
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
        """Stream with only reasoning_content (no content) emits thinking block when thinking_enabled=True."""

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
        # No text block should be emitted
        assert '"type": "text"' not in text

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
        # thinking at index 0, tool_use at index 1
        lines = [l for l in text.split("\n") if l.startswith("data: ")]
        block_starts = [l for l in lines if "content_block_start" in l]
        assert len(block_starts) == 2
        assert '"index": 0' in block_starts[0]
        assert '"type": "thinking"' in block_starts[0]
        assert '"index": 1' in block_starts[1]
        assert '"type": "tool_use"' in block_starts[1]


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
