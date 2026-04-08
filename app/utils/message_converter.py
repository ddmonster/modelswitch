from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator


def anthropic_to_openai_messages(data: dict) -> dict:
    """将 Anthropic Messages API 请求体转换为 OpenAI 格式"""
    messages = []

    # Anthropic 的 system 是顶层字段
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                messages.append({"role": "system", "content": " ".join(text_parts)})

    # 转换 tools: Anthropic -> OpenAI
    openai_tools = None
    if "tools" in data:
        openai_tools = []
        for tool in data["tools"]:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )

    # 转换 tool_choice: Anthropic -> OpenAI
    openai_tool_choice = None
    if "tool_choice" in data:
        tc = data["tool_choice"]
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "auto":
                openai_tool_choice = "auto"
            elif tc_type == "any":
                openai_tool_choice = "required"
            elif tc_type == "none":
                openai_tool_choice = "none"
            elif tc_type == "tool":
                openai_tool_choice = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }
        elif isinstance(tc, str):
            openai_tool_choice = tc

    # 转换 messages
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # H2 fix: collect tool_results separately, append after user text
                converted = []
                tool_results = []
                for block in content:
                    if block.get("type") == "text":
                        converted.append(
                            {"type": "text", "text": block.get("text", "")}
                        )
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        data_url = (
                            f"data:{source.get('media_type', 'image/png')};base64,"
                            f"{source.get('data', '')}"
                        )
                        converted.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        )
                    elif block.get("type") == "tool_result":
                        # tool_result -> OpenAI role: "tool" 消息
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                b.get("text", "")
                                for b in result_content
                                if b.get("type") == "text"
                            )
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(result_content)
                                if result_content
                                else "",
                            }
                        )
                # H2 fix: user text first, then tool results
                if converted:
                    messages.append({"role": "user", "content": converted})
                messages.extend(tool_results)
        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content if b.get("type") == "text"
                ]
                thinking_parts = [
                    b.get("thinking", "")
                    for b in content
                    if b.get("type") == "thinking"
                ]
                tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

                # 将 thinking 内容合并到 text 前面（OpenAI 不支持 thinking 块）
                all_text_parts = []
                if thinking_parts:
                    all_text_parts.extend(thinking_parts)
                all_text_parts.extend(text_parts)

                if tool_use_blocks:
                    tool_calls = []
                    for b in tool_use_blocks:
                        tool_calls.append(
                            {
                                "id": b.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": b.get("name", ""),
                                    "arguments": json.dumps(
                                        b.get("input", {}), ensure_ascii=False
                                    ),
                                },
                            }
                        )
                    text_content = " ".join(all_text_parts) if all_text_parts else None
                    messages.append(
                        {
                            "role": "assistant",
                            "content": text_content,
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    messages.append(
                        {"role": "assistant", "content": " ".join(all_text_parts)}
                    )

    result = {
        "model": data.get("model"),
        "messages": messages,
        "max_tokens": data.get("max_tokens"),
        "temperature": data.get("temperature"),
        "top_p": data.get("top_p"),
        "top_k": data.get("top_k"),
        "stream": data.get("stream", False),
        "stop": data.get("stop_sequences"),
        "tools": openai_tools,
        "tool_choice": openai_tool_choice,
    }

    # 转发 thinking 参数到 extra_body（OpenAI 兼容 provider 可能支持）
    thinking = data.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        result["reasoning_effort"] = "high"
        if thinking.get("budget_tokens"):
            result["max_tokens"] = thinking["budget_tokens"]

    return result


def convert_openai_to_anthropic_response(
    resp_data: dict, model: str, thinking_enabled: bool = False
) -> dict:
    """将 OpenAI ChatCompletion 响应转换为 Anthropic Messages 响应。

    Args:
        resp_data: OpenAI 格式的响应 dict
        model: 模型名称
        thinking_enabled: 客户端是否请求了 thinking（决定是否生成 thinking 块）
    """
    choices = resp_data.get("choices", [])
    content = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        reasoning = message.get("reasoning_content")
        msg_content = message.get("content", "")

        # C2/C3 fix: 根据 thinking_enabled 决定如何处理 reasoning_content
        if reasoning and thinking_enabled:
            content.append({"type": "thinking", "thinking": reasoning})

        # 确定要输出的文本内容
        text_to_emit = msg_content
        if not thinking_enabled and reasoning:
            # C3: thinking 未启用时，将 reasoning 合并到 text 块
            if msg_content:
                text_to_emit = reasoning + msg_content
            else:
                text_to_emit = reasoning

        # M4 fix: content 为空字符串时也要生成 text 块（用 is not None 判断）
        if text_to_emit is not None:
            content.append({"type": "text", "text": text_to_emit})

        # 处理 tool_calls
        for tc in message.get("tool_calls", []):
            tc_func = tc.get("function", {})
            try:
                tc_input = json.loads(tc_func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tc_input = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": tc_func.get("name", ""),
                    "input": tc_input,
                }
            )

        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }.get(finish_reason, "end_turn")

    # C3 fix: 如果 content 仍为空，添加空 text 块保证协议合规
    if not content:
        content.append({"type": "text", "text": ""})

    usage = resp_data.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse(event_type: str, data: dict) -> bytes:
    """生成 Anthropic SSE 事件"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _to_dict(obj):
    """深度转换对象为 dict，处理 Pydantic 模型嵌套未完全序列化的情况（如 BigModel SDK 的 ChoiceDelta）"""
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_dict(obj.model_dump(exclude_none=True))
    if hasattr(obj, "to_dict"):
        return _to_dict(obj.to_dict())
    return obj


async def openai_stream_to_anthropic(
    openai_stream: AsyncGenerator,
    model: str,
    request_id: str = "",
    thinking_enabled: bool = False,
) -> AsyncGenerator[bytes, None]:
    """将 OpenAI 格式的 SSE 流实时转换为 Anthropic 格式。

    Args:
        openai_stream: OpenAI 格式的流式 chunk 生成器
        model: 模型名称
        request_id: 请求 ID
        thinking_enabled: 客户端是否请求了 thinking。
            False 时，reasoning_content 会被合并到 text 块中，不生成 thinking 块。

    状态机: message_start -> [content_block_start -> delta* -> stop]* -> message_delta -> message_stop

    H1 fix: 使用递增计数器 next_block_index 分配块索引，
    不再基于 thinking/text 标志计算，避免 tool 先于 text 时索引冲突。
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    sent_message_start = False
    total_output_tokens = 0
    finish_reason = "end_turn"

    # H1 fix: 用递增计数器代替计算式索引
    next_block_index = 0  # 下一个可用的块索引
    open_block_index = -1  # 当前打开的块索引，-1 表示无
    text_block_opened = False
    thinking_block_opened = False
    tool_calls_map = {}  # {openai_tc_index: {"id", "name", "block_index"}}

    def _close_open_block():
        nonlocal open_block_index
        if open_block_index >= 0:
            idx = open_block_index
            open_block_index = -1
            return _sse(
                "content_block_stop", {"type": "content_block_stop", "index": idx}
            )
        return None

    try:
        async for chunk in openai_stream:
            chunk_data = _to_dict(chunk)
            if not isinstance(chunk_data, dict):
                continue

            choices = chunk_data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")
            reasoning_content = delta.get("reasoning_content")
            finish = choices[0].get("finish_reason")

            # C2 fix: thinking 未启用时，将 reasoning_content 合并到 content
            if reasoning_content is not None and not thinking_enabled:
                if content is None:
                    content = reasoning_content
                else:
                    content = reasoning_content + content
                reasoning_content = None

            # 生成 message_start（首次有内容的 chunk）
            if not sent_message_start:
                sent_message_start = True
                yield _sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )

            # 处理 tool_calls delta
            tool_calls_deltas = delta.get("tool_calls")
            if tool_calls_deltas:
                for tc_delta in tool_calls_deltas:
                    tc_index = tc_delta.get("index", 0)

                    if tc_index not in tool_calls_map:
                        # 关闭之前的块
                        close_ev = _close_open_block()
                        if close_ev:
                            yield close_ev

                        # H1 fix: 使用 next_block_index 而非计算式
                        block_idx = next_block_index
                        next_block_index += 1

                        tc_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        func = tc_delta.get("function", {})
                        tc_name = func.get("name", "")

                        tool_calls_map[tc_index] = {
                            "id": tc_id,
                            "name": tc_name,
                            "block_index": block_idx,
                        }
                        open_block_index = block_idx

                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc_id,
                                    "name": tc_name,
                                    "input": {},
                                },
                            },
                        )

                    # 发送 arguments delta
                    func = tc_delta.get("function", {})
                    args_fragment = func.get("arguments", "")
                    if args_fragment:
                        block_idx = tool_calls_map[tc_index]["block_index"]
                        open_block_index = block_idx
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args_fragment,
                                },
                            },
                        )

            # 处理 reasoning_content -> thinking 块（仅 thinking_enabled 时）
            if reasoning_content is not None:
                if not thinking_block_opened:
                    thinking_block_opened = True
                    # H1 fix: 使用 next_block_index
                    block_idx = next_block_index
                    next_block_index += 1
                    open_block_index = block_idx
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )

                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block_index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": reasoning_content,
                        },
                    },
                )

            # 处理文本 content
            if content is not None:
                if not text_block_opened:
                    text_block_opened = True
                    close_ev = _close_open_block()
                    if close_ev:
                        yield close_ev
                    # H1 fix: 使用 next_block_index
                    block_idx = next_block_index
                    next_block_index += 1
                    open_block_index = block_idx
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )

                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block_index,
                        "delta": {"type": "text_delta", "text": content},
                    },
                )
                total_output_tokens += 1

            # 检查 finish_reason
            if finish:
                finish_reason = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }.get(finish, "end_turn")

    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"openai_stream_to_anthropic error: {e}")
    finally:
        # 关闭仍打开的块
        close_ev = _close_open_block()
        if close_ev:
            yield close_ev

        if sent_message_start:
            # C4 fix: 如果没有任何块被打开，合成一个空 text 块保证协议合规
            if next_block_index == 0:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield _sse(
                    "content_block_stop", {"type": "content_block_stop", "index": 0}
                )

            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": finish_reason, "stop_sequence": None},
                    "usage": {"output_tokens": total_output_tokens},
                },
            )

            yield _sse("message_stop", {"type": "message_stop"})
