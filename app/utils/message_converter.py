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
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            })

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
                openai_tool_choice = {"type": "function", "function": {"name": tc["name"]}}
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
                converted = []
                for block in content:
                    if block.get("type") == "text":
                        converted.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        data_url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        converted.append({"type": "image_url", "image_url": {"url": data_url}})
                    elif block.get("type") == "tool_result":
                        # tool_result -> OpenAI role: "tool" 消息
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                b.get("text", "") for b in result_content if b.get("type") == "text"
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(result_content) if result_content else "",
                        })
                if converted:
                    messages.append({"role": "user", "content": converted})
        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

                if tool_use_blocks:
                    tool_calls = []
                    for b in tool_use_blocks:
                        tool_calls.append({
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                            }
                        })
                    text_content = " ".join(text_parts) if text_parts else None
                    messages.append({"role": "assistant", "content": text_content, "tool_calls": tool_calls})
                else:
                    messages.append({"role": "assistant", "content": " ".join(text_parts)})

    return {
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


def _sse(event_type: str, data: dict) -> bytes:
    """生成 Anthropic SSE 事件"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def openai_stream_to_anthropic(
    openai_stream: AsyncGenerator,
    model: str,
    request_id: str = "",
) -> AsyncGenerator[bytes, None]:
    """
    将 OpenAI 格式的 SSE 流实时转换为 Anthropic 格式。
    支持文本 content 和 tool_calls。
    状态机：message_start -> [content_block_start -> delta* -> stop]* -> message_delta -> message_stop
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    sent_message_start = False
    total_output_tokens = 0
    finish_reason = "end_turn"

    # 块追踪
    open_block_index = -1        # 当前打开的块索引，-1 表示无
    text_block_opened = False     # 文本块是否已打开（索引 0）
    tool_calls_map = {}           # {openai_tc_index: {"id", "name", "block_index"}}
    tool_block_counter = 0        # 已创建的 tool 块数量

    def _close_open_block():
        nonlocal open_block_index
        if open_block_index >= 0:
            idx = open_block_index
            open_block_index = -1
            return _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        return None

    try:
        async for chunk in openai_stream:
            # chunk 对象转 dict
            if hasattr(chunk, "model_dump"):
                chunk_data = chunk.model_dump(exclude_none=True)
            elif hasattr(chunk, "to_dict"):
                chunk_data = chunk.to_dict()
            elif isinstance(chunk, dict):
                chunk_data = chunk
            else:
                continue

            choices = chunk_data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")
            finish = choices[0].get("finish_reason")

            # 生成 message_start（首次）
            if not sent_message_start:
                sent_message_start = True
                yield _sse("message_start", {
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
                })

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

                        # 新 tool_use 块
                        tc_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        func = tc_delta.get("function", {})
                        tc_name = func.get("name", "")

                        # 块索引：文本块在 0，tool 块从 1（或 0 如果没有文本）
                        if text_block_opened:
                            block_idx = 1 + tool_block_counter
                        else:
                            block_idx = tool_block_counter
                        tool_block_counter += 1

                        tool_calls_map[tc_index] = {
                            "id": tc_id,
                            "name": tc_name,
                            "block_index": block_idx,
                        }
                        open_block_index = block_idx

                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name, "input": {}},
                        })

                    # 发送 arguments delta
                    func = tc_delta.get("function", {})
                    args_fragment = func.get("arguments", "")
                    if args_fragment:
                        block_idx = tool_calls_map[tc_index]["block_index"]
                        open_block_index = block_idx
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args_fragment},
                        })

            # 处理文本 content
            if content is not None:
                if not text_block_opened:
                    text_block_opened = True
                    close_ev = _close_open_block()
                    if close_ev:
                        yield close_ev
                    open_block_index = 0
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    })

                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                })
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
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": finish_reason, "stop_sequence": None},
                "usage": {"output_tokens": total_output_tokens},
            })

            yield _sse("message_stop", {"type": "message_stop"})
