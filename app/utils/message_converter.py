from __future__ import annotations

import json
import uuid
import re
from typing import Any, AsyncGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning/Thinking Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def is_openai_o_series(model: str) -> bool:
    """Detect OpenAI o-series reasoning models (o1, o3, o4-mini, etc.).

    These models require `max_completion_tokens` instead of `max_tokens`
    and support `reasoning_effort` parameter.
    """
    return (
        len(model) > 1
        and model.startswith("o")
        and model[1].isdigit()
    )


def is_gpt5_plus(model: str) -> bool:
    """Detect OpenAI GPT-5+ models that support reasoning_effort."""
    model_lower = model.lower()
    if model_lower.startswith("gpt-"):
        rest = model_lower[4:]
        first_char = rest[0] if rest else ""
        return first_char.isdigit() and int(first_char) >= 5
    return False


def supports_reasoning_effort(model: str) -> bool:
    """Check if a model supports reasoning_effort parameter.

    Supported families:
    - o-series: o1, o3, o4-mini, etc.
    - GPT-5+: gpt-5, gpt-5.1, gpt-5.4, gpt-5-codex, etc.
    """
    return is_openai_o_series(model) or is_gpt5_plus(model)


def resolve_reasoning_effort(body: dict) -> str | None:
    """Resolve the appropriate OpenAI `reasoning_effort` from an Anthropic request body.

    Priority:
    1. Explicit `output_config.effort` — preserves the user's intent directly.
       `low`/`medium`/`high` map 1:1; `max` maps to `xhigh`
       (supported by mainstream GPT models). Unknown values are ignored.
    2. Fallback: `thinking.type` + `budget_tokens`:
       - `adaptive` → `high` (mirrors optimizer semantics where adaptive ≈ max effort)
       - `enabled` with budget → `low` (<4,000) / `medium` (4,000–15,999) / `high` (≥16,000)
       - `enabled` without budget → `high` (conservative default)
       - `disabled` / absent → `None`

    Args:
        body: Anthropic request body dict

    Returns:
        One of "low", "medium", "high", "xhigh", or None
    """
    # Priority 1: explicit output_config.effort
    output_config = body.get("output_config", {})
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    if effort:
        if effort in ("low", "medium", "high"):
            return effort
        if effort == "max":
            return "xhigh"
        # Unknown value — do not inject

    # Priority 2: thinking.type + budget_tokens fallback
    thinking = body.get("thinking")
    if not thinking or not isinstance(thinking, dict):
        return None

    thinking_type = thinking.get("type")
    if thinking_type == "adaptive":
        return "high"
    if thinking_type == "enabled":
        budget = thinking.get("budget_tokens")
        if budget is None:
            return "high"  # enabled but no budget — assume strong reasoning
        budget = int(budget)
        if budget < 4000:
            return "low"
        elif budget < 16000:
            return "medium"
        return "high"

    # disabled or unknown
    return None


def clean_json_schema(schema: dict) -> dict:
    """Clean JSON schema by removing unsupported formats.

    Some OpenAI-compatible endpoints don't support certain JSON schema formats
    like `format: "uri"`. This function removes such unsupported formats.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # Remove unsupported format
    if result.get("format") == "uri":
        result.pop("format", None)

    # Recursively clean nested schema
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {
            k: clean_json_schema(v) for k, v in result["properties"].items()
        }

    if "items" in result:
        result["items"] = clean_json_schema(result["items"])

    if "additionalProperties" in result and isinstance(result["additionalProperties"], dict):
        result["additionalProperties"] = clean_json_schema(result["additionalProperties"])

    return result


def anthropic_to_openai_messages(data: dict) -> dict:
    """将 Anthropic Messages API 请求体转换为 OpenAI 格式

    Features:
    - Preserves cache_control markers for prompt caching
    - Handles o-series models (uses max_completion_tokens)
    - Maps thinking parameters to reasoning_effort with priority-based resolution
    - Filters BatchTool (Anthropic internal tool type)
    - Cleans JSON schema (removes unsupported formats like uri)
    """
    messages = []
    model = data.get("model", "")

    # Anthropic 的 system 是顶层字段
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Join system blocks into single message, preserving cache_control if any
            text_parts = []
            has_cache_control = False
            for block in system:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                    if block.get("cache_control"):
                        has_cache_control = True
            if text_parts:
                sys_msg = {"role": "system", "content": " ".join(text_parts)}
                # Only preserve cache_control on the first system message
                if has_cache_control:
                    for block in system:
                        if block.get("type") == "text" and block.get("cache_control"):
                            sys_msg["cache_control"] = block["cache_control"]
                            break
                messages.append(sys_msg)

    # 转换 tools: Anthropic -> OpenAI (过滤 BatchTool，保留 cache_control)
    openai_tools = None
    if "tools" in data:
        openai_tools = []
        for tool in data["tools"]:
            # Filter BatchTool (Anthropic internal type)
            if tool.get("type") == "BatchTool":
                continue
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": clean_json_schema(
                        tool.get("input_schema", {})
                    ),
                },
            }
            # Preserve cache_control on tools
            if tool.get("cache_control"):
                openai_tool["cache_control"] = tool["cache_control"]
            openai_tools.append(openai_tool)
        if not openai_tools:
            openai_tools = None

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
                        text_part = {"type": "text", "text": block.get("text", "")}
                        # Preserve cache_control on text blocks
                        if block.get("cache_control"):
                            text_part["cache_control"] = block["cache_control"]
                        converted.append(text_part)
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
                    # Note: thinking blocks in user messages are silently dropped
                    # (OpenAI doesn't have an equivalent)
                # H2 fix: user text first, then tool results
                # Simplify: if only single text block (no images), use string content
                # Some OpenAI-compatible providers (GLM/BigModel) don't support array content
                if converted:
                    if len(converted) == 1 and converted[0].get("type") == "text":
                        # Single text block -> simplify to string
                        text_content = converted[0].get("text", "")
                        # Preserve cache_control if present
                        user_msg = {"role": "user", "content": text_content}
                        if converted[0].get("cache_control"):
                            user_msg["cache_control"] = converted[0]["cache_control"]
                        messages.append(user_msg)
                    else:
                        messages.append({"role": "user", "content": converted})
                messages.extend(tool_results)
        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # Collect text and thinking parts
                text_parts = []
                thinking_parts = []
                tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
                has_cache_control = False

                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                        if block.get("cache_control"):
                            has_cache_control = True
                    elif block.get("type") == "thinking":
                        # Merge thinking into text for history preservation (OpenAI doesn't support thinking blocks)
                        thinking_parts.append(block.get("thinking", ""))
                        if block.get("cache_control"):
                            has_cache_control = True

                # Combine thinking and text parts
                all_text_parts = thinking_parts + text_parts
                combined_text = " ".join(all_text_parts) if all_text_parts else None

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
                    # Build assistant message with content and tool_calls
                    msg = {"role": "assistant", "tool_calls": tool_calls}
                    if combined_text:
                        msg["content"] = combined_text
                    else:
                        msg["content"] = None
                    messages.append(msg)
                elif combined_text:
                    messages.append({"role": "assistant", "content": combined_text})

    result = {
        "model": model,
        "messages": messages,
        "temperature": data.get("temperature"),
        "top_p": data.get("top_p"),
        "top_k": data.get("top_k"),
        "stream": data.get("stream", False),
        "stop": data.get("stop_sequences"),
        "tools": openai_tools,
        "tool_choice": openai_tool_choice,
    }

    # Handle max_tokens: o-series models need max_completion_tokens
    max_tokens = data.get("max_tokens")
    if max_tokens:
        if is_openai_o_series(model):
            result["max_completion_tokens"] = max_tokens
        else:
            result["max_tokens"] = max_tokens

    # Map Anthropic thinking → OpenAI reasoning_effort
    # For supported models: use sophisticated resolution
    # For other models: use simple fallback (backward compatible)
    if supports_reasoning_effort(model):
        effort = resolve_reasoning_effort(data)
        if effort:
            result["reasoning_effort"] = effort
    else:
        # Backward compatible: simple mapping for non-reasoning models
        thinking = data.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            result["reasoning_effort"] = "high"

    # Handle budget_tokens: it overrides max_tokens regardless of model type
    thinking = data.get("thinking")
    if isinstance(thinking, dict) and thinking.get("budget_tokens"):
        budget = thinking["budget_tokens"]
        if is_openai_o_series(model):
            result["max_completion_tokens"] = budget
        else:
            result["max_tokens"] = budget

    return result


def convert_openai_to_anthropic_response(
    resp_data: dict, model: str, thinking_enabled: bool = False
) -> dict:
    """将 OpenAI ChatCompletion 响应转换为 Anthropic Messages 响应。

    Args:
        resp_data: OpenAI 格式的响应 dict
        model: 模型名称
        thinking_enabled: 客户端是否请求了 thinking（决定是否生成 thinking 块）

    Features:
    - Handles refusal blocks (both content parts and message-level)
    - Maps content_filter finish_reason to end_turn
    - Maps cache token fields from OpenAI format to Anthropic format
    - Handles legacy function_call format
    """
    choices = resp_data.get("choices", [])
    content = []
    stop_reason = "end_turn"
    has_tool_use = False

    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        reasoning = message.get("reasoning_content")
        msg_content = message.get("content")

        # Handle content (string, array, or None)
        if msg_content:
            if isinstance(msg_content, str):
                text_to_emit = msg_content
                if not thinking_enabled and reasoning:
                    text_to_emit = reasoning + msg_content
                if thinking_enabled and reasoning:
                    content.append({"type": "thinking", "thinking": reasoning})
                if text_to_emit:
                    content.append({"type": "text", "text": text_to_emit})
            elif isinstance(msg_content, list):
                # Content parts array - handle text, output_text, and refusal
                for part in msg_content:
                    part_type = part.get("type", "")
                    if part_type in ("text", "output_text"):
                        text = part.get("text", "")
                        if text:
                            content.append({"type": "text", "text": text})
                    elif part_type == "refusal":
                        refusal = part.get("refusal", "")
                        if refusal:
                            content.append({"type": "text", "text": refusal})
        elif reasoning:
            # Only reasoning content, no text
            if thinking_enabled:
                content.append({"type": "thinking", "thinking": reasoning})
            else:
                content.append({"type": "text", "text": reasoning})

        # Handle message-level refusal (some providers put it here)
        refusal = message.get("refusal")
        if refusal and isinstance(refusal, str) and refusal:
            content.append({"type": "text", "text": refusal})

        # Handle tool_calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            has_tool_use = True
            for tc in tool_calls:
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

        # Handle legacy function_call format
        if not has_tool_use:
            function_call = message.get("function_call")
            if function_call:
                fc_name = function_call.get("name", "")
                fc_args = function_call.get("arguments")
                fc_id = function_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                if fc_name or fc_args:
                    try:
                        if isinstance(fc_args, str):
                            fc_input = json.loads(fc_args)
                        else:
                            fc_input = fc_args or {}
                    except (json.JSONDecodeError, TypeError):
                        fc_input = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": fc_id,
                            "name": fc_name,
                            "input": fc_input,
                        }
                    )
                    has_tool_use = True

        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "end_turn",
        }.get(finish_reason, "end_turn")

        # If we have tool_use but finish_reason doesn't indicate it, fix it
        if has_tool_use and stop_reason != "tool_use":
            stop_reason = "tool_use"

    # C3 fix: 如果 content 仍为空，添加空 text 块保证协议合规
    if not content:
        content.append({"type": "text", "text": ""})

    # Build usage with cache token mapping
    usage = resp_data.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    # Map cache tokens from OpenAI format to Anthropic format
    # OpenAI standard: prompt_tokens_details.cached_tokens
    cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    if cached_tokens:
        anthropic_usage["cache_read_input_tokens"] = cached_tokens

    # Some compatible servers return these fields directly
    if usage.get("cache_read_input_tokens"):
        anthropic_usage["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
    if usage.get("cache_creation_input_tokens"):
        anthropic_usage["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]

    return {
        "id": resp_data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
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

    Features:
    - Handles signature_delta events for thinking blocks
    - Maps content_filter finish_reason to end_turn
    - Handles function_call finish_reason

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
    thinking_signature = ""  # Track signature for thinking block
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
                    # Note: We start with empty signature, it will be filled at the end
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
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

            # Handle signature_delta for thinking blocks (some providers send this)
            # OpenAI reasoning models don't send signatures, but Anthropic-native providers do
            signature_delta = delta.get("signature")
            if signature_delta is not None and thinking_block_opened and thinking_enabled:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block_index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": signature_delta,
                        },
                    },
                )
                thinking_signature = signature_delta

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
                    "function_call": "tool_use",
                    "content_filter": "end_turn",
                }.get(finish, "end_turn")

            # Handle OpenAI reasoning_tokens in usage (if present in delta)
            # Some providers return usage in the final chunk
            usage = chunk_data.get("usage", {})
            if usage:
                # We don't have input_tokens at this point, but we can track output_tokens
                if usage.get("completion_tokens"):
                    total_output_tokens = usage.get("completion_tokens", total_output_tokens)

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
