from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
import re
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI has a 64-character limit for function/tool names
# Anthropic does not have this limit, so we need to truncate long names
OPENAI_MAX_TOOL_NAME_LENGTH = 64
TOOL_NAME_HASH_LENGTH = 8
TOOL_NAME_PREFIX_LENGTH = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1  # 55


# ─────────────────────────────────────────────────────────────────────────────
# Safe Parsing Helpers
# ─────────────────────────────────────────────────────────────────────────────


def truncate_tool_name(name: str) -> str:
    """
    Truncate tool names that exceed OpenAI's 64-character limit.

    Uses format: {55-char-prefix}_{8-char-hash} to avoid collisions
    when multiple tools have similar long names.

    Args:
        name: The original tool name

    Returns:
        The original name if <= 64 chars, otherwise truncated with hash
    """
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name

    # Create deterministic hash from full name to avoid collisions
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"


def create_tool_name_mapping(tools: list[dict]) -> dict[str, str]:
    """
    Create a mapping of truncated tool names to original names.

    Args:
        tools: List of tool definitions with 'name' field

    Returns:
        Dict mapping truncated names to original names (only for truncated tools)
    """
    mapping: dict[str, str] = {}
    for tool in tools:
        original_name = tool.get("name", "")
        truncated_name = truncate_tool_name(original_name)
        if truncated_name != original_name:
            mapping[truncated_name] = original_name
    return mapping


def restore_tool_name(name: str, mapping: dict[str, str]) -> str:
    """
    Restore original tool name from truncated name using mapping.

    Args:
        name: The potentially truncated tool name
        mapping: Dict of truncated -> original names

    Returns:
        Original name if in mapping, otherwise the input name
    """
    return mapping.get(name, name)


def _safe_json_parse(json_str: str, default: Any, context: str = "") -> Any:
    """Parse JSON string safely with logging on failure.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        context: Context string for logging (e.g., "tool_call arguments")

    Returns:
        Parsed JSON object or default value
    """
    if not json_str:
        return default
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON parse error in {context}: {e.msg} at position {e.pos}, "
            f"input preview: {json_str[:100]}"
        )
        return default


def _safe_get(obj: dict | Any, key: str, default: Any = None, required: bool = False) -> Any:
    """Get value from dict safely with optional required field check.

    Args:
        obj: Object to get value from (can be dict or object with attributes)
        key: Key to get
        default: Default value if key missing
        required: If True, log warning when key missing

    Returns:
        Value or default
    """
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)

    if required and value == default:
        logger.warning(f"Missing required field '{key}' in {type(obj).__name__}")

    return value


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
       - `adaptive` → `medium` (OpenAI default for adaptive reasoning)
       - `enabled` with budget → thresholds aligned with LiteLLM standard:
         >= 10000 → `high`, >= 5000 → `medium`, >= 2000 → `low`, < 2000 → `minimal`
       - `enabled` without budget → `high` (conservative default)
       - `disabled` / absent → `None`

    Args:
        body: Anthropic request body dict

    Returns:
        One of "minimal", "low", "medium", "high", "xhigh", or None
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
        # Use output_config.effort if available, otherwise default to medium
        if isinstance(output_config, dict) and output_config.get("effort"):
            return output_config["effort"]
        return "medium"
    if thinking_type == "enabled":
        budget = thinking.get("budget_tokens")
        if budget is None:
            return "high"  # enabled but no budget — assume strong reasoning
        budget = int(budget)
        # LiteLLM-aligned thresholds:
        # >= 10000 -> high, >= 5000 -> medium, >= 2000 -> low, < 2000 -> minimal
        if budget >= 10000:
            return "high"
        elif budget >= 5000:
            return "medium"
        elif budget >= 2000:
            return "low"
        else:
            return "minimal"

    # disabled or unknown
    return None


def resolve_reasoning_summary(body: dict) -> str | None:
    """Resolve the reasoning summary setting from Anthropic request body.

    Args:
        body: Anthropic request body dict

    Returns:
        One of "detailed", "auto", "concise", or None
    """
    thinking = body.get("thinking")
    if not thinking or not isinstance(thinking, dict):
        return None

    summary = thinking.get("summary")
    if summary:
        return summary

    # Default to "detailed" for enabled thinking (matches LiteLLM behavior)
    if thinking.get("type") == "enabled":
        return "detailed"

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


def anthropic_to_openai_messages(
    data: dict,
    preserve_thinking_blocks: bool = False,
) -> tuple[dict, dict[str, str]]:
    """将 Anthropic Messages API 请求体转换为 OpenAI 格式

    Args:
        data: Anthropic request body dict
        preserve_thinking_blocks: If True, preserve thinking blocks as separate
            reasoning_content instead of merging into text. Derived from thinking.type==enabled.

    Returns:
        Tuple of (OpenAI request dict, tool_name_mapping for restoring truncated names)

    Features:
    - Preserves cache_control markers for prompt caching
    - Handles o-series models (uses max_completion_tokens)
    - Maps thinking parameters to reasoning_effort with LiteLLM-aligned thresholds
    - Filters BatchTool (Anthropic internal tool type)
    - Cleans JSON schema (removes unsupported formats like uri)
    - Truncates tool names exceeding OpenAI's 64-char limit
    - Maps metadata.user_id to user field (truncated to 64 chars)
    - Handles output_format/output_config for structured outputs
    """
    messages = []
    model = data.get("model", "")

    # System conversion
    system = data.get("system")
    if system:
        messages.extend(_convert_system_block(system))

    # Tools conversion (returns tuple: tools, name_mapping)
    openai_tools, tool_name_mapping = _convert_tools_block(data.get("tools"))

    # Tool_choice conversion
    openai_tool_choice = _convert_tool_choice(data.get("tool_choice"))

    # Messages conversion
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            converted, tool_results = _convert_user_content(content)
            messages.extend(converted)
            messages.extend(tool_results)
        elif role == "assistant":
            converted_msg = _convert_assistant_content(content, preserve_thinking_blocks)
            messages.append(converted_msg)

    # Debug: log the converted messages structure
    for i, msg in enumerate(messages):
        content_type = type(msg.get("content")).__name__
        content_preview = None
        if isinstance(msg.get("content"), str):
            content_preview = msg.get("content", "")[:50]
        elif isinstance(msg.get("content"), list):
            content_preview = f"[list of {len(msg['content'])} items]"
            # Debug log for list content (changed from WARNING to DEBUG for performance)
            logger.debug(
                f"[message conversion] messages[{i}] has content as LIST - "
                f"role={msg.get('role')}, "
                f"content_types={[b.get('type') for b in msg['content']]}"
            )
        logger.debug(
            f"[message conversion] index={i}, role={msg.get('role')}, "
            f"content_type={content_type}, content_preview={content_preview}, "
            f"has_tool_calls={bool(msg.get('tool_calls'))}, "
            f"has_reasoning_content={bool(msg.get('reasoning_content'))}"
        )

    openai_request = _build_openai_request_dict(
        data, model, messages, openai_tools, openai_tool_choice
    )
    return openai_request, tool_name_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for anthropic_to_openai_messages decomposition
# ─────────────────────────────────────────────────────────────────────────────


def _convert_system_block(system: str | list) -> list[dict]:
    """Convert Anthropic system field to OpenAI system message(s).

    Args:
        system: Anthropic system field (string or list of blocks)

    Returns:
        List of OpenAI system messages
    """
    messages = []
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
    return messages


def _convert_tools_block(tools: list) -> tuple[list[dict] | None, dict[str, str]]:
    """Convert Anthropic tools to OpenAI function format.

    Filters BatchTool (Anthropic internal tool type) and preserves cache_control.
    Truncates tool names exceeding OpenAI's 64-character limit.

    Args:
        tools: Anthropic tools list

    Returns:
        Tuple of (OpenAI tools list or None if empty, tool_name_mapping for truncated names)
    """
    if not tools:
        return None, {}

    openai_tools = []
    tool_name_mapping: dict[str, str] = {}

    for tool in tools:
        if not isinstance(tool, dict):
            logger.warning(f"Invalid tool type: {type(tool).__name__}, expected dict")
            continue

        # Filter BatchTool (Anthropic internal type)
        if tool.get("type") == "BatchTool":
            continue

        # Validate required 'name' field
        original_name = tool.get("name")
        if not original_name:
            logger.warning(f"Tool missing required 'name' field, skipping")
            continue

        # Truncate name if exceeds OpenAI's 64-char limit
        truncated_name = truncate_tool_name(original_name)
        if truncated_name != original_name:
            tool_name_mapping[truncated_name] = original_name
            logger.debug(f"Tool name truncated: '{original_name}' -> '{truncated_name}'")

        openai_tool = {
            "type": "function",
            "function": {
                "name": truncated_name,
                "description": tool.get("description", ""),
                "parameters": clean_json_schema(tool.get("input_schema", {})),
            },
        }
        # Preserve cache_control on tools
        if tool.get("cache_control"):
            openai_tool["cache_control"] = tool["cache_control"]
        openai_tools.append(openai_tool)

    return (openai_tools if openai_tools else None), tool_name_mapping


def _convert_tool_choice(tool_choice: dict | str) -> dict | str | None:
    """Convert Anthropic tool_choice to OpenAI format.

    Args:
        tool_choice: Anthropic tool_choice field

    Returns:
        OpenAI tool_choice value or None
    """
    if not tool_choice:
        return None

    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "auto")
        if tc_type == "auto":
            return "auto"
        elif tc_type == "any":
            return "required"
        elif tc_type == "none":
            return "none"
        elif tc_type == "tool":
            tc_name = tool_choice.get("name")
            if not tc_name:
                logger.warning("tool_choice type='tool' missing required 'name' field, falling back to auto")
                return "auto"
            return {
                "type": "function",
                "function": {"name": tc_name},
            }
        else:
            logger.warning(f"Unknown tool_choice type: '{tc_type}', defaulting to auto")
            return "auto"
    elif isinstance(tool_choice, str):
        return tool_choice

    return None


def _convert_user_content(content: str | list) -> tuple[list[dict], list[dict]]:
    """Convert Anthropic user message content.

    Args:
        content: User message content (string or list of blocks)

    Returns:
        Tuple of (converted_messages, tool_result_messages)

    Note: Claude Code 2.x may send tool_use blocks in user messages in certain
    agent scenarios. We handle this by creating separate assistant messages
    with tool_calls for these blocks, maintaining conversation flow.
    """
    converted = []
    tool_results = []
    # Track tool_use blocks that appear in user messages (Claude Code 2.x edge case)
    user_tool_use_blocks = []

    if isinstance(content, str):
        converted.append({"role": "user", "content": content})
    elif isinstance(content, list):
        for block in content:
            if block.get("type") == "text":
                text_value = block.get("text", "")
                # Filter out empty/whitespace-only text blocks (prevents API Error 400)
                # Anthropic API rejects text content blocks with whitespace-only content
                if not text_value or not text_value.strip():
                    logger.debug("Skipping empty/whitespace-only text block in user content")
                    continue
                text_part = {"type": "text", "text": text_value}
                # Preserve cache_control on text blocks
                if block.get("cache_control"):
                    text_part["cache_control"] = block["cache_control"]
                converted.append(text_part)
            elif block.get("type") == "image":
                source = block.get("source", {})
                if not isinstance(source, dict):
                    logger.warning("image block 'source' is not a dict, skipping")
                    continue
                media_type = source.get("media_type", "image/png")
                image_data = source.get("data", "")
                if not image_data:
                    logger.warning("image block missing 'data' field, skipping")
                    continue
                data_url = f"data:{media_type};base64,{image_data}"
                converted.append({"type": "image_url", "image_url": {"url": data_url}})
            elif block.get("type") == "tool_result":
                # tool_result -> OpenAI role: "tool" 消息
                tool_use_id = block.get("tool_use_id")
                if not tool_use_id:
                    logger.warning("tool_result block missing required 'tool_use_id' field, using placeholder")
                    tool_use_id = "unknown_tool"

                # Capture is_error field - OpenAI doesn't support this directly,
                # but we encode it by prefixing content with error indicator
                is_error = block.get("is_error", False)

                result_content = block.get("content", "")

                # Handle different content types in tool_result:
                # 1. String content -> use directly
                # 2. List with only text blocks -> join text
                # 3. List with mixed content (images, etc.) -> JSON serialize to preserve all data
                if isinstance(result_content, list):
                    # Check if all blocks are text
                    all_text_blocks = all(
                        isinstance(b, dict) and b.get("type") == "text"
                        for b in result_content
                    )

                    if all_text_blocks:
                        # Simple case: only text blocks, join them
                        result_content = " ".join(
                            b.get("text", "") for b in result_content if isinstance(b, dict)
                        )
                    else:
                        # Complex case: contains non-text content (images, etc.)
                        # JSON serialize to preserve all data (matches reference implementation)
                        # This ensures images and other structured content are not lost
                        logger.debug(f"[tool_result] Content contains non-text blocks, JSON serializing")
                        result_content = json.dumps(result_content, ensure_ascii=False)

                # If tool execution failed, prefix with error indicator for visibility
                # This preserves the error information in a way OpenAI-compatible providers can handle
                if is_error and result_content:
                    result_content = f"[ERROR] {result_content}"

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": str(result_content) if result_content else "",
                })
            elif block.get("type") == "tool_use":
                # Claude Code 2.x edge case: tool_use in user message
                # This can happen in agent workflows where Claude acts as both user and assistant
                # We collect these and will create assistant messages with tool_calls
                logger.debug(f"[user] Found tool_use block in user message (Claude Code 2.x edge case)")
                user_tool_use_blocks.append(block)
            else:
                # Log unexpected block types for debugging
                logger.debug(f"[user] Unrecognized block type '{block.get('type')}' in user content")

        # Simplify: if only single text block (no images), use string content
        # Some OpenAI-compatible providers (GLM/BigModel) don't support array content
        if converted:
            has_images = any(b.get("type") == "image_url" for b in converted)
            if len(converted) == 1 and converted[0].get("type") == "text":
                text_content = converted[0].get("text", "")
                user_msg = {"role": "user", "content": text_content}
                if converted[0].get("cache_control"):
                    user_msg["cache_control"] = converted[0]["cache_control"]
                converted = [user_msg]
            elif not has_images:
                # Multiple text blocks only: join into single string for provider compatibility
                all_text = " ".join(b.get("text", "") for b in converted if b.get("type") == "text")
                user_msg = {"role": "user", "content": all_text}
                # Preserve cache_control from first block if any
                first_text = next((b for b in converted if b.get("type") == "text"), None)
                if first_text and first_text.get("cache_control"):
                    user_msg["cache_control"] = first_text["cache_control"]
                converted = [user_msg]
            else:
                # Has images: use array content format (required for multi-modal)
                converted = [{"role": "user", "content": converted}]

    # Handle tool_use blocks that appeared in user messages (Claude Code 2.x edge case)
    # In OpenAI format, tool calls belong to assistant messages, so we need to create
    # synthetic assistant messages for these tool_use blocks
    # This maintains conversation flow and ensures downstream providers can handle them
    if user_tool_use_blocks:
        tool_calls = []
        for b in user_tool_use_blocks:
            if not isinstance(b, dict):
                logger.warning(f"Invalid tool_use block in user message: {type(b).__name__}")
                continue

            tool_id = b.get("id")
            if not tool_id:
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                logger.debug(f"tool_use block in user message missing 'id', generated: {tool_id}")

            tool_name = b.get("name", "")
            tool_input = b.get("input", {})

            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_input, ensure_ascii=False),
                },
            })

        if tool_calls:
            # Create a synthetic assistant message with the tool calls
            # This represents Claude acting as an agent making tool calls
            assistant_msg = {"role": "assistant", "tool_calls": tool_calls, "content": None}
            # Insert before the user message to maintain conversation flow:
            # assistant (tool_calls) -> tool_result -> user (next message)
            converted.insert(0, assistant_msg)
            logger.debug(f"[Claude Code 2.x] Created synthetic assistant message with {len(tool_calls)} tool_calls")

    return converted, tool_results


def _convert_assistant_content(
    content: str | list,
    preserve_thinking_blocks: bool = False,
) -> dict:
    """Convert Anthropic assistant message content.

    Args:
        content: Assistant message content (string or list of blocks)
        preserve_thinking_blocks: If True, preserve thinking blocks as separate
            reasoning_content instead of merging into text.

    Returns:
        OpenAI assistant message dict
    """
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    if not isinstance(content, list):
        return {"role": "assistant", "content": content or ""}

    # Collect text and thinking parts
    text_parts = []
    thinking_parts = []
    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
    # Also collect tool_result blocks (should not appear in assistant, but check anyway)
    has_cache_control = False

    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            text_value = block.get("text", "")
            # Filter out empty/whitespace-only text blocks
            if text_value and text_value.strip():
                text_parts.append(text_value)
            if block.get("cache_control"):
                has_cache_control = True
        elif block_type == "thinking":
            thinking_value = block.get("thinking", "")
            # Filter out empty/whitespace-only thinking blocks
            if thinking_value and thinking_value.strip():
                thinking_parts.append(thinking_value)
            if block.get("cache_control"):
                has_cache_control = True
        elif block_type == "tool_use":
            # Already collected above
            pass
        elif block_type == "redacted_thinking":
            # Handle redacted thinking blocks (Claude Code 2.x privacy feature)
            # These contain thinking content that was redacted for privacy
            logger.debug(f"[assistant] Found redacted_thinking block, skipping")
        else:
            # Log unexpected block types but don't crash - handle gracefully
            logger.debug(f"[assistant] Unexpected block type '{block_type}' in content list, skipping")

    # Debug log the conversion
    logger.debug(
        f"[assistant conversion] preserve_thinking={preserve_thinking_blocks}, "
        f"text_parts={len(text_parts)}, thinking_parts={len(thinking_parts)}, "
        f"tool_use_blocks={len(tool_use_blocks)}"
    )

    # NEW: If preserve_thinking_blocks and we have thinking, emit reasoning_content separately
    if preserve_thinking_blocks and thinking_parts:
        combined_thinking = " ".join(thinking_parts)
        combined_text = " ".join(text_parts) if text_parts else None

        if tool_use_blocks:
            tool_calls = []
            for b in tool_use_blocks:
                if not isinstance(b, dict):
                    logger.warning(f"Invalid tool_use block type: {type(b).__name__}, expected dict")
                    continue

                tool_id = b.get("id")
                if not tool_id:
                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    logger.debug(f"tool_use block missing 'id', generated: {tool_id}")

                tool_name = b.get("name", "")
                tool_input = b.get("input", {})

                tool_calls.append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input, ensure_ascii=False),
                    },
                })

            msg = {"role": "assistant", "tool_calls": tool_calls}
            msg["content"] = combined_text
            msg["reasoning_content"] = combined_thinking
            return msg
        else:
            msg = {"role": "assistant"}
            if combined_text:
                msg["content"] = combined_text
            else:
                msg["content"] = None
            msg["reasoning_content"] = combined_thinking
            return msg

    # Backward compatible: merge thinking into text
    all_text_parts = thinking_parts + text_parts
    combined_text = " ".join(all_text_parts) if all_text_parts else None

    if tool_use_blocks:
        tool_calls = []
        for b in tool_use_blocks:
            if not isinstance(b, dict):
                logger.warning(f"Invalid tool_use block type: {type(b).__name__}, expected dict")
                continue

            tool_id = b.get("id")
            if not tool_id:
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                logger.debug(f"tool_use block missing 'id', generated: {tool_id}")

            tool_name = b.get("name", "")
            tool_input = b.get("input", {})

            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_input, ensure_ascii=False),
                },
            })

        if tool_calls:
            msg = {"role": "assistant", "tool_calls": tool_calls}
            msg["content"] = combined_text if combined_text else None
            return msg
    elif combined_text:
        return {"role": "assistant", "content": combined_text}

    return {"role": "assistant", "content": ""}


def _build_openai_request_dict(
    data: dict,
    model: str,
    messages: list,
    openai_tools: list | None,
    openai_tool_choice: dict | str | None,
) -> dict:
    """Build final OpenAI request dict from converted components.

    Args:
        data: Original Anthropic request body
        model: Model name
        messages: Converted messages
        openai_tools: Converted tools
        openai_tool_choice: Converted tool_choice

    Returns:
        OpenAI request dict

    Note: The following Anthropic parameters are intentionally NOT mapped:
        - top_k: Not supported by OpenAI or most compatible providers
        - stop_sequences: Mapped to 'stop' (OpenAI equivalent)
        - speed: Not supported by OpenAI Chat Completions API
    """
    result = {
        "model": model,
        "messages": messages,
        "temperature": data.get("temperature"),
        "top_p": data.get("top_p"),
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
    if supports_reasoning_effort(model):
        effort = resolve_reasoning_effort(data)
        if effort:
            result["reasoning_effort"] = effort
        # Add reasoning summary if specified (for models that support it)
        summary = resolve_reasoning_summary(data)
        if summary:
            result["reasoning_summary"] = summary
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

    # Map metadata.user_id → user field (truncated to 64 chars per Anthropic spec)
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and "user_id" in metadata:
        result["user"] = str(metadata["user_id"])[:64]

    # Handle output_format / output_config.format for structured outputs
    # Anthropic: output_format: {"type": "json_schema", "schema": {...}}
    # Anthropic: output_config: {"format": {"type": "json_schema", "schema": {...}}}
    # OpenAI: response_format: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}
    output_format: Any = data.get("output_format")
    output_config = data.get("output_config")
    if not isinstance(output_format, dict) and isinstance(output_config, dict):
        output_format = output_config.get("format")
    if isinstance(output_format, dict) and output_format.get("type") == "json_schema":
        schema = output_format.get("schema")
        if schema:
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            }

    return result


def convert_openai_to_anthropic_response(
    resp_data: dict,
    model: str,
    thinking_enabled: bool = False,
    preserve_thinking_blocks: bool = False,
    tool_name_mapping: dict[str, str] | None = None,
) -> dict:
    """将 OpenAI ChatCompletion 响应转换为 Anthropic Messages 响应。

    Args:
        resp_data: OpenAI 格式的响应 dict
        model: 模型名称
        thinking_enabled: 客户端是否请求了 thinking（决定是否生成 thinking 块）
        preserve_thinking_blocks: If True, preserve reasoning_content as separate thinking
            block instead of merging into text.
        tool_name_mapping: Mapping of truncated tool names to original names.
                          Used to restore original names for tools that exceeded
                          OpenAI's 64-char limit.

    Features:
    - Handles refusal blocks (both content parts and message-level)
    - Maps content_filter finish_reason to end_turn
    - Maps cache token fields from OpenAI format to Anthropic format
    - Handles legacy function_call format
    - Restores truncated tool names using tool_name_mapping
    """
    choices = resp_data.get("choices", [])
    content = []
    stop_reason = "end_turn"
    has_tool_use = False

    # Determine if to preserve thinking as separate block
    emit_separate_thinking = thinking_enabled or preserve_thinking_blocks

    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        reasoning = message.get("reasoning_content")
        msg_content = message.get("content")

        # Handle content (string, array, or None)
        if msg_content:
            if isinstance(msg_content, str):
                # Filter out empty/whitespace-only content
                if not msg_content.strip():
                    msg_content = None
                else:
                    if emit_separate_thinking and reasoning and reasoning.strip():
                        # Preserve thinking as separate block (only if non-empty)
                        content.append({"type": "thinking", "thinking": reasoning})
                        if msg_content and msg_content.strip():
                            content.append({"type": "text", "text": msg_content})
                    elif reasoning and reasoning.strip():
                        # Merge for backward compatibility
                        content.append({"type": "text", "text": reasoning + msg_content})
                    else:
                        content.append({"type": "text", "text": msg_content})
            elif isinstance(msg_content, list):
                # Content parts array - handle text, output_text, and refusal
                for part in msg_content:
                    part_type = part.get("type", "")
                    if part_type in ("text", "output_text"):
                        text = part.get("text", "")
                        # Filter empty/whitespace-only text
                        if text and text.strip():
                            content.append({"type": "text", "text": text})
                    elif part_type == "refusal":
                        refusal = part.get("refusal", "")
                        # Filter empty/whitespace-only refusal
                        if refusal and refusal.strip():
                            content.append({"type": "text", "text": refusal})
                # Add thinking block if preserving (only if non-empty)
                if emit_separate_thinking and reasoning and reasoning.strip():
                    # Insert thinking at beginning
                    content.insert(0, {"type": "thinking", "thinking": reasoning})
                elif reasoning and reasoning.strip():
                    # Merge: prepend to first text block
                    text_blocks = [b for b in content if b["type"] == "text"]
                    if text_blocks:
                        text_blocks[0]["text"] = reasoning + text_blocks[0]["text"]
                    else:
                        content.append({"type": "text", "text": reasoning})
        elif reasoning and reasoning.strip():
            # Only reasoning content, no text (must be non-empty)
            if emit_separate_thinking:
                content.append({"type": "thinking", "thinking": reasoning})
                # C5 fix: 有 thinking 块但没有 text 块时，补充空 text 块
                # 防止某些 Anthropic SDK 客户端期望至少有一个 text 块时报错
                content.append({"type": "text", "text": ""})
            else:
                content.append({"type": "text", "text": reasoning})

        # Handle message-level refusal (some providers put it here)
        refusal = message.get("refusal")
        if refusal and isinstance(refusal, str) and refusal.strip():
            content.append({"type": "text", "text": refusal})

        # Handle tool_calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            has_tool_use = True
            # C5 fix: If we have tool_calls but no text block yet, insert empty text block
            # BEFORE tool_use for Anthropic protocol compliance
            # (Anthropic expects: thinking -> text -> tool_use order)
            has_text_block = any(b.get("type") == "text" for b in content)
            if not has_text_block:
                # Insert empty text block at the appropriate position
                # If we have thinking blocks, insert after them; otherwise at start
                thinking_blocks = [b for b in content if b.get("type") == "thinking"]
                if thinking_blocks:
                    # Insert after thinking blocks
                    insert_idx = len(thinking_blocks)
                else:
                    # Insert at start
                    insert_idx = 0
                content.insert(insert_idx, {"type": "text", "text": ""})

            for tc in tool_calls:
                tc_func = tc.get("function", {})
                tc_args = tc_func.get("arguments", "{}")
                tc_input = _safe_json_parse(
                    tc_args,
                    default={},
                    context="tool_call arguments"
                )
                # Restore original tool name if it was truncated
                tc_name = tc_func.get("name", "")
                original_name = restore_tool_name(tc_name, tool_name_mapping or {})
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": original_name,
                        "input": tc_input,
                    }
                )

        # Handle legacy function_call format
        if not has_tool_use:
            function_call = message.get("function_call")
            if function_call:
                fc_name = function_call.get("name", "")
                # Restore original tool name if it was truncated
                original_name = restore_tool_name(fc_name, tool_name_mapping or {})
                fc_args = function_call.get("arguments")
                fc_id = function_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                if fc_name or fc_args:
                    # C5 fix: Ensure text block exists before tool_use
                    has_text_block = any(b.get("type") == "text" for b in content)
                    if not has_text_block:
                        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
                        insert_idx = len(thinking_blocks) if thinking_blocks else 0
                        content.insert(insert_idx, {"type": "text", "text": ""})

                    fc_input = _safe_json_parse(
                        fc_args if isinstance(fc_args, str) else "{}",
                        default={},
                        context="function_call arguments"
                    )
                    content.append(
                        {
                            "type": "tool_use",
                            "id": fc_id,
                            "name": original_name,
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
    """深度转换对象为 dict，处理 Pydantic 模型嵌套未完全序列化的情况（如 BigModel SDK 的 ChoiceDelta）

    Performance optimization: Skip conversion if already dict with primitive values.
    """
    if isinstance(obj, dict):
        # Quick check: if all values are primitives, return directly
        for v in obj.values():
            if isinstance(v, (dict, list)) or hasattr(v, "model_dump") or hasattr(v, "to_dict"):
                # Has nested objects, need recursive conversion
                return {k: _to_dict(v) for k, v in obj.items()}
        # All primitives, return directly
        return obj
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
    preserve_thinking_blocks: bool = False,
    tool_name_mapping: dict[str, str] | None = None,
) -> AsyncGenerator[bytes, None]:
    """将 OpenAI 格式的 SSE 流实时转换为 Anthropic 格式。

    Args:
        openai_stream: OpenAI 格式的流式 chunk 生成器
        model: 模型名称
        request_id: 请求 ID
        thinking_enabled: 客户端是否请求了 thinking。
            False 时，reasoning_content 会被合并到 text 块中，不生成 thinking 块。
        preserve_thinking_blocks: If True, preserve reasoning_content as separate thinking
            block instead of merging into text.
        tool_name_mapping: Mapping of truncated tool names to original names.
                          Used to restore original names for tools that exceeded
                          OpenAI's 64-char limit.

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

    # Determine if to preserve thinking as separate block
    emit_separate_thinking = thinking_enabled or preserve_thinking_blocks

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

            # C2 fix: thinking 未启用且未preserve时，将 reasoning_content 合并到 content
            if reasoning_content is not None and not emit_separate_thinking:
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

                        # C5 fix: Insert empty text block BEFORE first tool_use for protocol compliance
                        # (Anthropic expects: thinking -> text -> tool_use order)
                        # This handles two cases:
                        # 1. thinking_block_opened but no text_block -> insert text after thinking
                        # 2. no thinking and no text -> insert text at start before tool_use
                        if not text_block_opened and not tool_calls_map:
                            text_block_idx = next_block_index
                            next_block_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": text_block_idx,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            yield _sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": text_block_idx},
                            )
                            text_block_opened = True  # Mark as opened

                        # H1 fix: 使用 next_block_index 而非计算式
                        block_idx = next_block_index
                        next_block_index += 1

                        tc_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        func = tc_delta.get("function", {})
                        tc_name = func.get("name", "")
                        # Restore original tool name if it was truncated
                        original_name = restore_tool_name(tc_name, tool_name_mapping or {})

                        tool_calls_map[tc_index] = {
                            "id": tc_id,
                            "name": original_name,
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
                                    "name": original_name,
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

                    # Build thinking content_block
                    # Note: OpenAI models don't provide signatures for extended thinking.
                    # Anthropic protocol expects signature field, but empty signature is accepted.
                    # We omit the signature field entirely for OpenAI providers to avoid
                    # potential issues with strict Anthropic clients that expect valid signatures.
                    # Anthropic-native upstream providers will provide signature via signature_delta.
                    thinking_block = {"type": "thinking", "thinking": ""}
                    # Only include signature if upstream provides it (via signature_delta)
                    # For OpenAI providers, signature field is omitted entirely

                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": thinking_block,
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
            # When signature_delta is received, we track it but don't emit a separate event
            # The signature will be implicitly part of the thinking block state
            signature_delta = delta.get("signature")
            if signature_delta is not None and thinking_block_opened and emit_separate_thinking:
                # Track the signature - Anthropic clients use this for verification
                thinking_signature = signature_delta
                # Note: We don't emit signature_delta as a separate SSE event
                # because the signature is already set via content_block_start or
                # will be valid at message_stop. Anthropic SDK handles this correctly.

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

    except asyncio.CancelledError:
        # Client disconnected - propagate cancellation without sending error event
        logger.debug(f"[{request_id}] Anthropic stream conversion cancelled by client")
        raise
    except Exception as e:
        logger.error(
            f"[{request_id}] openai_stream_to_anthropic error: "
            f"{type(e).__name__}: {str(e)[:200]}"
        )
        # Don't try to send error SSE - client may have disconnected
        # Just re-raise so upstream can handle
        raise
    finally:
        # 关闭仍打开的块
        close_ev = _close_open_block()
        if close_ev:
            yield close_ev

        if sent_message_start:
            # C4 fix: 如果没有任何块被打开，合成一个空 text 块保证协议合规
            # C5 fix: 如果只有 thinking 块被打开但没有 text 块，也合成一个空 text 块
            # 防止客户端 SDK 在处理只有 thinking 块的响应时尝试访问 .text 属性报错
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
            elif thinking_block_opened and not text_block_opened:
                # 有 thinking 块但没有 text 块 - 补一个空 text 块防止客户端报错
                # 某些 Anthropic SDK 客户端期望每个消息至少有一个 text 块
                block_idx = next_block_index
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield _sse(
                    "content_block_stop", {"type": "content_block_stop", "index": block_idx}
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
