"""Usage tracking and logging helpers for route handlers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

_conv_logger = logging.getLogger("modelswitch.conversations")


class StreamAccumulator:
    """Accumulate streaming output for tracking/logging.

    Collects text content and tool calls from OpenAI stream chunks,
    providing a summary suitable for conversation logging.
    """

    def __init__(self):
        self.collected_text: list[str] = []
        self.collected_tool_calls: dict[int, dict] = {}  # index -> {"name", "arguments"}

    def process_chunk(self, chunk_data) -> None:
        """Process a chunk and accumulate its content.

        Args:
            chunk_data: OpenAI chunk dict or object with .model_dump() method
        """
        if chunk_data is None:
            return
        # Convert to dict if needed
        if hasattr(chunk_data, "model_dump"):
            chunk_data = chunk_data.model_dump(exclude_none=True)
        elif hasattr(chunk_data, "to_dict"):
            chunk_data = chunk_data.to_dict()
        if not isinstance(chunk_data, dict):
            return

        choices = chunk_data.get("choices", [])
        if not choices:
            return
        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            return

        # Accumulate text content
        if delta.get("content"):
            self.collected_text.append(delta["content"])
        elif delta.get("reasoning_content"):
            self.collected_text.append(delta["reasoning_content"])

        # Accumulate tool calls
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            func = tc.get("function", {})
            if idx not in self.collected_tool_calls:
                self.collected_tool_calls[idx] = {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                }
            else:
                if func.get("name"):
                    self.collected_tool_calls[idx]["name"] = func["name"]
                if func.get("arguments"):
                    self.collected_tool_calls[idx]["arguments"] += func["arguments"]

    def get_output_summary(self) -> list[dict] | None:
        """Build output summary for tracking.

        Returns:
            List of content parts (text + tool_use), or None if empty
        """
        parts = []
        if self.collected_text:
            parts.append({"type": "text", "text": "".join(self.collected_text)})
        for idx in sorted(self.collected_tool_calls):
            tc = self.collected_tool_calls[idx]
            parts.append({
                "type": "tool_use",
                "name": tc["name"],
                "arguments": tc["arguments"],
            })
        return parts if parts else None


def _extract_usage(result):
    """从 AdapterResponse 中提取 token 用量"""
    tokens_in = 0
    tokens_out = 0
    if result.usage:
        tokens_in = result.usage.get("prompt_tokens", 0)
        tokens_out = result.usage.get("completion_tokens", 0)
    elif result.success and result.body:
        body = result.body
        if hasattr(body, "model_dump"):
            body_dict = body.model_dump(exclude_none=True)
        elif isinstance(body, dict):
            body_dict = body
        else:
            body_dict = {}
        usage = body_dict.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
    return tokens_in, tokens_out


def _extract_output(result):
    """从 AdapterResponse 中提取输出内容（非流式）"""
    if not result.success or not result.body:
        return None
    body = result.body
    if hasattr(body, "model_dump"):
        body = body.model_dump(exclude_none=True)
    elif not isinstance(body, dict):
        return None
    choices = body.get("choices", [])
    if not choices:
        return None
    msg = choices[0].get("message", {})
    parts = []
    if msg.get("content"):
        parts.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        parts.append(
            {
                "type": "tool_use",
                "name": func.get("name", ""),
                "arguments": func.get("arguments", ""),
            }
        )
    return parts or None


async def track_request(
    app_state,
    request_id,
    model,
    result,
    api_key_alias="",
    messages=None,
    stream_output=None,
):
    """记录请求的用量统计和日志。在路由处理完成后调用。"""
    from app.utils.logging import add_log_to_buffer

    tokens_in, tokens_out = _extract_usage(result)
    provider = getattr(result, "adapter_name", "") or "unknown"
    latency = getattr(result, "latency_ms", 0) or 0

    # 记录用量统计
    usage_tracker = getattr(app_state, "usage_tracker", None)
    if usage_tracker:
        await usage_tracker.record(
            provider=provider,
            model=model,
            api_key_alias=api_key_alias,
            success=result.success,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            status_code=result.status_code,
        )

    # 记录到内存日志缓冲
    level = "INFO" if result.success else "ERROR"
    msg = (
        f"model={model} adapter={provider} "
        f"status={result.status_code} latency={latency:.0f}ms "
        f"tokens={tokens_in}+{tokens_out}"
    )
    add_log_to_buffer(request_id, level, msg, api_key=api_key_alias)

    # 写入会话日志（conversations.jsonl）
    if messages is not None:
        output = stream_output if stream_output is not None else _extract_output(result)
        record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model": model,
            "adapter": provider,
            "api_key": api_key_alias,
            "success": result.success,
            "latency_ms": round(latency),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "messages": messages,
            "output": output,
        }
        if not result.success:
            record["error"] = result.error
        try:
            _conv_logger.info(json.dumps(record, ensure_ascii=False))
            # Also index the record for fast queries
            try:
                from app.utils.logging import get_conv_handler

                handler = get_conv_handler()
                if handler:
                    from app.services.conv_indexer import get_conv_indexer

                    indexer = get_conv_indexer()
                    if indexer:
                        raw_line = json.dumps(record, ensure_ascii=False)
                        indexer.index(
                            record=record,
                            file_path=Path(handler.current_base_filename).name,
                            byte_offset=handler.last_byte_offset,
                            line_length=len(raw_line.encode("utf-8")),
                        )
            except Exception:
                pass  # Indexing failure should not break request logging
        except Exception:
            pass
