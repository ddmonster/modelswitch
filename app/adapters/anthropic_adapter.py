from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from openai import APIStatusError, APITimeoutError, APIConnectionError

from app.adapters.base import AdapterResponse, BaseAdapter, create_error_response
from app.models.config_models import ProviderConfig
from app.utils.logging import get_adapter_logger

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """基于 anthropic SDK 的适配器，内部做 OpenAI ↔ Anthropic 格式互转"""

    def __init__(self, provider_config: ProviderConfig):
        super().__init__(provider_config)
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {"api_key": provider_config.api_key}
        if provider_config.base_url:
            kwargs["base_url"] = provider_config.base_url
        self._client = AsyncAnthropic(**kwargs)

    async def chat_completion(
        self,
        model_name: str,
        messages: list,
        stream: bool = False,
        timeout: int = 60,
        request_id: str = "",
        **kwargs: Any,
    ) -> AdapterResponse:
        start = time.monotonic()

        logger.debug(
            f"[{request_id}] api_call provider={self.name} "
            f"model={model_name} api_base={self.provider.base_url} "
            f"stream={stream} timeout={timeout}"
        )

        adapter_logger = get_adapter_logger(self.name, request_id)

        # Log request details
        adapter_logger.log_request(
            model=model_name,
            stream=stream,
            timeout=timeout,
            messages_count=len(messages),
            params=kwargs,
        )

        try:
            create_kwargs = self._prepare_create_kwargs(messages, model_name, timeout, **kwargs)

            adapter_logger.debug(
                f"api_call_start",
                model=model_name,
                stream=stream,
                kwargs_keys=list(create_kwargs.keys()),
            )

            try:
                if stream:
                    return await self._stream(
                        create_kwargs, model_name, start, request_id, adapter_logger
                    )
                else:
                    return await self._non_stream(
                        create_kwargs, model_name, start, request_id, adapter_logger
                    )
            except AnthropicAPITimeoutError as e:
                raise APITimeoutError(str(e))
            except AnthropicAPIConnectionError as e:
                raise APIConnectionError(str(e))
            except AnthropicAPIStatusError as e:
                raise APIStatusError(
                    str(e),
                    response=e.response,
                    body=e.body,  # type: ignore[arg-type]
                )

        except APITimeoutError as e:
            latency = (time.monotonic() - start) * 1000
            adapter_logger.log_error(
                model=model_name,
                error_type="timeout",
                error_message=str(e),
                latency_ms=latency,
                status_code=504,
            )
            return create_error_response(
                self.name, model_name, latency, 504, str(e), request_id
            )

        except APIConnectionError as e:
            latency = (time.monotonic() - start) * 1000
            adapter_logger.log_error(
                model=model_name,
                error_type="connection",
                error_message=str(e),
                latency_ms=latency,
                status_code=503,
            )
            return create_error_response(
                self.name, model_name, latency, 503, str(e), request_id
            )

        except APIStatusError as e:
            latency = (time.monotonic() - start) * 1000
            adapter_logger.log_error(
                model=model_name,
                error_type="api_status",
                error_message=str(e),
                latency_ms=latency,
                status_code=e.status_code,
                upstream_response=str(e.response) if e.response else "",
            )
            return create_error_response(
                self.name, model_name, latency, e.status_code, str(e), request_id
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            adapter_logger.log_error(
                model=model_name,
                error_type="unexpected",
                error_message=f"{type(e).__name__}: {str(e)}",
                latency_ms=latency,
                status_code=502,
            )
            logger.exception(f"[{request_id}] Unexpected error in Anthropic adapter")
            return create_error_response(
                self.name, model_name, latency, 502, str(e), request_id
            )

    def _prepare_create_kwargs(
        self,
        messages: list,
        model_name: str,
        timeout: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for Anthropic messages.create()."""
        # Convert OpenAI format messages to Anthropic format
        system, anth_messages = _openai_to_anthropic_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": anth_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "timeout": timeout,
        }
        if system:
            create_kwargs["system"] = system

        # Standard parameters
        for p in ("temperature", "top_p", "top_k"):
            if p in kwargs:
                create_kwargs[p] = kwargs[p]
        if "stop" in kwargs:
            create_kwargs["stop_sequences"] = kwargs["stop"]

        # Convert tools
        if "tools" in kwargs:
            create_kwargs["tools"] = [
                {
                    "name": t.get("function", {}).get("name", ""),
                    "description": t.get("function", {}).get("description", ""),
                    "input_schema": t.get("function", {}).get("parameters", {}),
                }
                for t in kwargs["tools"]
            ]

        # Convert tool_choice
        if "tool_choice" in kwargs:
            tc = kwargs["tool_choice"]
            if tc == "auto":
                create_kwargs["tool_choice"] = {"type": "auto"}
            elif tc == "required":
                create_kwargs["tool_choice"] = {"type": "any"}
            elif isinstance(tc, dict):
                name = tc.get("function", {}).get("name", "")
                create_kwargs["tool_choice"] = {"type": "tool", "name": name}

        return create_kwargs

    async def _non_stream(
        self,
        create_kwargs: dict,
        model_name: str,
        start: float,
        request_id: str,
        adapter_logger: Any,
    ) -> AdapterResponse:
        response = await self._client.messages.create(**create_kwargs)
        latency = (time.monotonic() - start) * 1000

        openai_resp = _anthropic_response_to_openai(response)
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        # Log successful response
        adapter_logger.log_response_start(
            model=model_name,
            latency_ms=latency,
            status_code=200,
            usage=usage,
        )

        # Log response structure at debug level
        content_preview = ""
        tool_calls_count = 0
        for block in response.content:
            if block.type == "text":
                content_preview = block.text[:100] if block.text else ""
            elif block.type == "tool_use":
                tool_calls_count += 1

        adapter_logger.debug(
            f"response_structure",
            content_preview=content_preview,
            tool_calls=tool_calls_count,
            finish_reason=response.stop_reason,
        )

        return AdapterResponse(
            status_code=200,
            success=True,
            body=openai_resp,
            adapter_name=self.name,
            model_name=model_name,
            latency_ms=latency,
            usage=usage,
        )

    async def _stream(
        self,
        create_kwargs: dict,
        model_name: str,
        start: float,
        request_id: str,
        adapter_logger: Any,
    ) -> AdapterResponse:
        raw_stream = await self._client.messages.create(stream=True, **create_kwargs)
        resp_ref = None  # 用于在闭包中引用 AdapterResponse

        async def stream_generator():
            tool_call_index = -1
            chunk_count = 0
            try:
                async for event in raw_stream:
                    # 从 Anthropic 流式事件中捕获 usage
                    if event.type == "message_start" and hasattr(event, "message"):
                        if hasattr(event.message, "usage"):
                            input_tokens = getattr(event.message.usage, "input_tokens", 0) or 0
                            if resp_ref is not None:
                                resp_ref.usage = {
                                    "prompt_tokens": input_tokens,
                                    "completion_tokens": 0,
                                }
                    elif event.type == "message_delta" and hasattr(event, "usage"):
                        output_tokens = getattr(event.usage, "output_tokens", 0) or 0
                        if resp_ref is not None:
                            current = resp_ref.usage or {}
                            resp_ref.usage = {
                                "prompt_tokens": current.get("prompt_tokens", 0),
                                "completion_tokens": output_tokens,
                            }

                    chunk = _anthropic_event_to_openai_chunk(
                        event, model_name, tool_call_index
                    )
                    if chunk is not None:
                        if isinstance(chunk, dict) and "_tc_index_bump" in chunk:
                            tool_call_index = chunk.pop("_tc_index_bump")
                        chunk_count += 1
                        yield chunk

                elapsed = (time.monotonic() - start) * 1000

                # Log stream completion
                adapter_logger.log_stream_complete(
                    model=model_name,
                    total_chunks=chunk_count,
                    latency_ms=elapsed,
                    usage=resp_ref.usage if resp_ref else None,
                )

            except asyncio.CancelledError:
                # Client disconnected - clean up and propagate cancellation
                elapsed = (time.monotonic() - start) * 1000
                adapter_logger.debug(
                    f"stream_cancelled",
                    model=model_name,
                    chunks=chunk_count,
                    latency_ms=f"{elapsed:.0f}",
                )
                logger.debug(f"[{request_id}] Stream cancelled by client")
                raise  # Must re-raise CancelledError

            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                adapter_logger.log_error(
                    model=model_name,
                    error_type="stream",
                    error_message=f"{type(e).__name__}: {str(e)}",
                    latency_ms=elapsed,
                    status_code=502,
                )
                logger.exception(f"[{request_id}] Stream error in Anthropic adapter")
                raise

        latency = (time.monotonic() - start) * 1000
        adapter_resp = AdapterResponse(
            status_code=200,
            success=True,
            stream=stream_generator(),
            adapter_name=self.name,
            model_name=model_name,
            latency_ms=latency,
        )
        resp_ref = adapter_resp
        return adapter_resp


# ========== 格式转换工具函数 ==========


def _openai_to_anthropic_messages(messages: list):
    """将 OpenAI 格式消息转换为 Anthropic 格式。返回 (system, messages)。"""
    system_parts: list[dict] = []
    anthropic_messages: list[dict] = []

    for msg in messages:
        role = (
            msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        )
        content = (
            msg.get("content", "")
            if isinstance(msg, dict)
            else getattr(msg, "content", "")
        )

        if role == "system":
            system_parts.append({"type": "text", "text": content or ""})
        elif role == "user":
            anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls = (
                msg.get("tool_calls")
                if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None)
            ) or []
            if tool_calls:
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = (
                        tc.get("function", {}) if isinstance(tc, dict) else tc.function
                    )
                    args = (
                        func.get("arguments", "{}")
                        if isinstance(func, dict)
                        else func.arguments
                    )
                    name = func.get("name", "") if isinstance(func, dict) else func.name
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else tc.id
                    try:
                        input_data = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": name,
                            "input": input_data,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                anthropic_messages.append(
                    {"role": "assistant", "content": content or ""}
                )
        elif role == "tool":
            tool_call_id = (
                msg.get("tool_call_id", "")
                if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content or "",
                        }
                    ],
                }
            )

    return system_parts or None, anthropic_messages


def _anthropic_response_to_openai(response) -> dict:
    """将 Anthropic Messages 响应转换为 OpenAI ChatCompletion 格式 dict。"""
    content_text = ""
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            content_text = block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    finish_reason = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(response.stop_reason, "stop")

    message: dict[str, Any] = {"role": "assistant", "content": content_text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": response.id,
        "object": "chat.completion",
        "model": response.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    }


def _anthropic_event_to_openai_chunk(event, model: str, current_tc_index: int):
    """将 Anthropic 流式事件转换为 OpenAI chunk dict。返回 None 表示跳过。"""
    event_type = event.type

    if event_type == "content_block_start":
        block = event.content_block
        if block.type == "text":
            return _make_chunk(model, content="")
        elif block.type == "thinking":
            return None  # thinking blocks 不需要转换为 OpenAI 格式
        elif block.type == "tool_use":
            new_index = current_tc_index + 1
            chunk = _make_chunk(
                model,
                tool_call={
                    "index": new_index,
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": ""},
                },
            )
            chunk["_tc_index_bump"] = new_index
            return chunk

    elif event_type == "content_block_delta":
        delta = event.delta
        if delta.type == "text_delta":
            return _make_chunk(model, content=delta.text)
        elif delta.type == "thinking_delta":
            return None  # thinking deltas 不需要转换为 OpenAI 格式
        elif delta.type == "input_json_delta":
            return _make_chunk(
                model,
                tool_call={
                    "index": max(current_tc_index, 0),
                    "function": {"arguments": delta.partial_json},
                },
            )

    elif event_type == "message_delta":
        stop_reason = event.delta.stop_reason
        finish_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }.get(stop_reason, "stop")
        return _make_chunk(model, finish_reason=finish_reason)

    return None


def _make_chunk(
    model: str,
    content: str | None = None,
    tool_call: dict | None = None,
    finish_reason: str | None = None,
) -> dict:
    """创建 OpenAI 格式的流式 chunk dict。"""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_call is not None:
        delta["tool_calls"] = [tool_call]
    return {
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
