from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import APIStatusError, APITimeoutError

from app.adapters.base import AdapterResponse, BaseAdapter
from app.models.config_models import ProviderConfig

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
        from anthropic import APIStatusError as AStatus
        from anthropic import APITimeoutError as ATimeout

        start = time.monotonic()

        logger.debug(
            f"[{request_id}] api_call provider={self.name} "
            f"model={model_name} api_base={self.provider.base_url} "
            f"stream={stream} timeout={timeout}"
        )

        try:
            # 将 OpenAI 格式消息转换为 Anthropic 格式
            system, anth_messages = _openai_to_anthropic_messages(messages)

            create_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": anth_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "timeout": timeout,
            }
            if system:
                create_kwargs["system"] = system
            for p in ("temperature", "top_p", "top_k"):
                if p in kwargs:
                    create_kwargs[p] = kwargs[p]
            if "stop" in kwargs:
                create_kwargs["stop_sequences"] = kwargs["stop"]

            # 转换 tools
            if "tools" in kwargs:
                create_kwargs["tools"] = [
                    {
                        "name": t.get("function", {}).get("name", ""),
                        "description": t.get("function", {}).get("description", ""),
                        "input_schema": t.get("function", {}).get("parameters", {}),
                    }
                    for t in kwargs["tools"]
                ]

            # 转换 tool_choice
            if "tool_choice" in kwargs:
                tc = kwargs["tool_choice"]
                if tc == "auto":
                    create_kwargs["tool_choice"] = {"type": "auto"}
                elif tc == "required":
                    create_kwargs["tool_choice"] = {"type": "any"}
                elif isinstance(tc, dict):
                    name = tc.get("function", {}).get("name", "")
                    create_kwargs["tool_choice"] = {"type": "tool", "name": name}

            try:
                if stream:
                    return await self._stream(
                        create_kwargs, model_name, start, request_id
                    )
                else:
                    return await self._non_stream(
                        create_kwargs, model_name, start, request_id
                    )
            except ATimeout as e:
                raise APITimeoutError(str(e))
            except AStatus as e:
                raise APIStatusError(
                    str(e),
                    response=e.response,
                    body=e.body,  # type: ignore[arg-type]
                )

        except APITimeoutError as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning(
                f"[{request_id}] api_timeout provider={self.name} latency={latency:.0f}ms"
            )
            return AdapterResponse(
                status_code=504,
                success=False,
                adapter_name=self.name,
                model_name=model_name,
                latency_ms=latency,
                error=str(e),
            )
        except APIStatusError as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning(
                f"[{request_id}] api_error provider={self.name} "
                f"status={e.status_code} error={e}"
            )
            return AdapterResponse(
                status_code=e.status_code,
                success=False,
                adapter_name=self.name,
                model_name=model_name,
                latency_ms=latency,
                error=str(e),
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"[{request_id}] api_exception provider={self.name} error={e}")
            return AdapterResponse(
                status_code=502,
                success=False,
                adapter_name=self.name,
                model_name=model_name,
                latency_ms=latency,
                error=str(e),
            )

    async def _non_stream(
        self, create_kwargs: dict, model_name: str, start: float, request_id: str
    ) -> AdapterResponse:
        response = await self._client.messages.create(**create_kwargs)
        latency = (time.monotonic() - start) * 1000

        openai_resp = _anthropic_response_to_openai(response)
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        logger.debug(
            f"[{request_id}] api_response provider={self.name} "
            f"status=200 latency={latency:.0f}ms usage={usage}"
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
        self, create_kwargs: dict, model_name: str, start: float, request_id: str
    ) -> AdapterResponse:
        raw_stream = await self._client.messages.create(stream=True, **create_kwargs)
        resp_ref = None  # 用于在闭包中引用 AdapterResponse

        async def stream_generator():
            tool_call_index = -1
            try:
                chunk_count = 0
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
                logger.debug(
                    f"[{request_id}] stream_complete provider={self.name} "
                    f"chunks={chunk_count} elapsed={elapsed:.0f}ms"
                )
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                logger.error(
                    f"[{request_id}] stream_error provider={self.name} "
                    f"error={e} elapsed={elapsed:.0f}ms"
                )
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
