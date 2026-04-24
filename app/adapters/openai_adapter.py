from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import AsyncOpenAI, APITimeoutError, APIStatusError, APIConnectionError

from app.adapters.base import AdapterResponse, BaseAdapter, create_error_response
from app.models.config_models import ProviderConfig
from app.core.request_queue import get_queue_manager
from app.utils.logging import get_adapter_logger, get_protocol_logger

logger = logging.getLogger(__name__)

# OpenAI chat completion API 标准参数（会被直接传递给 create()）
# 非标准参数会通过 extra_body 传递给上游（vLLM 支持）
_OPENAI_STANDARD_PARAMS = {
    # 核心参数
    "max_tokens", "temperature", "top_p", "stop", "seed", "n",
    # 工具调用
    "tools", "tool_choice",
    # 格式控制
    "response_format", "stream_options", "max_completion_tokens",
    # 惩罚参数
    "frequency_penalty", "presence_penalty",
    # 日志概率
    "logprobs", "top_logprobs",
    # 其他标准参数
    "logit_bias", "metadata",
    # 用户标识
    "user",
}

# Provider-specific message field handling
# Some providers don't support reasoning_content in request messages
_UNSUPPORTED_MESSAGE_FIELDS = {
    # reasoning_content is only supported by certain providers (DeepSeek, OpenAI o-series)
    # vLLM and most OpenAI-compatible providers don't support it
    "reasoning_content",
}


def _filter_message_fields(messages: list, provider_name: str) -> list:
    """Filter unsupported fields from messages for provider compatibility.

    Some providers (vLLM, GLM, etc.) don't recognize fields like reasoning_content
    and may reject requests or misinterpret them.

    Args:
        messages: List of message dicts
        provider_name: Provider name for logging

    Returns:
        Filtered messages list
    """
    filtered = []
    for msg in messages:
        filtered_msg = {}
        for key, value in msg.items():
            if key in _UNSUPPORTED_MESSAGE_FIELDS:
                logger.debug(
                    f"Filtering unsupported field '{key}' from message for provider {provider_name}"
                )
                continue
            filtered_msg[key] = value
        filtered.append(filtered_msg)
    return filtered


def _make_reasoning_chunk(ref_chunk, model: str, reasoning_text: str):
    """构造一个包含完整推理内容的合成流式 chunk（dict 格式）"""
    if hasattr(ref_chunk, "id"):
        chunk_id = ref_chunk.id
        created = ref_chunk.created
    else:
        chunk_id = ref_chunk.get("id", "")
        created = ref_chunk.get("created", 0)
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "reasoning_content": reasoning_text},
        }],
    }


class OpenAIAdapter(BaseAdapter):
    """基于 openai SDK 的适配器，兼容 DashScope / GLM / OpenAI 等 provider"""

    def __init__(self, provider_config: ProviderConfig):
        super().__init__(provider_config)
        self._client = AsyncOpenAI(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
        )
        self._queue_manager = get_queue_manager()
        self._use_queue = provider_config.max_concurrent > 0
        
        # 如果配置了并发限制，注册到队列管理器
        if self._use_queue:
            self._queue_manager.register_provider(
                provider_name=provider_config.name,
                max_concurrent=provider_config.max_concurrent,
                max_queue_size=provider_config.max_queue_size,
                queue_timeout=provider_config.queue_timeout,
            )

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

        # 如果配置了队列，通过队列执行
        if self._use_queue:
            logger.debug(f"[{request_id}] 使用请求队列: {self.name}")
            return await self._queue_manager.execute(
                self.name,
                self._do_chat_completion,
                model_name, messages, stream, timeout, request_id,
                **kwargs
            )

        # 直接执行
        return await self._do_chat_completion(
            model_name, messages, stream, timeout, request_id, **kwargs
        )

    async def _do_chat_completion(
        self,
        model_name: str,
        messages: list,
        stream: bool = False,
        timeout: int = 60,
        request_id: str = "",
        **kwargs: Any,
    ) -> AdapterResponse:
        start = time.monotonic()
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
            create_kwargs = self._build_create_kwargs(
                model_name, messages, stream, timeout, request_id, **kwargs
            )

            # Log create_kwargs at debug level
            adapter_logger.debug(
                f"api_call_start",
                model=model_name,
                stream=stream,
                kwargs_keys=list(create_kwargs.keys()),
            )

            response = await self._client.chat.completions.create(**create_kwargs)

            if stream:
                return await self._create_stream_response(
                    response, model_name, start, request_id, adapter_logger
                )
            else:
                return self._create_non_stream_response(
                    response, model_name, start, request_id, adapter_logger
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
            # Get detailed upstream error info
            upstream_detail = ""
            if e.response:
                try:
                    # Try to get response body as text
                    if hasattr(e.response, "text"):
                        upstream_detail = e.response.text[:500]
                    elif hasattr(e.response, "content"):
                        upstream_detail = str(e.response.content)[:500]
                except Exception:
                    upstream_detail = str(e.response)[:300]

            adapter_logger.log_error(
                model=model_name,
                error_type="api_status",
                error_message=str(e),
                latency_ms=latency,
                status_code=e.status_code,
                upstream_response=upstream_detail,
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
            logger.exception(f"[{request_id}] Unexpected error in OpenAI adapter")
            return create_error_response(
                self.name, model_name, latency, 502, str(e), request_id
            )

    def _build_create_kwargs(
        self,
        model_name: str,
        messages: list,
        stream: bool,
        timeout: int,
        request_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for client.chat.completions.create()."""
        # 创建协议日志器
        protocol_logger = get_protocol_logger(request_id, "openai")

        # Filter unsupported fields from messages for provider compatibility
        # reasoning_content is not recognized by most OpenAI-compatible providers
        filtered_messages = _filter_message_fields(messages, self.name)

        # Separate standard params from extension params
        standard_params: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _OPENAI_STANDARD_PARAMS:
                standard_params[k] = v
            else:
                extra_body[k] = v

        create_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": filtered_messages,
            "stream": stream,
            "timeout": timeout,
            **standard_params,
        }

        # Auto-enable usage for streaming (unless provider doesn't support)
        stream_options_applied = None
        stream_options_disabled_reason = ""
        if stream and "stream_options" not in create_kwargs:
            if self.provider.disable_stream_options:
                stream_options_disabled_reason = "provider.disable_stream_options=true (vLLM compatibility)"
                protocol_logger.log_stream_options(
                    requested=None,
                    applied=None,
                    disabled_reason=stream_options_disabled_reason,
                )
            else:
                create_kwargs["stream_options"] = {"include_usage": True}
                stream_options_applied = create_kwargs["stream_options"]
                protocol_logger.log_stream_options(
                    requested=None,
                    applied=stream_options_applied,
                    disabled_reason="",
                )
        elif stream and "stream_options" in create_kwargs:
            # 用户已提供 stream_options
            protocol_logger.log_stream_options(
                requested=create_kwargs["stream_options"],
                applied=create_kwargs["stream_options"],
                disabled_reason="",
            )

        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if self.provider.custom_headers:
            create_kwargs["extra_headers"] = self.provider.custom_headers

        # 记录上游请求参数
        protocol_logger.log_upstream_request(
            adapter=self.name,
            model=model_name,
            stream=stream,
            create_kwargs_keys=list(create_kwargs.keys()),
            extra_body_keys=list(extra_body.keys()) if extra_body else None,
            extra_headers_keys=list(self.provider.custom_headers.keys()) if self.provider.custom_headers else None,
        )

        # Debug level: 记录完整的 kwargs 结构
        if standard_params:
            protocol_logger.debug(
                f"standard_params_detail",
                keys=list(standard_params.keys()),
            )
        if extra_body:
            protocol_logger.debug(
                f"extra_body_detail",
                keys=list(extra_body.keys()),
            )

        return create_kwargs

    async def _create_stream_response(
        self,
        response,
        model_name: str,
        start: float,
        request_id: str,
        adapter_logger: Any,
    ) -> AdapterResponse:
        """Create AdapterResponse with stream generator."""
        protocol_logger = get_protocol_logger(request_id, "openai")
        resp_ref = None  # Reference for AdapterResponse in closure

        async def stream_generator():
            try:
                chunk_count = 0
                content_chars = 0
                reasoning_buffer = []
                reasoning_flushed = False
                model_id = model_name
                tool_calls_count = 0
                first_chunk_logged = False

                async for chunk in response:
                    chunk_count += 1
                    # Capture usage from final chunk
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        if resp_ref is not None:
                            resp_ref.usage = {
                                "prompt_tokens": chunk.usage.prompt_tokens or 0,
                                "completion_tokens": chunk.usage.completion_tokens or 0,
                            }

                    if hasattr(chunk, "choices") and chunk.choices:
                        choice = chunk.choices[0]
                        # Handle dict or SDK object delta
                        if isinstance(choice, dict):
                            delta = choice.get("delta", {})
                            rc = delta.get("reasoning_content")
                            ct = delta.get("content")
                            # 检查 tool_calls delta
                            tc_delta = delta.get("tool_calls")
                        else:
                            delta = getattr(choice, "delta", None)
                            if delta is None:
                                continue
                            rc = getattr(delta, "reasoning_content", None)
                            ct = getattr(delta, "content", None)
                            tc_delta = getattr(delta, "tool_calls", None)

                        # 记录首个 chunk 的格式（用于排查协议问题）
                        if not first_chunk_logged:
                            delta_keys = list(delta.keys()) if isinstance(delta, dict) else []
                            has_usage = hasattr(chunk, "usage") and chunk.usage is not None
                            protocol_logger.log_chunk_format(
                                chunk_index=0,
                                has_delta=True,
                                delta_keys=delta_keys,
                                has_tool_calls_delta=tc_delta is not None and len(tc_delta) > 0,
                                has_usage=has_usage,
                            )
                            first_chunk_logged = True

                        if rc:
                            reasoning_buffer.append(rc)
                        if ct:
                            # Flush reasoning on first content
                            if reasoning_buffer and not reasoning_flushed:
                                reasoning_text = "".join(reasoning_buffer)
                                content_chars += len(reasoning_text)
                                yield _make_reasoning_chunk(chunk, model_id, reasoning_text)
                                reasoning_flushed = True
                            content_chars += len(ct)
                            yield chunk
                        elif rc:
                            continue  # Buffer only, don't send
                        else:
                            yield chunk  # Finish reason chunk
                    else:
                        yield chunk

                # Flush remaining reasoning at end
                if reasoning_buffer and not reasoning_flushed:
                    reasoning_text = "".join(reasoning_buffer)
                    content_chars += len(reasoning_text)
                    try:
                        yield _make_reasoning_chunk(chunk, model_id, reasoning_text)
                    except Exception as e:
                        adapter_logger.log_parse_error(
                            model=model_name,
                            parse_stage="flush_reasoning",
                            error_message=str(e),
                            raw_data_preview=reasoning_text[:100],
                        )

                elapsed = (time.monotonic() - start) * 1000
                # Fallback: estimate tokens if usage not reported
                if resp_ref is not None and resp_ref.usage is None and content_chars > 0:
                    estimated = max(1, content_chars // 3)
                    adapter_logger.debug(
                        f"usage_estimate",
                        estimated_tokens=estimated,
                        content_chars=content_chars,
                    )
                    resp_ref.usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": estimated,
                    }

                # Log stream completion
                adapter_logger.log_stream_complete(
                    model=model_name,
                    total_chunks=chunk_count,
                    latency_ms=elapsed,
                    usage=resp_ref.usage if resp_ref else None,
                )

                # 协议日志：流式响应格式总结
                protocol_logger.log_response_format(
                    response_type="chat.completion.chunk",
                    has_tool_calls=tool_calls_count > 0,
                    has_reasoning=len(reasoning_buffer) > 0,
                    finish_reason="stream_complete",
                    content_preview=f"{chunk_count} chunks, {content_chars} chars",
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
                logger.exception(f"[{request_id}] Stream error in OpenAI adapter")
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

    def _create_non_stream_response(
        self,
        response,
        model_name: str,
        start: float,
        request_id: str,
        adapter_logger: Any,
    ) -> AdapterResponse:
        """Create AdapterResponse for non-stream response."""
        protocol_logger = get_protocol_logger(request_id, "openai")
        latency = (time.monotonic() - start) * 1000
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
            }

        # Log response
        adapter_logger.log_response_start(
            model=model_name,
            latency_ms=latency,
            status_code=200,
            usage=usage,
        )

        # Log response structure at debug level
        has_tool_calls = False
        has_reasoning = False
        finish_reason = ""
        content_preview = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            # Handle both SDK object and dict (for mocked tests)
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                content_preview = msg.get("content", "")[:100] if msg.get("content") else ""
                tool_calls_count = len(msg.get("tool_calls", []) or [])
                has_tool_calls = tool_calls_count > 0
                # 检查 reasoning_content
                has_reasoning = "reasoning_content" in msg and msg["reasoning_content"]
                finish_reason = choice.get("finish_reason", "")
            else:
                msg = getattr(choice, "message", None)
                if msg:
                    content_preview = ""
                    if hasattr(msg, "content") and msg.content:
                        content_preview = msg.content[:100]
                    tool_calls_count = len(getattr(msg, "tool_calls", []) or [])
                    has_tool_calls = tool_calls_count > 0
                    # 检查 reasoning_content
                    has_reasoning = hasattr(msg, "reasoning_content") and getattr(msg, "reasoning_content", None)
                    finish_reason = getattr(choice, "finish_reason", "")
                else:
                    content_preview = ""
                    tool_calls_count = 0
                    finish_reason = ""

            adapter_logger.debug(
                f"response_structure",
                content_preview=content_preview,
                tool_calls=tool_calls_count,
                finish_reason=finish_reason,
            )

            # 协议日志：响应格式分析
            protocol_logger.log_response_format(
                response_type="chat.completion",
                has_tool_calls=has_tool_calls,
                has_reasoning=has_reasoning,
                finish_reason=finish_reason,
                content_preview=content_preview,
            )

        return AdapterResponse(
            status_code=200,
            success=True,
            body=response,
            adapter_name=self.name,
            model_name=model_name,
            latency_ms=latency,
            usage=usage,
        )
