from __future__ import annotations

import logging
import time
from typing import Any

from openai import AsyncOpenAI, APITimeoutError, APIStatusError

from app.adapters.base import AdapterResponse, BaseAdapter
from app.models.config_models import ProviderConfig
from app.core.request_queue import get_queue_manager

logger = logging.getLogger(__name__)

# OpenAI chat completion API 标准参数
_OPENAI_STANDARD_PARAMS = {
    "max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice",
    "response_format", "seed", "n", "frequency_penalty", "presence_penalty",
    "logprobs", "top_logprobs", "stream_options", "max_completion_tokens",
}


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

        try:
            # 分离标准参数和扩展参数（DashScope 等可能有非标准参数如 top_k）
            standard_params: dict[str, Any] = {}
            extra_body: dict[str, Any] = {}
            for k, v in kwargs.items():
                if k in _OPENAI_STANDARD_PARAMS:
                    standard_params[k] = v
                else:
                    extra_body[k] = v

            create_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": stream,
                "timeout": timeout,
                **standard_params,
            }
            # 流式请求自动启用 usage 统计
            if stream and "stream_options" not in create_kwargs:
                create_kwargs["stream_options"] = {"include_usage": True}
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            if self.provider.custom_headers:
                create_kwargs["extra_headers"] = self.provider.custom_headers

            response = await self._client.chat.completions.create(**create_kwargs)

            if stream:
                resp_ref = None  # 用于在闭包中引用 AdapterResponse

                async def stream_generator():
                    try:
                        chunk_count = 0
                        content_chars = 0
                        reasoning_buffer = []  # 缓存推理内容
                        reasoning_flushed = False
                        model_id = model_name  # 用于构造合成 chunk

                        async for chunk in response:
                            chunk_count += 1
                            # 捕获最终 chunk 中的 usage（stream_options.include_usage）
                            if hasattr(chunk, "usage") and chunk.usage is not None:
                                if resp_ref is not None:
                                    resp_ref.usage = {
                                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                                        "completion_tokens": chunk.usage.completion_tokens or 0,
                                    }

                            if hasattr(chunk, "choices") and chunk.choices:
                                choice = chunk.choices[0]
                                # delta 可能是 SDK 对象或 mock dict
                                if isinstance(choice, dict):
                                    delta = choice.get("delta", {})
                                    rc = delta.get("reasoning_content")
                                    ct = delta.get("content")
                                else:
                                    delta = getattr(choice, "delta", None)
                                    if delta is None:
                                        continue
                                    rc = getattr(delta, "reasoning_content", None)
                                    ct = getattr(delta, "content", None)

                                if rc:
                                    reasoning_buffer.append(rc)
                                if ct:
                                    # 首次出现 content 时，一次性发送缓存的推理内容
                                    if reasoning_buffer and not reasoning_flushed:
                                        reasoning_text = "".join(reasoning_buffer)
                                        content_chars += len(reasoning_text)
                                        yield _make_reasoning_chunk(chunk, model_id, reasoning_text)
                                        reasoning_flushed = True
                                    content_chars += len(ct)
                                    yield chunk
                                elif rc:
                                    # 只有 reasoning_content，暂不发送（继续缓冲）
                                    continue
                                else:
                                    # 既无 reasoning 也无 content（如 finish_reason chunk），直接透传
                                    yield chunk
                            else:
                                yield chunk
                        # 流结束：如果只有推理内容没有实际 content，也要刷新缓冲
                        if reasoning_buffer and not reasoning_flushed:
                            reasoning_text = "".join(reasoning_buffer)
                            content_chars += len(reasoning_text)
                            # 使用最后一个 chunk 作为模板（或构造最小 chunk）
                            try:
                                yield _make_reasoning_chunk(chunk, model_id, reasoning_text)
                            except Exception:
                                pass
                        elapsed = (time.monotonic() - start) * 1000
                        # Fallback: provider 未返回 usage 时，从流式内容估算 token 数
                        if resp_ref is not None and resp_ref.usage is None and content_chars > 0:
                            estimated = max(1, content_chars // 3)
                            logger.debug(
                                f"[{request_id}] usage_not_reported provider={self.name} "
                                f"estimated_output_tokens={estimated} from {content_chars} chars"
                            )
                            resp_ref.usage = {
                                "prompt_tokens": 0,
                                "completion_tokens": estimated,
                            }
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
                    status_code=200, success=True,
                    stream=stream_generator(),
                    adapter_name=self.name, model_name=model_name,
                    latency_ms=latency,
                )
                resp_ref = adapter_resp
                return adapter_resp
            else:
                # 合并 reasoning_content 到 content（GLM 推理模型将输出放在 reasoning_content）
                if hasattr(response, "choices"):
                    for choice in response.choices:
                        msg = choice.message
                        reasoning = getattr(msg, "reasoning_content", None)
                        if reasoning:
                            msg.content = reasoning + (msg.content or "")
                latency = (time.monotonic() - start) * 1000
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                    }

                logger.debug(
                    f"[{request_id}] api_response provider={self.name} "
                    f"status=200 latency={latency:.0f}ms usage={usage}"
                )

                return AdapterResponse(
                    status_code=200, success=True, body=response,
                    adapter_name=self.name, model_name=model_name,
                    latency_ms=latency, usage=usage,
                )

        except APITimeoutError as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning(f"[{request_id}] api_timeout provider={self.name} latency={latency:.0f}ms")
            return AdapterResponse(
                status_code=504, success=False, adapter_name=self.name,
                model_name=model_name, latency_ms=latency, error=str(e)
            )
        except APIStatusError as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning(
                f"[{request_id}] api_error provider={self.name} "
                f"status={e.status_code} error={e}"
            )
            return AdapterResponse(
                status_code=e.status_code, success=False, adapter_name=self.name,
                model_name=model_name, latency_ms=latency, error=str(e)
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"[{request_id}] api_exception provider={self.name} error={e}")
            return AdapterResponse(
                status_code=502, success=False, adapter_name=self.name,
                model_name=model_name, latency_ms=latency, error=str(e)
            )
