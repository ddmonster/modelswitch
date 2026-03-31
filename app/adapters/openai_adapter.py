from __future__ import annotations

import logging
import time
from typing import Any

from openai import AsyncOpenAI, APITimeoutError, APIStatusError

from app.adapters.base import AdapterResponse, BaseAdapter
from app.models.config_models import ProviderConfig

logger = logging.getLogger(__name__)

# OpenAI chat completion API 标准参数
_OPENAI_STANDARD_PARAMS = {
    "max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice",
    "response_format", "seed", "n", "frequency_penalty", "presence_penalty",
    "logprobs", "top_logprobs", "stream_options", "max_completion_tokens",
}


class OpenAIAdapter(BaseAdapter):
    """基于 openai SDK 的适配器，兼容 DashScope / GLM / OpenAI 等 provider"""

    def __init__(self, provider_config: ProviderConfig):
        super().__init__(provider_config)
        self._client = AsyncOpenAI(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
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
                        async for chunk in response:
                            chunk_count += 1
                            # 捕获最终 chunk 中的 usage（stream_options.include_usage）
                            if hasattr(chunk, "usage") and chunk.usage is not None:
                                if resp_ref is not None:
                                    resp_ref.usage = {
                                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                                        "completion_tokens": chunk.usage.completion_tokens or 0,
                                    }
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
                    status_code=200, success=True,
                    stream=stream_generator(),
                    adapter_name=self.name, model_name=model_name,
                    latency_ms=latency,
                )
                resp_ref = adapter_resp
                return adapter_resp
            else:
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
