"""LiteLLM-based adapter for unified provider access.

Uses litellm.acompletion for standardized API calls across providers,
while preserving ModelSwitch's custom features:
- Chain routing with fallback
- Circuit breaker
- Request queuing
- Custom logging

This adapter replaces the direct OpenAI SDK approach with LiteLLM's unified API.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import litellm
from litellm import acompletion

from app.adapters.base import AdapterResponse, BaseAdapter, create_error_response
from app.models.config_models import ProviderConfig
from app.core.request_queue import get_queue_manager
from app.utils.logging import get_adapter_logger, get_protocol_logger

logger = logging.getLogger(__name__)

# Configure LiteLLM settings
litellm.drop_params = True  # Drop unsupported params instead of raising error
litellm.set_verbose = False  # Disable LiteLLM's own verbose logging

# LiteLLM model prefix mapping for provider types
_PROVIDER_PREFIX_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "azure": "azure",
    "dashscope": "openai",  # DashScope uses OpenAI-compatible API
    "glm": "openai",  # GLM uses OpenAI-compatible API
    "deepseek": "deepseek",
    "vertex_ai": "vertex_ai",
    "bedrock": "bedrock",
}


class LiteLLMAdapter(BaseAdapter):
    """Adapter using litellm.acompletion for unified provider access."""

    def __init__(self, provider_config: ProviderConfig):
        super().__init__(provider_config)
        self._queue_manager = get_queue_manager()
        self._use_queue = provider_config.max_concurrent > 0

        # If configured, register provider to queue manager
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
        """Execute chat completion using litellm.acompletion."""
        start = time.monotonic()

        logger.debug(
            f"[{request_id}] litellm_call provider={self.name} "
            f"model={model_name} stream={stream} timeout={timeout}"
        )

        # Use queue if configured
        if self._use_queue:
            logger.debug(f"[{request_id}] Using request queue: {self.name}")
            return await self._queue_manager.execute(
                self.name,
                self._do_chat_completion,
                model_name, messages, stream, timeout, request_id,
                **kwargs
            )

        # Execute directly
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
        """Execute chat completion via litellm.acompletion."""
        start = time.monotonic()
        adapter_logger = get_adapter_logger(self.name, request_id)
        protocol_logger = get_protocol_logger(request_id, "litellm")

        # Log request details
        adapter_logger.log_request(
            model=model_name,
            stream=stream,
            timeout=timeout,
            messages_count=len(messages),
            params=kwargs,
        )

        try:
            # Build LiteLLM model name
            litellm_model = self._build_model_name(model_name)

            # Build kwargs for litellm.acompletion
            create_kwargs = self._build_litellm_kwargs(
                litellm_model, messages, stream, timeout, **kwargs
            )

            # Log upstream request
            protocol_logger.log_upstream_request(
                adapter=self.name,
                model=litellm_model,
                stream=stream,
                create_kwargs_keys=list(create_kwargs.keys()),
            )

            adapter_logger.debug(
                f"api_call_start",
                model=litellm_model,
                stream=stream,
                kwargs_keys=list(create_kwargs.keys()),
            )

            # Execute via litellm
            response = await acompletion(**create_kwargs)

            if stream:
                return await self._create_stream_response(
                    response, model_name, start, request_id, adapter_logger
                )
            else:
                return self._create_non_stream_response(
                    response, model_name, start, request_id, adapter_logger
                )

        except litellm.Timeout as e:
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

        except litellm.APIConnectionError as e:
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

        except litellm.BadRequestError as e:
            latency = (time.monotonic() - start) * 1000
            adapter_logger.log_error(
                model=model_name,
                error_type="bad_request",
                error_message=str(e),
                latency_ms=latency,
                status_code=400,
            )
            return create_error_response(
                self.name, model_name, latency, 400, str(e), request_id
            )

        except litellm.APIError as e:
            latency = (time.monotonic() - start) * 1000
            # Extract status code if available
            status_code = getattr(e, "status_code", 502) or 502
            adapter_logger.log_error(
                model=model_name,
                error_type="api_error",
                error_message=str(e),
                latency_ms=latency,
                status_code=status_code,
            )
            return create_error_response(
                self.name, model_name, latency, status_code, str(e), request_id
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
            logger.exception(f"[{request_id}] Unexpected error in LiteLLM adapter")
            return create_error_response(
                self.name, model_name, latency, 502, str(e), request_id
            )

    def _build_model_name(self, model_name: str) -> str:
        """Build complete LiteLLM model name with prefix.

        LiteLLM uses model prefixes to route requests:
        - openai/gpt-4 -> OpenAI API
        - anthropic/claude-3 -> Anthropic API
        - openai/model with api_base -> Custom OpenAI-compatible endpoint

        Args:
            model_name: Model name without prefix

        Returns:
            LiteLLM model name with appropriate prefix
        """
        # If model already has prefix, use it directly
        if "/" in model_name:
            return model_name

        # Get provider prefix
        prefix = _PROVIDER_PREFIX_MAP.get(self.provider.provider, "openai")

        return f"{prefix}/{model_name}"

    def _build_litellm_kwargs(
        self,
        model: str,
        messages: list,
        stream: bool,
        timeout: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for litellm.acompletion."""
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "timeout": timeout,
        }

        # Add provider-specific base_url for OpenAI-compatible providers
        # LiteLLM will use this instead of the default endpoint
        if self.provider.base_url:
            create_kwargs["api_base"] = self.provider.base_url

        # Add API key
        if self.provider.api_key:
            create_kwargs["api_key"] = self.provider.api_key

        # Add supported kwargs (LiteLLM standard params)
        supported_params = {
            "max_tokens", "temperature", "top_p", "stop", "seed", "n",
            "tools", "tool_choice", "response_format", "stream_options",
            "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs",
            "logit_bias", "metadata", "user",
            # Reasoning params (for supported models)
            "reasoning_effort", "max_completion_tokens",
        }

        for k, v in kwargs.items():
            if k in supported_params and v is not None:
                create_kwargs[k] = v

        # Add custom headers if configured
        if self.provider.custom_headers:
            create_kwargs["extra_headers"] = self.provider.custom_headers

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
        resp_ref = None

        async def stream_generator():
            try:
                chunk_count = 0
                content_chars = 0

                async for chunk in response:
                    chunk_count += 1

                    # Capture usage from final chunk (LiteLLM includes this)
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        if resp_ref is not None:
                            usage_dict = chunk.usage
                            if hasattr(usage_dict, "model_dump"):
                                usage_dict = usage_dict.model_dump()
                            resp_ref.usage = {
                                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                                "completion_tokens": usage_dict.get("completion_tokens", 0),
                            }

                    # Count content chars for token estimation fallback
                    if hasattr(chunk, "choices") and chunk.choices:
                        choice = chunk.choices[0]
                        if isinstance(choice, dict):
                            delta = choice.get("delta", {})
                            ct = delta.get("content", "")
                        else:
                            delta = getattr(choice, "delta", None)
                            ct = getattr(delta, "content", "") if delta else ""
                        if ct:
                            content_chars += len(ct)

                    yield chunk

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

                adapter_logger.log_stream_complete(
                    model=model_name,
                    total_chunks=chunk_count,
                    latency_ms=elapsed,
                    usage=resp_ref.usage if resp_ref else None,
                )

            except asyncio.CancelledError:
                elapsed = (time.monotonic() - start) * 1000
                adapter_logger.debug(
                    f"stream_cancelled",
                    model=model_name,
                    chunks=chunk_count,
                    latency_ms=f"{elapsed:.0f}",
                )
                logger.debug(f"[{request_id}] Stream cancelled by client")
                raise

            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                adapter_logger.log_error(
                    model=model_name,
                    error_type="stream",
                    error_message=f"{type(e).__name__}: {str(e)}",
                    latency_ms=elapsed,
                    status_code=502,
                )
                logger.exception(f"[{request_id}] Stream error in LiteLLM adapter")
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
        latency = (time.monotonic() - start) * 1000

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage_dict = response.usage
            if hasattr(usage_dict, "model_dump"):
                usage_dict = usage_dict.model_dump()
            usage = {
                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                "completion_tokens": usage_dict.get("completion_tokens", 0),
            }

        adapter_logger.log_response_start(
            model=model_name,
            latency_ms=latency,
            status_code=200,
            usage=usage,
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


# Factory function for backward compatibility
def create_adapter(provider_config: ProviderConfig) -> BaseAdapter:
    """Create adapter based on provider configuration.

    Uses LiteLLMAdapter for all providers except Anthropic (which has native SDK).
    """
    # Anthropic has its own native SDK adapter for full streaming control
    if provider_config.provider == "anthropic":
        from app.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(provider_config)

    # All other providers use LiteLLM for unified access
    return LiteLLMAdapter(provider_config)


# Backward compatibility aliases
LiteLLMAdapter = LiteLLMAdapter  # Already defined above
OpenAIAdapter = LiteLLMAdapter  # Alias for backward compatibility