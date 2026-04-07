from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.adapters.base import AdapterResponse, BaseAdapter
from app.adapters.litellm_adapter import create_adapter
from app.core.circuit_breaker import CircuitBreaker
from app.models.config_models import GatewayConfig, ModelAdapterRef, ModelConfig, ProviderConfig

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}


class ChainRouter:
    """核心链路路由器：adapter 模式直连 / chain 模式 fallback"""

    def __init__(self, config: GatewayConfig):
        self._config = config
        self._providers: Dict[str, ProviderConfig] = {}
        self._models: Dict[str, ModelConfig] = {}
        self._adapters: Dict[str, BaseAdapter] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._load_config(config)

    def _load_config(self, config: GatewayConfig) -> None:
        """加载配置，建立 provider 和 adapter 映射"""
        self._config = config
        self._providers = {p.name: p for p in config.providers}

        for provider_config in config.providers:
            if provider_config.name not in self._adapters:
                self._adapters[provider_config.name] = create_adapter(provider_config)
            else:
                # 更新已有 adapter 的配置
                self._adapters[provider_config.name] = create_adapter(provider_config)
            # 重置熔断器
            if provider_config.name in self._circuit_breakers:
                self._circuit_breakers[provider_config.name].reset()
            else:
                self._circuit_breakers[provider_config.name] = CircuitBreaker()

        self._models = config.models

    def reload_config(self, config: GatewayConfig) -> None:
        """热重载配置"""
        logger.info("ChainRouter 热重载配置")
        self._load_config(config)

    def _resolve_model(self, model_name: str) -> Optional[ModelConfig]:
        """查找模型配置，支持大小写不敏感匹配"""
        config = self._models.get(model_name)
        if config:
            return config
        lower = model_name.lower()
        for k, v in self._models.items():
            if k.lower() == lower:
                return v
        return None

    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        return self._resolve_model(model_name)

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def get_providers(self) -> Dict[str, ProviderConfig]:
        return self._providers

    def get_adapters(self) -> Dict[str, BaseAdapter]:
        return self._adapters

    async def execute_chat(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        request_id: str = "",
        **kwargs: Any,
    ) -> AdapterResponse:
        """
        执行路由：根据 model 配置选择 adapter/chain 模式。
        """
        model_config = self._resolve_model(model)
        if not model_config:
            return AdapterResponse(
                status_code=404, success=False,
                error=f"Model '{model}' not found"
            )

        if not model_config.adapters:
            return AdapterResponse(
                status_code=503, success=False,
                error=f"No adapters configured for model '{model}'"
            )

        logger.debug(
            f"[{request_id}] chain_start model={model} "
            f"mode={model_config.mode} "
            f"adapters=[{', '.join(f'{r.adapter}(p{r.priority})' for r in model_config.adapters)}]"
        )

        if model_config.mode == "adapter":
            return await self._execute_adapter_mode(model_config, model, messages, stream, request_id, **kwargs)
        else:
            if stream:
                # 用可变容器在 generator 和外层之间传递 adapter 信息
                adapter_info = {"name": "", "latency": 0.0}

                async def wrapped_stream():
                    async for chunk in self._execute_chat_stream(
                        model_config, model, messages, request_id,
                        _adapter_info=adapter_info, **kwargs,
                    ):
                        yield chunk

                return AdapterResponse(
                    status_code=200, success=True,
                    stream=wrapped_stream(),
                    adapter_name="", model_name=model,
                    _stream_adapter_info=adapter_info,
                )
            else:
                return await self._execute_chain_mode(model_config, model, messages, request_id, **kwargs)

    async def _execute_adapter_mode(
        self, model_config: ModelConfig, model: str, messages: list,
        stream: bool, request_id: str, **kwargs
    ) -> AdapterResponse:
        """adapter 模式：直接调用指定 provider，失败直接返回"""
        ref = model_config.adapters[0]
        provider = self._providers.get(ref.adapter)

        if not provider:
            return AdapterResponse(
                status_code=503, success=False,
                error=f"Provider '{ref.adapter}' not found"
            )
        if not provider.enabled:
            return AdapterResponse(
                status_code=503, success=False,
                error=f"Provider '{ref.adapter}' is disabled"
            )

        cb = self._circuit_breakers.get(ref.adapter)
        if cb and not cb.can_execute():
            return AdapterResponse(
                status_code=503, success=False,
                error=f"Provider '{ref.adapter}' circuit breaker is open"
            )

        adapter = self._adapters[ref.adapter]
        result = await adapter.chat_completion(
            model_name=ref.model_name,
            messages=messages,
            stream=stream,
            timeout=ref.timeout,
            request_id=request_id,
            **kwargs,
        )

        if result.success and cb:
            cb.record_success()
        elif not result.success and cb:
            cb.record_failure()

        return result

    async def _execute_chain_mode(
        self, model_config: ModelConfig, model: str, messages: list,
        request_id: str, **kwargs
    ) -> AdapterResponse:
        """chain 模式：按 priority 排序依次尝试，失败 fallback"""
        sorted_refs = sorted(model_config.adapters, key=lambda r: r.priority)
        last_error = ""

        for ref in sorted_refs:
            provider = self._providers.get(ref.adapter)
            if not provider or not provider.enabled:
                continue

            cb = self._circuit_breakers.get(ref.adapter)
            if cb and not cb.can_execute():
                logger.warning(
                    f"[{request_id}] circuit_breaker OPEN provider={ref.adapter}, skipping"
                )
                continue

            adapter = self._adapters[ref.adapter]
            logger.debug(
                f"[{request_id}] trying adapter={ref.adapter} "
                f"model={ref.model_name} timeout={ref.timeout}"
            )

            for attempt in range(2):  # 单 adapter 最多重试 1 次
                result = await adapter.chat_completion(
                    model_name=ref.model_name,
                    messages=messages,
                    stream=False,
                    timeout=ref.timeout,
                    request_id=request_id,
                    **kwargs,
                )

                if result.success:
                    logger.debug(
                        f"[{request_id}] adapter={ref.adapter} SUCCESS "
                        f"status={result.status_code} latency={result.latency_ms:.0f}ms"
                    )
                    if cb:
                        cb.record_success()
                    result.adapter_name = ref.adapter
                    return result

                # 记录失败
                if cb:
                    cb.record_failure()

                if result.status_code in NON_RETRYABLE_STATUS_CODES:
                    last_error = result.error or f"HTTP {result.status_code}"
                    logger.debug(
                        f"[{request_id}] adapter={ref.adapter} FAIL "
                        f"status={result.status_code} (non-retryable), fallback"
                    )
                    break
                elif attempt == 0:
                    logger.debug(
                        f"[{request_id}] adapter={ref.adapter} FAIL "
                        f"status={result.status_code}, retrying..."
                    )
                    await asyncio.sleep(0.5)
                else:
                    last_error = result.error or f"HTTP {result.status_code}"
                    logger.debug(
                        f"[{request_id}] adapter={ref.adapter} FAIL "
                        f"status={result.status_code} after retry, fallback"
                    )

        logger.error(f"[{request_id}] chain_failed model={model} all_adapters_exhausted")
        return AdapterResponse(
            status_code=502, success=False,
            error=f"All adapters failed. Last error: {last_error}"
        )

    async def _execute_chat_stream(
        self, model_config: ModelConfig, model: str, messages: list,
        request_id: str, _adapter_info: Optional[dict] = None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        流式链路执行：探测式首块检查。
        对每个 adapter，读取首块验证，有效则锁定透传，失败则 fallback。
        一旦开始透传数据，后续错误直接传给客户端。
        """
        sorted_refs = sorted(model_config.adapters, key=lambda r: r.priority)
        last_error = ""

        logger.debug(f"[{request_id}] stream_start model={model}")

        for ref in sorted_refs:
            provider = self._providers.get(ref.adapter)
            if not provider or not provider.enabled:
                continue

            cb = self._circuit_breakers.get(ref.adapter)
            if cb and not cb.can_execute():
                continue

            adapter = self._adapters[ref.adapter]
            logger.debug(
                f"[{request_id}] trying adapter={ref.adapter} stream=true"
            )

            has_sent = False
            try:
                result = await adapter.chat_completion(
                    model_name=ref.model_name,
                    messages=messages,
                    stream=True,
                    timeout=ref.timeout,
                    request_id=request_id,
                    **kwargs,
                )

                if not result.success or result.stream is None:
                    last_error = result.error or f"HTTP {result.status_code}"
                    if cb:
                        cb.record_failure()
                    continue

                # adapter 连接成功，立即记录（即使客户端随后断开也能追踪）
                if _adapter_info is not None:
                    _adapter_info["name"] = ref.adapter
                    _adapter_info["latency"] = result.latency_ms

                # 探测首块
                first_chunk = None
                chunk_count = 0

                async for chunk in result.stream:
                    chunk_count += 1
                    if first_chunk is None:
                        first_chunk = chunk
                        logger.debug(
                            f"[{request_id}] adapter={ref.adapter} "
                            f"first_chunk_received latency={result.latency_ms:.0f}ms, stream_locked"
                        )
                    yield chunk
                    has_sent = True

                if has_sent and cb:
                    cb.record_success()
                elif not has_sent and cb:
                    cb.record_failure()
                    continue

                if has_sent:
                    # 流结束后回传 usage（由 adapter stream generator 捕获）
                    if _adapter_info is not None and result.usage:
                        _adapter_info["usage"] = result.usage
                    return

            except Exception as e:
                last_error = str(e)
                if cb:
                    cb.record_failure()
                logger.error(
                    f"[{request_id}] adapter={ref.adapter} stream_error: {e}"
                )
                # 已经向客户端发送了数据，不能再 fallback 到下一个 adapter
                if has_sent:
                    raise
                continue

        # 所有适配器都失败了，yield 错误 chunk
        logger.error(f"[{request_id}] stream_chain_failed model={model}")
        yield {"_stream_error": True, "error": {"message": f"All adapters failed: {last_error}", "type": "upstream_error"}}
