"""Usage tracking and logging helpers for route handlers."""
from __future__ import annotations


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


async def track_request(app_state, request_id, model, result, api_key_alias=""):
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
