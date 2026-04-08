from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.message_converter import (
    _to_dict,
    anthropic_to_openai_messages,
    convert_openai_to_anthropic_response,
    openai_stream_to_anthropic,
)

router = APIRouter()


@router.post("/anthropic/messages")
@router.post("/v1/messages")
async def messages(request: Request):
    """Anthropic 兼容的 Messages API（支持流式和非流式）"""
    chain_router = request.app.state.chain_router
    request_id = getattr(request.state, "request_id", "")

    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)

    # 模型权限检查
    api_key_config = getattr(request.state, "api_key_config", None)
    if api_key_config:
        allowed_models = api_key_config.get("allowed_models", [])
        if allowed_models and model not in allowed_models:
            return JSONResponse(
                status_code=403,
                content={
                    "type": "error",
                    "error": {
                        "type": "permission_error",
                        "message": f"Model '{model}' not allowed",
                    },
                },
            )

    # C2 fix: 判断客户端是否请求了 thinking
    thinking = body.get("thinking")
    thinking_enabled = isinstance(thinking, dict) and thinking.get("type") == "enabled"

    # 将 Anthropic 请求转换为 OpenAI 格式
    openai_body = anthropic_to_openai_messages(body)
    messages_list = openai_body.pop("messages", [])

    kwargs = {
        k: v
        for k, v in openai_body.items()
        if v is not None
        and k not in ("model", "messages", "stream")
        and not (isinstance(v, (list, dict)) and not v)
    }

    if stream:
        return await _handle_stream(
            request,
            chain_router,
            model,
            messages_list,
            request_id,
            kwargs,
            thinking_enabled=thinking_enabled,
        )
    else:
        return await _handle_non_stream(
            request,
            chain_router,
            model,
            messages_list,
            request_id,
            kwargs,
            thinking_enabled=thinking_enabled,
        )


async def _record(
    app_state,
    request_id,
    model,
    result,
    api_key_alias="",
    messages=None,
    stream_output=None,
):
    """记录用量统计和日志"""
    from app.utils.tracking import track_request

    await track_request(
        app_state,
        request_id,
        model,
        result,
        api_key_alias,
        messages=messages,
        stream_output=stream_output,
    )


async def _handle_non_stream(
    request,
    chain_router,
    model,
    messages,
    request_id,
    kwargs,
    thinking_enabled=False,
):
    """处理非流式 Anthropic 请求"""
    result = await chain_router.execute_chat(
        model=model,
        messages=messages,
        stream=False,
        request_id=request_id,
        **kwargs,
    )

    api_key_alias = getattr(request.state, "api_key_name", "")
    await _record(
        request.app.state, request_id, model, result, api_key_alias, messages=messages
    )

    if not result.success:
        return JSONResponse(
            status_code=result.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": result.error},
            },
        )

    resp = result.body
    if hasattr(resp, "model_dump"):
        resp_data = resp.model_dump(exclude_none=True)
    elif hasattr(resp, "to_dict"):
        resp_data = resp.to_dict()
    else:
        resp_data = resp

    # C2/C3 fix: 传入 thinking_enabled，控制是否生成 thinking 块
    anthropic_response = convert_openai_to_anthropic_response(
        resp_data, model, thinking_enabled=thinking_enabled
    )

    return JSONResponse(
        content=anthropic_response,
        headers={
            "X-Request-ID": request_id,
            "X-Adapter-Name": result.adapter_name,
        },
    )


async def _handle_stream(
    request,
    chain_router,
    model,
    messages,
    request_id,
    kwargs,
    thinking_enabled=False,
):
    """处理流式 Anthropic 请求"""
    result = await chain_router.execute_chat(
        model=model,
        messages=messages,
        stream=True,
        request_id=request_id,
        **kwargs,
    )

    if not result.success:
        api_key_alias = getattr(request.state, "api_key_name", "")
        await _record(
            request.app.state,
            request_id,
            model,
            result,
            api_key_alias,
            messages=messages,
        )

        async def error_gen():
            error_data = {
                "type": "error",
                "error": {"type": "api_error", "message": result.error},
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        return StreamingResponse(error_gen(), media_type="text/event-stream")

    api_key_alias = getattr(request.state, "api_key_name", "")

    # 流式输出累积器
    collected_text = []
    collected_tool_calls = {}  # index -> {"name": str, "arguments": str}

    async def capturing_stream(raw_stream):
        """包装原始流，在传给转换器之前累积输出内容"""
        async for chunk in raw_stream:
            if isinstance(chunk, dict) and chunk.get("_stream_error"):
                yield chunk
                return
            chunk_data = _to_dict(chunk)
            if not isinstance(chunk_data, dict):
                yield chunk
                continue

            choices = chunk_data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    collected_text.append(delta["content"])
                elif delta.get("reasoning_content"):
                    collected_text.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    func = tc.get("function", {})
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                        }
                    else:
                        if func.get("name"):
                            collected_tool_calls[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            collected_tool_calls[idx]["arguments"] += func["arguments"]
            yield chunk

    async def generate():
        stream_error = None

        async def safe_stream(raw_stream):
            """拦截 _stream_error，正常 chunk 透传"""
            nonlocal stream_error
            async for chunk in raw_stream:
                if isinstance(chunk, dict) and chunk.get("_stream_error"):
                    stream_error = chunk.get("error", chunk)
                    return
                yield chunk

        try:
            # C2 fix: 传入 thinking_enabled，控制是否生成 thinking 块
            async for anth_chunk in openai_stream_to_anthropic(
                capturing_stream(safe_stream(result.stream)),
                model,
                request_id,
                thinking_enabled=thinking_enabled,
            ):
                yield anth_chunk

            # 如果所有 adapter 失败，发送错误事件
            if stream_error:
                error_data = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": stream_error.get("message", "All adapters failed"),
                    },
                }
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        except Exception as e:
            error_data = {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        finally:
            info = getattr(result, "_stream_adapter_info", None)
            if info:
                result.adapter_name = info.get("name", "")
                result.latency_ms = info.get("latency", 0.0)
                if info.get("usage"):
                    result.usage = info["usage"]

            # 构建流式输出摘要
            stream_output = None
            parts = []
            if collected_text:
                parts.append({"type": "text", "text": "".join(collected_text)})
            for idx in sorted(collected_tool_calls):
                tc = collected_tool_calls[idx]
                parts.append(
                    {
                        "type": "tool_use",
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                )
            if parts:
                stream_output = parts

            await _record(
                request.app.state,
                request_id,
                model,
                result,
                api_key_alias,
                messages=messages,
                stream_output=stream_output,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
            "X-Adapter-Name": result.adapter_name or "",
        },
    )
