from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.message_converter import _to_dict

router = APIRouter()


@router.get("/openai/models")
@router.get("/v1/models")
async def list_models(request: Request):
    """列出所有已配置的虚拟模型（OpenAI 兼容格式）"""
    chain_router = request.app.state.chain_router
    model_names = chain_router.list_models()

    import time

    models_data = []
    for name in model_names:
        model_config = chain_router.get_model(name)
        models_data.append(
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "modelswitch",
                "permission": [],
                "root": name,
                "parent": None,
                "description": model_config.description if model_config else "",
            }
        )

    return JSONResponse(
        content={
            "object": "list",
            "data": models_data,
        }
    )


@router.post("/openai/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容的聊天补全接口（支持流式和非流式）"""
    chain_router = request.app.state.chain_router

    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # 模型权限检查
    api_key_config = getattr(request.state, "api_key_config", None)
    if api_key_config:
        allowed_models = api_key_config.get("allowed_models", [])
        if allowed_models and model not in allowed_models:
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": f"Model '{model}' not allowed for this API key",
                        "type": "forbidden",
                    }
                },
            )

    request_id = getattr(request.state, "request_id", "")

    # 提取额外参数（跳过空列表/空字典，如 tools: [] 会被上游拒绝）
    kwargs = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "user",
        "tools",
        "tool_choice",
        "response_format",
    ):
        if key in body:
            val = body[key]
            if val is None or (isinstance(val, (list, dict)) and not val):
                continue
            kwargs[key] = val

    if stream:
        return await _handle_stream(
            request, chain_router, model, messages, request_id, kwargs
        )
    else:
        return await _handle_non_stream(
            request, chain_router, model, messages, request_id, kwargs
        )


async def _record(
    app_state,
    request_id,
    model,
    result,
    api_key_alias,
    messages=None,
    stream_output=None,
):
    """记录用量和日志"""
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
    request, chain_router, model, messages, request_id, kwargs
):
    """处理非流式请求"""
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
                "error": {
                    "message": result.error,
                    "type": "upstream_error",
                    "code": "server_error",
                }
            },
            headers={"X-Request-ID": request_id},
        )

    resp_body = result.body
    if resp_body is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Empty response from upstream",
                    "type": "upstream_error",
                }
            },
        )

    if hasattr(resp_body, "model_dump"):
        response_data = resp_body.model_dump(exclude_none=True)
    elif hasattr(resp_body, "to_dict"):
        response_data = resp_body.to_dict()
    else:
        response_data = str(resp_body)

    return JSONResponse(
        content=response_data,
        headers={
            "X-Request-ID": request_id,
            "X-Adapter-Name": (result.adapter_name or "").encode("latin-1", errors="replace").decode("latin-1"),
        },
    )


async def _handle_stream(request, chain_router, model, messages, request_id, kwargs):
    """处理流式请求"""
    model_config = chain_router.get_model(model)
    if not model_config:
        error_msg = {
            "error": {
                "message": f"Model '{model}' not found",
                "type": "invalid_request_error",
            }
        }
        error_json = json.dumps(error_msg)

        async def error_gen():
            yield f"data: {error_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(error_gen(), media_type="text/event-stream")

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
            yield f"data: {json.dumps({'error': {'message': result.error, 'type': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(error_gen(), media_type="text/event-stream")

    adapter_name = result.adapter_name
    api_key_alias = getattr(request.state, "api_key_name", "")

    # 流式输出累积器
    collected_text = []
    collected_tool_calls = {}  # index -> {"name": str, "arguments": str}

    async def generate():
        try:
            if result.stream:
                async for chunk in result.stream:
                    if isinstance(chunk, dict) and chunk.get("_stream_error"):
                        yield f"data: {json.dumps(chunk.get('error', chunk), ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    chunk_data = _to_dict(chunk)
                    if not isinstance(chunk_data, dict):
                        chunk_data = str(chunk)

                    # 累积输出内容
                    if isinstance(chunk_data, dict):
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
                                        collected_tool_calls[idx]["arguments"] += func[
                                            "arguments"
                                        ]

                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"
            yield "data: [DONE]\n\n"
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
            "X-Adapter-Name": (adapter_name or "").encode("latin-1", errors="replace").decode("latin-1"),
        },
    )
