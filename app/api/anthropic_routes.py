from __future__ import annotations

import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.message_converter import anthropic_to_openai_messages, openai_stream_to_anthropic

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
                content={"type": "error", "error": {"type": "permission_error", "message": f"Model '{model}' not allowed"}},
            )

    # 将 Anthropic 请求转换为 OpenAI 格式
    openai_body = anthropic_to_openai_messages(body)
    messages_list = openai_body.pop("messages", [])

    kwargs = {k: v for k, v in openai_body.items()
              if v is not None and k not in ("model", "messages", "stream")
              and not (isinstance(v, (list, dict)) and not v)}

    if stream:
        return await _handle_stream(request, chain_router, model, messages_list, request_id, kwargs)
    else:
        return await _handle_non_stream(request, chain_router, model, messages_list, request_id, kwargs)


async def _record(app_state, request_id, model, result, api_key_alias="",
                  messages=None, stream_output=None):
    """记录用量统计和日志"""
    from app.utils.tracking import track_request
    await track_request(app_state, request_id, model, result, api_key_alias,
                        messages=messages, stream_output=stream_output)


async def _handle_non_stream(request, chain_router, model, messages, request_id, kwargs):
    """处理非流式 Anthropic 请求"""
    result = await chain_router.execute_chat(
        model=model,
        messages=messages,
        stream=False,
        request_id=request_id,
        **kwargs,
    )

    api_key_alias = getattr(request.state, "api_key_name", "")
    await _record(request.app.state, request_id, model, result, api_key_alias,
                  messages=messages)

    if not result.success:
        return JSONResponse(
            status_code=result.status_code,
            content={"type": "error", "error": {"type": "api_error", "message": result.error}},
        )

    resp = result.body
    if hasattr(resp, "model_dump"):
        resp_data = resp.model_dump(exclude_none=True)
    elif hasattr(resp, "to_dict"):
        resp_data = resp.to_dict()
    else:
        resp_data = resp

    anthropic_response = _convert_openai_to_anthropic_response(resp_data, model)

    return JSONResponse(
        content=anthropic_response,
        headers={
            "X-Request-ID": request_id,
            "X-Adapter-Name": result.adapter_name,
        },
    )


def _convert_openai_to_anthropic_response(resp_data: dict, model: str) -> dict:
    """将 OpenAI ChatCompletion 响应转换为 Anthropic Messages 响应"""
    import uuid

    choices = resp_data.get("choices", [])
    content = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        msg_content = message.get("content", "")

        if msg_content:
            content.append({"type": "text", "text": msg_content})

        # 处理 tool_calls
        for tc in message.get("tool_calls", []):
            tc_func = tc.get("function", {})
            try:
                tc_input = json.loads(tc_func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tc_input = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": tc_func.get("name", ""),
                "input": tc_input,
            })

        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }.get(finish_reason, "end_turn")

    usage = resp_data.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def _handle_stream(request, chain_router, model, messages, request_id, kwargs):
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
        await _record(request.app.state, request_id, model, result, api_key_alias,
                      messages=messages)

        async def error_gen():
            error_data = {"type": "error", "error": {"type": "api_error", "message": result.error}}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    api_key_alias = getattr(request.state, "api_key_name", "")

    # 流式输出累积器
    collected_text = []
    collected_tool_calls = {}  # index -> {"name": str, "arguments": str}

    async def capturing_stream(raw_stream):
        """包装原始流，在传给转换器之前累积输出内容"""
        async for chunk in raw_stream:
            if hasattr(chunk, "model_dump"):
                chunk_data = chunk.model_dump(exclude_none=True)
            elif isinstance(chunk, dict):
                chunk_data = chunk
            else:
                yield chunk
                continue

            choices = chunk_data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    collected_text.append(delta["content"])
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
        try:
            async for chunk in openai_stream_to_anthropic(
                capturing_stream(result.stream), model, request_id
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
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
                parts.append({"type": "tool_use", "name": tc["name"], "arguments": tc["arguments"]})
            if parts:
                stream_output = parts

            await _record(request.app.state, request_id, model, result, api_key_alias,
                          messages=messages, stream_output=stream_output)

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
