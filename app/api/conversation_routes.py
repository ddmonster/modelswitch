"""对话记录查看 API —— 读取 conversations.jsonl 提供筛选/分页/详情。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["conversations"])

# ---------- mtime 缓存 ----------
_cache: dict = {"mtime": 0.0, "records": [], "api_keys": [], "models": []}


def _get_log_path(app_state) -> Path:
    config = getattr(app_state, "config", None)
    log_dir = "logs"
    if config and hasattr(config, "gateway"):
        log_dir = getattr(config.gateway, "log_dir", "logs")
    return Path(log_dir) / "conversations.jsonl"


def _load_records(log_path: Path) -> list[dict]:
    """解析 JSONL 文件，返回记录列表（保留原始行号）。"""
    if not log_path.exists():
        return []

    mtime = os.path.getmtime(log_path)
    if mtime == _cache["mtime"] and _cache["records"]:
        return _cache["records"]

    records = []
    api_keys_set: set[str] = set()
    models_set: set[str] = set()

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_line"] = line_no
                records.append(rec)
                if rec.get("api_key"):
                    api_keys_set.add(rec["api_key"])
                if rec.get("model"):
                    models_set.add(rec["model"])
            except (json.JSONDecodeError, Exception):
                continue

    _cache["mtime"] = mtime
    _cache["records"] = records
    _cache["api_keys"] = sorted(api_keys_set)
    _cache["models"] = sorted(models_set)
    return records


def _output_preview(record: dict) -> tuple[str, bool]:
    """从 output 中提取预览文本和是否含有 tool_use。"""
    output = record.get("output")
    has_tool_use = False
    preview = ""

    if not output or not isinstance(output, list):
        return preview, has_tool_use

    for block in output:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "")
            if not preview:
                preview = text[:100]
        elif btype == "tool_use":
            has_tool_use = True
            if not preview:
                preview = f"[tool_use: {block.get('name', '')}]"

    return preview, has_tool_use


@router.get("/api/conversations")
async def list_conversations(
    request: Request,
    api_key: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    success: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """列出对话记录（仅元数据，不含 messages/output）。"""
    log_path = _get_log_path(request.app.state)
    records = _load_records(log_path)

    # 筛选
    filtered = records
    if api_key:
        filtered = [r for r in filtered if r.get("api_key") == api_key]
    if model:
        filtered = [r for r in filtered if r.get("model") == model]
    if success is not None and success != "":
        s_bool = success.lower() == "true"
        filtered = [r for r in filtered if r.get("success") == s_bool]

    # 逆序（最新在前）
    filtered = list(reversed(filtered))
    total = len(filtered)

    # 分页
    page = filtered[offset: offset + limit]

    # 构建列表项（不含 messages/output）
    items = []
    for rec in page:
        preview, has_tool_use = _output_preview(rec)
        items.append({
            "line": rec.get("_line", 0),
            "timestamp": rec.get("timestamp", ""),
            "request_id": rec.get("request_id", ""),
            "model": rec.get("model", ""),
            "adapter": rec.get("adapter", ""),
            "api_key": rec.get("api_key", ""),
            "success": rec.get("success", True),
            "latency_ms": rec.get("latency_ms", 0),
            "tokens_in": rec.get("tokens_in", 0),
            "tokens_out": rec.get("tokens_out", 0),
            "output_preview": preview,
            "has_tool_use": has_tool_use,
        })

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
        "api_keys": _cache.get("api_keys", []),
        "models": _cache.get("models", []),
    }


@router.get("/api/conversations/{line}")
async def get_conversation_detail(request: Request, line: int):
    """获取某行对话的完整记录（含 messages 和 output）。"""
    log_path = _get_log_path(request.app.state)
    records = _load_records(log_path)

    for rec in records:
        if rec.get("_line") == line:
            # 返回完整记录（去掉内部字段）
            result = {k: v for k, v in rec.items() if not k.startswith("_")}
            result["line"] = line
            return result

    return JSONResponse(status_code=404, content={"error": "Record not found"})
