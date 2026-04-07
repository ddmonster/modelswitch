"""对话记录查看 API —— 内存高效版本，只缓存轻量元数据。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["conversations"])

# ---------- 轻量缓存：只存 api_keys 和 models ----------
_cache: dict = {
    "mtimes": {},  # 文件修改时间
    "api_keys": [],  # 所有出现过的 api_key
    "models": [],  # 所有出现过的 model
}


def _get_log_dir(app_state) -> Path:
    """获取日志目录路径。"""
    config = getattr(app_state, "config", None)
    log_dir = "logs"
    if config and hasattr(config, "gateway"):
        log_dir = getattr(config.gateway, "log_dir", "logs")
    return Path(log_dir)


def _get_all_log_paths(log_dir: Path) -> list[Path]:
    """
    获取所有对话日志文件路径，按时间排序（旧→新）。
    返回顺序：.N, .N-1, ..., .2, .1, 当前文件（从旧到新）
    """
    base_path = log_dir / "conversations.jsonl"
    paths = []

    if not base_path.exists():
        return paths

    # 查找所有备份文件
    backup_paths = []
    for i in range(1, 100):  # 支持最多 99 个备份文件
        backup_path = log_dir / f"conversations.jsonl.{i}"
        if backup_path.exists():
            backup_paths.append(backup_path)
        else:
            break

    # 备份文件按数字倒序排列（数字越大越旧）
    backup_paths.sort(key=lambda p: int(p.suffix[1:]), reverse=True)

    # 先添加备份文件（旧），再添加当前文件（新）
    paths.extend(backup_paths)
    paths.append(base_path)

    return paths


def _get_mtimes(log_paths: list[Path]) -> dict:
    """获取所有文件的修改时间。"""
    mtimes = {}
    for path in log_paths:
        if path.exists():
            mtimes[str(path)] = os.path.getmtime(path)
    return mtimes


def _iter_records(log_dir: Path) -> Generator[dict, None, None]:
    """
    流式读取所有对话记录（生成器）。
    按时间顺序返回（旧→新）。
    每条记录带有全局行号 _line。
    """
    log_paths = _get_all_log_paths(log_dir)
    global_line = 0

    for log_path in log_paths:
        if not log_path.exists():
            continue
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["_line"] = global_line
                    yield rec
                    global_line += 1
                except (json.JSONDecodeError, Exception):
                    continue


def _iter_light_metadata(log_dir: Path) -> Generator[dict, None, None]:
    """
    流式读取所有对话记录的轻量元数据（生成器）。
    不包含 messages 和 output 字段，节省内存。
    按时间顺序返回（旧→新）。
    """
    log_paths = _get_all_log_paths(log_dir)
    global_line = 0

    for log_path in log_paths:
        if not log_path.exists():
            continue
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # 只保留列表需要的轻量字段
                    yield {
                        "_line": global_line,
                        "timestamp": rec.get("timestamp", ""),
                        "request_id": rec.get("request_id", ""),
                        "model": rec.get("model", ""),
                        "adapter": rec.get("adapter", ""),
                        "api_key": rec.get("api_key", ""),
                        "success": rec.get("success", True),
                        "latency_ms": rec.get("latency_ms", 0),
                        "tokens_in": rec.get("tokens_in", 0),
                        "tokens_out": rec.get("tokens_out", 0),
                        "output": rec.get("output"),  # 用于提取预览
                    }
                    global_line += 1
                except (json.JSONDecodeError, Exception):
                    continue


def _get_record_at_line(log_dir: Path, target_line: int) -> Optional[dict]:
    """获取指定全局行号的完整记录。"""
    for rec in _iter_records(log_dir):
        if rec.get("_line") == target_line:
            return rec
    return None


def _update_keys_models_cache(log_dir: Path, force: bool = False) -> None:
    """
    更新 api_keys 和 models 缓存。
    只在文件修改时才重新扫描。
    """
    log_paths = _get_all_log_paths(log_dir)
    current_mtimes = _get_mtimes(log_paths)

    # 如果文件没有变化，不更新
    if not force and current_mtimes == _cache.get("mtimes", {}):
        return

    # 重新扫描所有文件收集 api_keys 和 models
    api_keys_set: set[str] = set()
    models_set: set[str] = set()

    for rec in _iter_records(log_dir):
        if rec.get("api_key"):
            api_keys_set.add(rec["api_key"])
        if rec.get("model"):
            models_set.add(rec["model"])

    _cache["mtimes"] = current_mtimes
    _cache["api_keys"] = sorted(api_keys_set)
    _cache["models"] = sorted(models_set)


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
    """
    列出对话记录（仅元数据，不含 messages/output）。

    内存优化：只读取轻量元数据，不读取完整的 messages 和 output。
    """
    log_dir = _get_log_dir(request.app.state)

    # 更新 api_keys 和 models 缓存
    _update_keys_models_cache(log_dir)

    # 构建筛选条件
    filters = {}
    if api_key:
        filters["api_key"] = api_key
    if model:
        filters["model"] = model
    if success is not None and success != "":
        filters["success"] = success.lower() == "true"

    # 收集符合筛选条件的轻量元数据
    # 注意：这里只存储轻量元数据，不存储完整的 messages/output
    filtered_items = []
    for rec in _iter_light_metadata(log_dir):
        # 应用筛选条件
        if filters.get("api_key") and rec.get("api_key") != filters["api_key"]:
            continue
        if filters.get("model") and rec.get("model") != filters["model"]:
            continue
        if (
            filters.get("success") is not None
            and rec.get("success") != filters["success"]
        ):
            continue

        # 提取预览
        preview, has_tool_use = _output_preview(rec)
        filtered_items.append(
            {
                "line": rec["_line"],
                "timestamp": rec["timestamp"],
                "request_id": rec["request_id"],
                "model": rec["model"],
                "adapter": rec["adapter"],
                "api_key": rec["api_key"],
                "success": rec["success"],
                "latency_ms": rec["latency_ms"],
                "tokens_in": rec["tokens_in"],
                "tokens_out": rec["tokens_out"],
                "output_preview": preview,
                "has_tool_use": has_tool_use,
            }
        )

    # 反转（最新在前）并分页
    total = len(filtered_items)
    filtered_items.reverse()
    page_items = filtered_items[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": page_items,
        "api_keys": _cache.get("api_keys", []),
        "models": _cache.get("models", []),
    }


@router.get("/api/conversations/{line}")
async def get_conversation_detail(request: Request, line: int):
    """获取某行对话的完整记录（含 messages 和 output）。"""
    log_dir = _get_log_dir(request.app.state)

    rec = _get_record_at_line(log_dir, line)
    if rec:
        # 返回完整记录（去掉内部字段）
        result = {k: v for k, v in rec.items() if not k.startswith("_")}
        result["line"] = line
        return result

    return JSONResponse(status_code=404, content={"error": "Record not found"})
