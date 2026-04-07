"""对话记录查看 API —— 使用 SQLite 索引进行高效查询。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["conversations"])


def _get_log_dir(app_state) -> str:
    """获取日志目录路径。"""
    config = getattr(app_state, "config", None)
    log_dir = "logs"
    if config and hasattr(config, "gateway"):
        log_dir = getattr(config.gateway, "log_dir", "logs")
    return log_dir


def _get_indexer(request: Request):
    """Get the ConvIndexer from app state."""
    return getattr(request.app.state, "conv_indexer", None)


@router.get("/api/conversations")
async def list_conversations(
    request: Request,
    model: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    success: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    列出对话记录（仅元数据，不含 messages/output）。

    使用 SQLite 索引查询，避免全文件扫描。
    如果索引不可用，返回空结果。
    """
    indexer = _get_indexer(request)
    if not indexer:
        return {
            "total": 0,
            "limit": limit,
            "offset": offset,
            "items": [],
            "api_keys": [],
            "models": [],
        }

    # Parse success filter
    success_bool = None
    if success is not None and success != "":
        success_bool = success.lower() == "true"

    # Query from index
    items, total = indexer.query(
        model=model,
        api_key=api_key,
        success=success_bool,
        limit=limit,
        offset=offset,
    )

    # Get distinct values for filter dropdowns
    api_keys = indexer.get_distinct("api_key")
    models = indexer.get_distinct("model")

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
        "api_keys": api_keys,
        "models": models,
    }


@router.get("/api/conversations/{record_id}")
async def get_conversation_detail(request: Request, record_id: int):
    """获取指定记录的完整内容（含 messages 和 output）。

    使用索引定位后通过 fseek 直接读取，时间复杂度 O(1)。
    """
    indexer = _get_indexer(request)
    if not indexer:
        return JSONResponse(status_code=404, content={"error": "Index not available"})

    log_dir = _get_log_dir(request.app.state)
    record = indexer.read_record(record_id, log_dir)

    if record:
        # Remove internal fields, keep everything else
        result = {k: v for k, v in record.items()}
        return result

    return JSONResponse(status_code=404, content={"error": "Record not found"})


@router.post("/api/conversations/rebuild")
async def rebuild_index(request: Request):
    """重建会话日志索引（管理员操作）。

    全量扫描所有 JSONL 文件并重建 SQLite 索引。
    这是耗时操作，仅在索引损坏时使用。
    """
    indexer = _get_indexer(request)
    if not indexer:
        return JSONResponse(status_code=500, content={"error": "Index not initialized"})

    log_dir = _get_log_dir(request.app.state)
    try:
        count = indexer.rebuild_from_logs(log_dir)
        return {"message": f"Index rebuilt: {count} records indexed", "total": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/conversations/index/stats")
async def get_index_stats(request: Request):
    """获取索引统计信息。"""
    indexer = _get_indexer(request)
    if not indexer:
        return JSONResponse(status_code=404, content={"error": "Index not available"})

    return indexer.get_stats()
