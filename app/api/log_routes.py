from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

from app.utils.logging import get_logs_filtered

router = APIRouter(tags=["logs"])


@router.get("/api/logs")
async def get_logs(
    request: Request,
    tail: int = Query(100, ge=1, le=1000),
    level: Optional[str] = None,
    request_id: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """查询调试日志"""
    logs = get_logs_filtered(tail=tail, level=level, request_id=request_id, api_key=api_key)
    return {"total": len(logs), "logs": logs}
