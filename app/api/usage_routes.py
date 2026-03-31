from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

router = APIRouter(tags=["usage"])


@router.get("/api/usage")
async def get_usage(
    request: Request,
    group_by: str = Query("provider", enum=["provider", "model", "api_key"]),
    date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """按维度聚合用量统计"""
    tracker = request.app.state.usage_tracker
    today = datetime.now().strftime("%Y-%m-%d")

    if date:
        date_from = date
        date_to = date

    result = await tracker.aggregate(
        group_by=group_by,
        date_from=date_from or today,
        date_to=date_to or today,
    )
    return result


@router.get("/api/usage/{item_name}/detail")
async def get_usage_detail(
    request: Request,
    item_name: str,
    group_by: str = Query("provider"),
    sub_group: str = Query("model"),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """下钻查看子维度明细"""
    tracker = request.app.state.usage_tracker
    today = datetime.now().strftime("%Y-%m-%d")

    result = await tracker.get_detail(
        group_by=group_by,
        item_name=item_name,
        sub_group=sub_group,
        date_from=date_from or today,
        date_to=date_to or today,
    )
    return result
