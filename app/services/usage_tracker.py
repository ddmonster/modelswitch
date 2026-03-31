from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.config_models import ApiKeyConfig

logger = logging.getLogger(__name__)


class UsageTracker:
    """用量统计追踪器：SQLite 持久化 + 内存热缓存"""

    def __init__(self, db_path: str = "data/usage.db", flush_interval: int = 10):
        self.db_path = db_path
        self.flush_interval = flush_interval
        self._pending_records: List[tuple] = []
        self._flush_task = None
        self._shared_db = None  # For in-memory testing: shared aiosqlite connection

    def _connect(self):
        """Get a database connection context manager. Reuses shared connection if available."""
        import aiosqlite
        if self._shared_db is not None:
            return _SharedDbCtx(self._shared_db)
        return aiosqlite.connect(self.db_path)

    async def init(self) -> None:
        """初始化数据库"""
        import aiosqlite
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        async with self._connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    api_key_alias TEXT,
                    api_key_masked TEXT,
                    success INTEGER DEFAULT 1,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0,
                    status_code INTEGER DEFAULT 200
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_records(provider)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_records(model)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage_records(api_key_alias)
            """)
            await db.commit()

    async def record(
        self,
        provider: str,
        model: str,
        api_key_alias: str = "",
        api_key_masked: str = "",
        success: bool = True,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0,
        status_code: int = 200,
    ) -> None:
        """记录一次请求"""
        now = datetime.now().isoformat()
        self._pending_records.append((
            now, provider, model, api_key_alias, api_key_masked,
            1 if success else 0, tokens_in, tokens_out, latency_ms, status_code
        ))

    async def flush(self) -> None:
        """批量写入到 SQLite"""
        if not self._pending_records:
            return

        records = self._pending_records[:]
        self._pending_records.clear()

        try:
            async with self._connect() as db:
                await db.executemany(
                    """INSERT INTO usage_records
                       (timestamp, provider, model, api_key_alias, api_key_masked,
                        success, tokens_in, tokens_out, latency_ms, status_code)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    records,
                )
                await db.commit()
        except Exception as e:
            logger.error(f"flush usage records failed: {e}")
            self._pending_records.extend(records)

    async def aggregate(
        self,
        group_by: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        按维度聚合查询。
        group_by: "provider" | "model" | "api_key"
        """
        import aiosqlite

        col_map = {
            "provider": "provider",
            "model": "model",
            "api_key": "api_key_alias",
        }
        col = col_map.get(group_by, "provider")

        where_clauses = []
        params = []
        if date_from:
            where_clauses.append("timestamp >= ?")
            params.append(f"{date_from}T00:00:00")
        if date_to:
            where_clauses.append("timestamp <= ?")
            params.append(f"{date_to}T23:59:59")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row

            # 主聚合查询
            query = f"""
                SELECT {col} as name,
                       COUNT(*) as total_requests,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as success_count,
                       SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as fail_count,
                       COALESCE(SUM(tokens_in), 0) as tokens_in,
                       COALESCE(SUM(tokens_out), 0) as tokens_out,
                       COALESCE(AVG(latency_ms), 0) as avg_latency_ms
                FROM usage_records
                {where_sql}
                GROUP BY {col}
                ORDER BY total_requests DESC
            """
            rows = await db.execute(query, params)
            groups = []
            total = 0
            for row in await rows.fetchall():
                total += row["total_requests"]
                success_rate = (row["success_count"] / row["total_requests"] * 100) if row["total_requests"] > 0 else 0
                groups.append({
                    "name": row["name"],
                    "total_requests": row["total_requests"],
                    "success_count": row["success_count"],
                    "fail_count": row["fail_count"],
                    "success_rate": round(success_rate, 1),
                    "tokens_in": row["tokens_in"],
                    "tokens_out": row["tokens_out"],
                    "avg_latency_ms": round(row["avg_latency_ms"], 0),
                })

            return {
                "group_by": group_by,
                "total": total,
                "groups": groups,
            }

    async def get_detail(
        self,
        group_by: str,
        item_name: str,
        sub_group: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        extra_filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """下钻查询：查看某个分组下的子维度明细，支持多级过滤"""
        import aiosqlite

        col_map = {"provider": "provider", "model": "model", "api_key": "api_key_alias"}
        parent_col = col_map.get(group_by, group_by)
        child_col = col_map.get(sub_group, sub_group)

        where_clauses = [f"{parent_col} = ?"]
        params: list = [item_name]

        # 额外过滤条件（多级下钻用）
        if extra_filters:
            for dim, value in extra_filters.items():
                fc = col_map.get(dim, dim)
                where_clauses.append(f"{fc} = ?")
                params.append(value)

        if date_from:
            where_clauses.append("timestamp >= ?")
            params.append(f"{date_from}T00:00:00")
        if date_to:
            where_clauses.append("timestamp <= ?")
            params.append(f"{date_to}T23:59:59")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            query = f"""
                SELECT {child_col} as name,
                       COUNT(*) as total_requests,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as success_count,
                       SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as fail_count,
                       COALESCE(SUM(tokens_in), 0) as tokens_in,
                       COALESCE(SUM(tokens_out), 0) as tokens_out,
                       COALESCE(AVG(latency_ms), 0) as avg_latency_ms
                FROM usage_records
                {where_sql}
                GROUP BY {child_col}
                ORDER BY total_requests DESC
            """
            rows = await db.execute(query, params)
            results = []
            for row in await rows.fetchall():
                results.append({
                    "name": row["name"],
                    "total_requests": row["total_requests"],
                    "success_count": row["success_count"],
                    "fail_count": row["fail_count"],
                    "tokens_in": row["tokens_in"],
                    "tokens_out": row["tokens_out"],
                    "avg_latency_ms": round(row["avg_latency_ms"], 0),
                })
            return results

    async def close(self) -> None:
        """关闭前刷新所有待写入记录"""
        await self.flush()
        if self._shared_db is not None:
            await self._shared_db.close()
            self._shared_db = None


class _SharedDbCtx:
    """Context manager wrapper for a shared aiosqlite connection."""

    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *args):
        pass
