"""Conversation log SQLite index — metadata + byte offset lookup."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional


class ConvIndexer:
    """SQLite index for conversation JSONL logs.

    Stores lightweight metadata + (file_path, byte_offset, line_length)
    so that list queries hit SQLite and detail reads use fseek.
    """

    def __init__(self, db_path: str = "data/conv_index.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if not exist."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conv_index (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                request_id   TEXT DEFAULT '',
                model        TEXT NOT NULL DEFAULT '',
                adapter      TEXT DEFAULT '',
                api_key      TEXT DEFAULT '',
                success      INTEGER NOT NULL DEFAULT 1,
                latency_ms   REAL DEFAULT 0,
                tokens_in    INTEGER DEFAULT 0,
                tokens_out   INTEGER DEFAULT 0,
                has_tool_use INTEGER DEFAULT 0,
                preview      TEXT DEFAULT '',
                file_path    TEXT NOT NULL DEFAULT '',
                byte_offset  INTEGER NOT NULL DEFAULT 0,
                line_length  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conv_index(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_conv_model ON conv_index(model);
            CREATE INDEX IF NOT EXISTS idx_conv_api_key ON conv_index(api_key);
            CREATE INDEX IF NOT EXISTS idx_conv_success ON conv_index(success);
            CREATE INDEX IF NOT EXISTS idx_conv_file ON conv_index(file_path);
        """)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ConvIndexer is closed")
        return self._conn

    # ========== Write ==========

    def index(
        self, *, record: dict, file_path: str, byte_offset: int, line_length: int
    ):
        """Insert a single record into the index."""
        preview, has_tool_use = self._extract_preview(record)
        self.conn.execute(
            """INSERT INTO conv_index
               (timestamp, request_id, model, adapter, api_key, success,
                latency_ms, tokens_in, tokens_out, has_tool_use, preview,
                file_path, byte_offset, line_length)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("timestamp", ""),
                record.get("request_id", ""),
                record.get("model", ""),
                record.get("adapter", ""),
                record.get("api_key", ""),
                1 if record.get("success") else 0,
                record.get("latency_ms", 0),
                record.get("tokens_in", 0),
                record.get("tokens_out", 0),
                1 if has_tool_use else 0,
                preview,
                file_path,
                byte_offset,
                line_length,
            ),
        )
        self.conn.commit()

    def batch_index(self, entries: list[dict]):
        """Insert multiple records in a single transaction."""
        rows = []
        for entry in entries:
            record = entry["record"]
            preview, has_tool_use = self._extract_preview(record)
            rows.append(
                (
                    record.get("timestamp", ""),
                    record.get("request_id", ""),
                    record.get("model", ""),
                    record.get("adapter", ""),
                    record.get("api_key", ""),
                    1 if record.get("success") else 0,
                    record.get("latency_ms", 0),
                    record.get("tokens_in", 0),
                    record.get("tokens_out", 0),
                    1 if has_tool_use else 0,
                    preview,
                    entry.get("file_path", ""),
                    entry.get("byte_offset", 0),
                    entry.get("line_length", 0),
                )
            )
        self.conn.executemany(
            """INSERT INTO conv_index
               (timestamp, request_id, model, adapter, api_key, success,
                latency_ms, tokens_in, tokens_out, has_tool_use, preview,
                file_path, byte_offset, line_length)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    # ========== Read ==========

    def query(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Query metadata with filters and pagination.

        Returns (items, total_count).
        """
        conditions = []
        params: list = []

        if model is not None:
            conditions.append("model = ?")
            params.append(model)
        if api_key is not None:
            conditions.append("api_key = ?")
            params.append(api_key)
        if success is not None:
            conditions.append("success = ?")
            params.append(1 if success else 0)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Total count
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM conv_index {where}", params
        ).fetchone()
        total = row[0]

        # Paginated results (newest first)
        rows = self.conn.execute(
            f"""SELECT id, timestamp, request_id, model, adapter, api_key,
                       success, latency_ms, tokens_in, tokens_out,
                       has_tool_use, preview
                FROM conv_index {where}
                ORDER BY id DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        items = [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "request_id": r["request_id"],
                "model": r["model"],
                "adapter": r["adapter"],
                "api_key": r["api_key"],
                "success": bool(r["success"]),
                "latency_ms": r["latency_ms"],
                "tokens_in": r["tokens_in"],
                "tokens_out": r["tokens_out"],
                "output_preview": r["preview"],
                "has_tool_use": bool(r["has_tool_use"]),
            }
            for r in rows
        ]

        return items, total

    def get_location(self, record_id: int) -> Optional[dict]:
        """Get file location for a record by ID.

        Returns dict with file_path, byte_offset, line_length or None.
        """
        row = self.conn.execute(
            "SELECT file_path, byte_offset, line_length FROM conv_index WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row:
            return {
                "file_path": row["file_path"],
                "byte_offset": row["byte_offset"],
                "line_length": row["line_length"],
            }
        return None

    def read_record(self, record_id: int, log_dir: str) -> Optional[dict]:
        """Read a full record from the JSONL file using the index.

        Uses fseek to jump directly to the record's byte offset.
        """
        loc = self.get_location(record_id)
        if not loc:
            return None

        full_path = Path(log_dir) / loc["file_path"]
        if not full_path.exists():
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                f.seek(loc["byte_offset"])
                raw = f.read(loc["line_length"])
                record = json.loads(raw)
            record["id"] = record_id
            return record
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def get_distinct(self, field: str) -> list[str]:
        """Get distinct values for a field (for filter dropdowns)."""
        if field not in ("model", "api_key", "adapter"):
            return []
        rows = self.conn.execute(
            f"SELECT DISTINCT {field} FROM conv_index WHERE {field} != '' ORDER BY {field}"
        ).fetchall()
        return [r[0] for r in rows]

    def get_stats(self) -> dict:
        """Get index statistics."""
        row = self.conn.execute("SELECT COUNT(*) FROM conv_index").fetchone()
        total = row[0]

        file_info = self.conn.execute(
            "SELECT file_path, COUNT(*) as cnt FROM conv_index GROUP BY file_path ORDER BY file_path"
        ).fetchall()

        files = {r["file_path"]: r["cnt"] for r in file_info}

        db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0

        return {
            "total_records": total,
            "files": files,
            "index_size_bytes": db_size,
            "db_path": self._db_path,
        }

    def record_count(self) -> int:
        """Quick count of indexed records."""
        row = self.conn.execute("SELECT COUNT(*) FROM conv_index").fetchone()
        return row[0]

    # ========== Rebuild ==========

    def clear(self):
        """Delete all records from the index."""
        self.conn.execute("DELETE FROM conv_index")
        self.conn.commit()

    def rebuild_from_logs(self, log_dir: str):
        """Full rebuild: scan all JSONL files and re-index everything.

        This is expensive (O(N) full scan) but only needed:
        - On first run (no index exists)
        - Manual rebuild triggered by admin
        """
        import time as _time

        base_dir = Path(log_dir)
        base_path = base_dir / "conversations.jsonl"
        if not base_path.exists():
            return 0

        # Find all rotated files (old to new)
        paths = []
        for i in range(1, 100):
            p = base_dir / f"conversations.jsonl.{i}"
            if p.exists():
                paths.append(p)
            else:
                break
        # Oldest first: .99, .98, ..., .2, .1
        paths.sort(key=lambda p: int(p.suffix[1:]), reverse=True)
        paths.append(base_path)

        # Clear existing index
        self.clear()

        total_indexed = 0
        batch = []
        batch_size = 500

        for file_path in paths:
            if not file_path.exists():
                continue
            offset = 0
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.rstrip("\n")
                        if not stripped.strip():
                            offset += len(line.encode("utf-8"))
                            continue
                        try:
                            rec = json.loads(stripped)
                            batch.append(
                                {
                                    "record": rec,
                                    "file_path": file_path.name,
                                    "byte_offset": offset,
                                    "line_length": len(stripped.encode("utf-8")),
                                }
                            )
                            total_indexed += 1
                        except (json.JSONDecodeError, Exception):
                            pass
                        offset += len(line.encode("utf-8"))

                        if len(batch) >= batch_size:
                            self.batch_index(batch)
                            batch = []
            except OSError:
                continue

        if batch:
            self.batch_index(batch)

        return total_indexed

    # ========== Internal ==========

    @staticmethod
    def _extract_preview(record: dict) -> tuple[str, bool]:
        """Extract preview text and tool_use flag from a record's output field."""
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


# ========== Module-level singleton accessor ==========
# Set by app startup in main.py; used by tracking.py to index records
# without needing access to app.state.

_instance: Optional[ConvIndexer] = None


def get_conv_indexer() -> Optional[ConvIndexer]:
    """Return the currently active ConvIndexer instance, or None."""
    return _instance


def set_conv_indexer(indexer: ConvIndexer):
    """Set the module-level ConvIndexer instance (called from app startup)."""
    global _instance
    _instance = indexer
