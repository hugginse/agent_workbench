"""Node 级 trace：span 模型与 SQLite 存储（可回放）。

写入为旁路：span 失败不影响任务执行。
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = ["NoopSpanStore", "SpanRecord", "SpanStore", "SqliteSpanStore"]


def _utc_now() -> datetime:
    """带时区的 UTC 当前时间。"""
    return datetime.now(UTC)


class SpanRecord(BaseModel):
    """一次节点/阶段 span。

    Attributes:
        id: span 唯一 id。
        task_id: 所属任务。
        name: 节点名（如 plan / execute / agent）。
        status: ok | error。
        started_at: 开始（UTC）。
        duration_ms: 耗时毫秒。
        detail: 可选摘要（不存完整 prompt/密钥）。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: str = Field(default="ok")
    started_at: datetime = Field(default_factory=_utc_now)
    duration_ms: float = Field(default=0.0, ge=0.0)
    detail: str = Field(default="")


@runtime_checkable
class SpanStore(Protocol):
    """span 存储端口。"""

    def record(self, span: SpanRecord) -> None:
        """写入一条 span。"""
        ...

    def list_by_task(self, task_id: str) -> list[SpanRecord]:
        """按任务升序返回 span（用于回放）。"""
        ...


class SqliteSpanStore:
    """SQLite span 仓库（`.data/traces.db` 或测试临时路径）。"""

    def __init__(self, db_path: str | Path) -> None:
        """打开/创建库并建表。

        Args:
            db_path: SQLite 文件路径。
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                detail TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_spans_task ON spans(task_id, started_at)"
        )
        self._conn.commit()

    def record(self, span: SpanRecord) -> None:
        """写入 span。"""
        self._conn.execute(
            """
            INSERT INTO spans
                (id, task_id, name, status, started_at, duration_ms, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                span.id,
                span.task_id,
                span.name,
                span.status,
                span.started_at.isoformat(),
                span.duration_ms,
                span.detail,
            ),
        )
        self._conn.commit()

    def list_by_task(self, task_id: str) -> list[SpanRecord]:
        """按时间升序读取任务 span。"""
        rows = self._conn.execute(
            """
            SELECT * FROM spans
            WHERE task_id = ?
            ORDER BY started_at ASC, rowid ASC
            """,
            (task_id,),
        ).fetchall()
        out: list[SpanRecord] = []
        for row in rows:
            started = datetime.fromisoformat(str(row["started_at"]))
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            out.append(
                SpanRecord(
                    id=str(row["id"]),
                    task_id=str(row["task_id"]),
                    name=str(row["name"]),
                    status=str(row["status"]),
                    started_at=started,
                    duration_ms=float(row["duration_ms"]),
                    detail=str(row["detail"] or ""),
                )
            )
        return out

    def close(self) -> None:
        """关闭连接。"""
        self._conn.close()


class NoopSpanStore:
    """空实现（未启用 trace 时的占位）。"""

    def record(self, span: SpanRecord) -> None:
        """丢弃 span。"""
        return None

    def list_by_task(self, task_id: str) -> list[SpanRecord]:
        """返回空列表。"""
        return []
