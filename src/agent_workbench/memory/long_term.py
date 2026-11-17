"""长期记忆：任务复盘 Protocol 与 SQLite 实现。

Supervisor 经后续 `memory_recall` 工具检索注入（Phase 3 #19）。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = [
    "LongTermMemoryRepository",
    "SqliteLongTermMemory",
    "TaskReflection",
]

# 复盘结局（与任务终态对齐，便于检索过滤）
_OUTCOMES = frozenset({"completed", "failed", "degraded"})


def _utc_now() -> datetime:
    """带时区的 UTC 当前时间。"""
    return datetime.now(UTC)


class TaskReflection(BaseModel):
    """一次任务完成后的复盘记录。

    Attributes:
        id: 记录唯一标识。
        task_id: 对应任务 id（同任务重复写入覆盖）。
        task_description: 任务描述（检索用）。
        summary: 复盘摘要（关键结论 / 教训 / 可复用要点）。
        outcome: 结局 completed | failed | degraded。
        created_at: 写入时间（UTC）。
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="记录标识")
    task_id: str = Field(min_length=1, description="任务标识")
    task_description: str = Field(min_length=1, description="任务描述")
    summary: str = Field(min_length=1, description="复盘摘要")
    outcome: str = Field(default="completed", description="结局")
    created_at: datetime = Field(default_factory=_utc_now, description="写入时间 UTC")

    def model_post_init(self, __context: object) -> None:
        """校验 outcome 合法（Pydantic v2 hook）。"""
        if self.outcome not in _OUTCOMES:
            msg = f"outcome 必须是 {_OUTCOMES} 之一，当前 {self.outcome!r}"
            raise ValueError(msg)


@runtime_checkable
class LongTermMemoryRepository(Protocol):
    """长期记忆端口：保存 / 查询任务复盘。"""

    def save(self, reflection: TaskReflection) -> None:
        """写入复盘；同 task_id 覆盖旧记录。"""
        ...

    def get_by_task_id(self, task_id: str) -> TaskReflection | None:
        """按任务 id 读取；不存在返回 None。"""
        ...

    def search(self, query: str, *, limit: int = 5) -> list[TaskReflection]:
        """按关键词检索 summary / task_description（相关度由调用方排序）。"""
        ...

    def list_recent(self, *, limit: int = 10) -> list[TaskReflection]:
        """按写入时间倒序返回最近复盘。"""
        ...


class SqliteLongTermMemory:
    """基于 SQLite 的长期记忆实现（自包含，无外部服务）。"""

    def __init__(self, db_path: str | Path) -> None:
        """打开/创建数据库并初始化表。

        Args:
            db_path: SQLite 文件路径；`:memory:` 为内存库。
        """
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """创建复盘表（幂等）。"""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_reflections (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                task_description TEXT NOT NULL,
                summary TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reflections_created_at "
            "ON task_reflections(created_at DESC)"
        )
        self._conn.commit()

    def save(self, reflection: TaskReflection) -> None:
        """写入或按 task_id 覆盖复盘记录。"""
        self._conn.execute(
            """
            INSERT INTO task_reflections
                (id, task_id, task_description, summary, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                id = excluded.id,
                task_description = excluded.task_description,
                summary = excluded.summary,
                outcome = excluded.outcome,
                created_at = excluded.created_at
            """,
            (
                reflection.id,
                reflection.task_id,
                reflection.task_description,
                reflection.summary,
                reflection.outcome,
                reflection.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_by_task_id(self, task_id: str) -> TaskReflection | None:
        """按任务 id 读取复盘。"""
        row = self._conn.execute(
            "SELECT * FROM task_reflections WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return self._to_model(row) if row else None

    def search(self, query: str, *, limit: int = 5) -> list[TaskReflection]:
        """关键词 LIKE 检索（summary / task_description）。"""
        if limit < 1:
            return []
        pattern = f"%{query.strip()}%"
        if not query.strip():
            return self.list_recent(limit=limit)
        rows = self._conn.execute(
            """
            SELECT * FROM task_reflections
            WHERE summary LIKE ? OR task_description LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
        return [self._to_model(r) for r in rows]

    def list_recent(self, *, limit: int = 10) -> list[TaskReflection]:
        """最近复盘（时间倒序）。"""
        if limit < 1:
            return []
        rows = self._conn.execute(
            """
            SELECT * FROM task_reflections
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._to_model(r) for r in rows]

    def close(self) -> None:
        """关闭底层连接。"""
        self._conn.close()

    def __enter__(self) -> SqliteLongTermMemory:
        """支持 with 上下文管理。"""
        return self

    def __exit__(self, *exc: object) -> None:
        """退出时关闭连接。"""
        self.close()

    @staticmethod
    def _to_model(row: sqlite3.Row) -> TaskReflection:
        """行记录 → TaskReflection。"""
        created = datetime.fromisoformat(str(row["created_at"]))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return TaskReflection(
            id=str(row["id"]),
            task_id=str(row["task_id"]),
            task_description=str(row["task_description"]),
            summary=str(row["summary"]),
            outcome=str(row["outcome"]),
            created_at=created,
        )
