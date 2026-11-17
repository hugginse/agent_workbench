"""任务（Task）领域 schema。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = ["Task", "TaskStatus"]


class TaskStatus(StrEnum):
    """任务生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utc_now() -> datetime:
    """返回带时区的 UTC 当前时间。"""
    return datetime.now(UTC)


def _new_id() -> str:
    """生成任务标识（UUID4 字符串）。"""
    return str(uuid4())


class Task(BaseModel):
    """用户提交的一次任务。

    Attributes:
        id: 任务唯一标识，缺省自动生成 UUID4。
        description: 任务描述（用户输入的原始目标）。
        status: 生命周期状态，默认 pending。
        created_at: 创建时间（UTC）。
        updated_at: 最近更新时间（UTC）。
        error: 失败时的用户可见错误信息；成功时为 None。
    """

    id: str = Field(default_factory=_new_id, description="任务唯一标识")
    description: str = Field(min_length=1, description="任务描述")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    created_at: datetime = Field(default_factory=_utc_now, description="创建时间 UTC")
    updated_at: datetime = Field(default_factory=_utc_now, description="更新时间 UTC")
    error: str | None = Field(default=None, description="失败原因（用户可见）")
