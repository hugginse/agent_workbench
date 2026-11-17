"""任务结果（TaskResult）领域 schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_workbench.domain.schemas.task import TaskStatus

__all__ = ["TaskResult"]


class TaskResult(BaseModel):
    """任务执行的结构化交付结果。

    Attributes:
        task_id: 对应 Task.id。
        status: 终态（completed / failed；不接受中间态）。
        answer: 最终交付正文。
        artifacts: 产出物标识列表（路径或结构化产物 id）。
        error: 失败时的用户可见原因；成功时为 None。
    """

    task_id: str = Field(min_length=1, description="任务标识")
    status: TaskStatus = Field(description="任务终态")
    answer: str = Field(default="", description="最终交付正文")
    artifacts: list[str] = Field(default_factory=list, description="产出物列表")
    error: str | None = Field(default=None, description="失败原因（用户可见）")
