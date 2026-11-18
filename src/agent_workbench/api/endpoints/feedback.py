"""反馈端点：对任务 up/down 评分（内存存储，进程内有效）。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from agent_workbench.application.services.task import TaskNotFoundError, TaskService

__all__ = ["FeedbackStore", "router"]

router = APIRouter(prefix="/feedback", tags=["feedback"])

Rating = Literal["up", "down"]


class FeedbackRequest(BaseModel):
    """POST /feedback 请求体。"""

    task_id: str = Field(min_length=1)
    rating: Rating = Field(description="up 或 down")


class FeedbackResponse(BaseModel):
    """反馈结果。"""

    task_id: str
    rating: Rating
    count_up: int
    count_down: int


class FeedbackStore:
    """任务反馈聚合（内存；同任务可重复提交，取最近一次 rating 并计次）。"""

    def __init__(self) -> None:
        self._ratings: dict[str, Rating] = {}
        self._up: dict[str, int] = {}
        self._down: dict[str, int] = {}

    def submit(self, task_id: str, rating: Rating) -> FeedbackResponse:
        """记录一次反馈并返回聚合计数。"""
        prev = self._ratings.get(task_id)
        if prev == "up":
            self._up[task_id] = max(0, self._up.get(task_id, 0) - 1)
        elif prev == "down":
            self._down[task_id] = max(0, self._down.get(task_id, 0) - 1)
        self._ratings[task_id] = rating
        if rating == "up":
            self._up[task_id] = self._up.get(task_id, 0) + 1
        else:
            self._down[task_id] = self._down.get(task_id, 0) + 1
        return FeedbackResponse(
            task_id=task_id,
            rating=rating,
            count_up=self._up.get(task_id, 0),
            count_down=self._down.get(task_id, 0),
        )

    def get(self, task_id: str) -> FeedbackResponse | None:
        """读取当前反馈；无记录返回 None。"""
        rating = self._ratings.get(task_id)
        if rating is None:
            return None
        return FeedbackResponse(
            task_id=task_id,
            rating=rating,
            count_up=self._up.get(task_id, 0),
            count_down=self._down.get(task_id, 0),
        )


def get_feedback_store(request: Request) -> FeedbackStore:
    """从 app.state 取 FeedbackStore（create_app 装配）。

    Raises:
        HTTPException: 未装配（500）。
    """
    store: FeedbackStore | None = getattr(request.app.state, "feedback_store", None)
    if store is None:
        msg = "反馈服务未初始化"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return store


def get_task_service(request: Request) -> TaskService:
    """解析 TaskService。

    Raises:
        HTTPException: 未装配（500）。
    """
    service: TaskService | None = getattr(request.app.state, "task_service", None)
    if service is None:
        msg = "任务服务未初始化"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return service


@router.post("", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    store: FeedbackStore = Depends(get_feedback_store),
    service: TaskService = Depends(get_task_service),
) -> FeedbackResponse:
    """提交任务 up/down 反馈；任务须存在。

    Raises:
        HTTPException: 任务不存在（404）。
    """
    try:
        service.get_task(body.task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在") from exc
    return store.submit(body.task_id, body.rating)


@router.get("/{task_id}", response_model=FeedbackResponse)
def get_feedback(
    task_id: str,
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    """查询任务当前反馈。

    Raises:
        HTTPException: 无反馈（404）。
    """
    result = store.get(task_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="暂无反馈")
    return result
