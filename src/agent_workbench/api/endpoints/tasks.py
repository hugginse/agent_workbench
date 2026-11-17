"""任务端点：创建、查询、SSE 进度流。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_workbench.application.services.task import TaskNotFoundError, TaskService
from agent_workbench.domain.schemas import Task

__all__ = ["router"]

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskCreateRequest(BaseModel):
    """POST /tasks 请求体。"""

    description: str = Field(min_length=1, max_length=8000, description="任务描述")


def get_task_service(request: Request) -> TaskService:
    """从应用状态解析 TaskService（构造在 create_app 完成）。

    Raises:
        HTTPException: 服务未装配（500）。
    """
    service: TaskService | None = getattr(request.app.state, "task_service", None)
    if service is None:
        msg = "任务服务未初始化"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return service


def _sse(data: dict[str, Any]) -> str:
    """编码一条 SSE data 行。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
async def create_task(
    body: TaskCreateRequest,
    service: TaskService = Depends(get_task_service),
) -> Task:
    """创建任务并立即开始执行。"""
    return await service.create_task(body.description)


@router.get("/{task_id}", response_model=Task)
def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
) -> Task:
    """查询任务当前状态。"""
    try:
        return service.get_task(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在") from exc


@router.get("/{task_id}/stream")
async def stream_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
) -> StreamingResponse:
    """SSE 进度流：重放已有事件，再推送增量，终态后结束。

    事件类型：task_started / task_completed / task_failed。
    """
    try:
        service.get_task(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在") from exc

    async def event_generator() -> AsyncIterator[str]:
        async for event in service.stream_events(task_id):
            yield _sse(event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
