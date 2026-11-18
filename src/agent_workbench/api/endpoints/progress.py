"""进度回放端点：GET /progress/{task_id}。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from agent_workbench.application.services.task import TaskNotFoundError, TaskService
from agent_workbench.observability.tracing import SpanRecord, SpanStore

__all__ = ["router"]

router = APIRouter(prefix="/progress", tags=["progress"])


class ProgressResponse(BaseModel):
    """任务 span 回放响应。"""

    task_id: str
    status: str = Field(description="任务当前状态")
    spans: list[SpanRecord] = Field(default_factory=list)


def _get_services(request: Request) -> tuple[TaskService, SpanStore]:
    """解析 TaskService 与 SpanStore。

    Raises:
        HTTPException: 未装配（500）。
    """
    service = getattr(request.app.state, "task_service", None)
    store = getattr(request.app.state, "span_store", None)
    if service is None or store is None:
        msg = "服务未初始化"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return service, store


@router.get("/{task_id}", response_model=ProgressResponse)
def get_progress(
    task_id: str,
    services: tuple[TaskService, SpanStore] = Depends(_get_services),
) -> ProgressResponse:
    """回放任务节点 span（按时间升序）。

    Raises:
        HTTPException: 任务不存在（404）。
    """
    service, store = services
    try:
        task = service.get_task(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在") from exc
    spans = store.list_by_task(task_id)
    return ProgressResponse(task_id=task_id, status=task.status.value, spans=spans)
