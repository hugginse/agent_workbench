"""指标端点：GET /metrics（成功率 / 延迟 / token 成本）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from agent_workbench.application.services.task import TaskService
from agent_workbench.observability.metrics import ServiceMetrics, compute_service_metrics

__all__ = ["router"]

router = APIRouter(tags=["metrics"])


def _get_service(request: Request) -> TaskService:
    """解析 TaskService；未装配时 500。"""
    from fastapi import HTTPException, status

    service: TaskService | None = getattr(request.app.state, "task_service", None)
    if service is None:
        msg = "任务服务未初始化"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return service


@router.get("/metrics", response_model=ServiceMetrics)
def get_metrics(service: TaskService = Depends(_get_service)) -> ServiceMetrics:
    """返回服务级成功率、延迟分位与 token 成本（估算）。"""
    return compute_service_metrics(service.list_metric_samples())
