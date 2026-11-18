"""服务级指标：成功率 / 延迟分位 / token 成本（GET /metrics）。

数据来自 TaskService 终态采样；token 为启发式估算（与 harness 一致）。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from agent_workbench.application.services.task import TaskMetricSample
from agent_workbench.harness.metrics import percentile

__all__ = ["ServiceMetrics", "compute_service_metrics"]


class ServiceMetrics(BaseModel):
    """聚合服务指标。"""

    task_count: int = Field(default=0, description="终态任务数")
    completed_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="completed / task_count")
    latency_p50_ms: float = Field(default=0.0, ge=0.0)
    latency_p95_ms: float = Field(default=0.0, ge=0.0)
    token_total_sum: int = Field(default=0, ge=0, description="估算 token 合计")
    token_mean: float = Field(default=0.0, ge=0.0, description="每任务平均估算 token")


def compute_service_metrics(samples: Sequence[TaskMetricSample]) -> ServiceMetrics:
    """由终态采样聚合指标。

    Args:
        samples: TaskService 记录的采样。

    Returns:
        ServiceMetrics；空样本时全 0。
    """
    if not samples:
        return ServiceMetrics()

    completed = sum(1 for s in samples if s.status == "completed")
    failed = sum(1 for s in samples if s.status == "failed")
    durations = [s.duration_ms for s in samples]
    token_sum = sum(s.token_total for s in samples)
    n = len(samples)
    return ServiceMetrics(
        task_count=n,
        completed_count=completed,
        failed_count=failed,
        success_rate=round(completed / n, 4),
        latency_p50_ms=round(percentile(durations, 50), 3),
        latency_p95_ms=round(percentile(durations, 95), 3),
        token_total_sum=token_sum,
        token_mean=round(token_sum / n, 4),
    )
