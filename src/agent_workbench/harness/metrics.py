"""评测指标：任务通过率与链路稳定性（Phase 5 #29）。

通过判定（默认 judge，可替换）：
- 普通任务（should_fail=false）：completed 且 answer 非空 → 通过；
- 应失败任务（should_fail=true）：failed，或 completed 且（拒答 / 降级 / 空答）→ 通过。

链路稳定性：
- step_success_rate：无异常跑完的比例（completed / total）；
- retry_rate：发生过回修（repair_count>0）的任务占比；
- mean_repairs：平均回修次数。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_workbench.harness.dataset import EvalTask

__all__ = [
    "AggregateMetrics",
    "RunResult",
    "default_judge",
    "summarize_runs",
]

_ABSTAIN_MARKERS = ("拒答", "无证据", "无匹配", "无法给出")


class RunResult(BaseModel):
    """单条任务一次运行的结果（runner 写入）。"""

    task_id: str = Field(description="评测任务 id")
    should_fail: bool = Field(description="是否应失败样本")
    status: str = Field(description="completed | failed")
    answer: str = Field(default="", description="最终答案正文")
    degraded: bool = Field(default=False, description="是否降级交付")
    repair_count: int = Field(default=0, description="回修次数")
    step_count: int = Field(default=0, description="步数/工具轮次")
    duration_ms: float = Field(default=0.0, description="耗时毫秒")
    error: str | None = Field(default=None, description="失败原因（用户可见级）")
    passed: bool = Field(default=False, description="是否通过默认/自定义 judge")


class AggregateMetrics(BaseModel):
    """数据集聚合指标。"""

    task_count: int
    pass_count: int
    task_success_rate: float = Field(ge=0.0, le=1.0, description="judge 通过率")
    step_success_rate: float = Field(
        ge=0.0, le=1.0, description="无异常跑完比例（completed/total）"
    )
    retry_rate: float = Field(ge=0.0, le=1.0, description="发生回修的任务占比")
    mean_repairs: float = Field(ge=0.0, description="平均回修次数")
    normal_count: int = Field(default=0, description="非 should_fail 条数")
    normal_pass_count: int = Field(default=0, description="普通任务通过数")
    should_fail_count: int = Field(default=0, description="应失败样本条数")
    should_fail_pass_count: int = Field(default=0, description="应失败样本中优雅处理数")


def default_judge(task: EvalTask, result: RunResult) -> bool:
    """默认通过判定（见模块 docstring）。

    Args:
        task: 评测任务定义。
        result: 运行结果。

    Returns:
        是否通过。
    """
    if task.should_fail:
        if result.status == "failed":
            return True
        if result.status == "completed":
            answer = result.answer.strip()
            if result.degraded or not answer:
                return True
            return any(marker in answer for marker in _ABSTAIN_MARKERS)
        return False

    if result.status != "completed":
        return False
    return bool(result.answer.strip())


def summarize_runs(runs: list[RunResult]) -> AggregateMetrics:
    """汇总已写入 `passed` 的运行结果为聚合指标。

    Args:
        runs: runner 产出的逐条结果（judge 已在 run_one 内完成）。

    Returns:
        AggregateMetrics。
    """
    if not runs:
        return AggregateMetrics(
            task_count=0,
            pass_count=0,
            task_success_rate=0.0,
            step_success_rate=0.0,
            retry_rate=0.0,
            mean_repairs=0.0,
        )

    pass_count = sum(1 for r in runs if r.passed)
    completed = sum(1 for r in runs if r.status == "completed")
    retried = sum(1 for r in runs if r.repair_count > 0)
    total_repairs = sum(r.repair_count for r in runs)
    normal = [r for r in runs if not r.should_fail]
    should_fail = [r for r in runs if r.should_fail]

    return AggregateMetrics(
        task_count=len(runs),
        pass_count=pass_count,
        task_success_rate=round(pass_count / len(runs), 4),
        step_success_rate=round(completed / len(runs), 4),
        retry_rate=round(retried / len(runs), 4),
        mean_repairs=round(total_repairs / len(runs), 4),
        normal_count=len(normal),
        normal_pass_count=sum(1 for r in normal if r.passed),
        should_fail_count=len(should_fail),
        should_fail_pass_count=sum(1 for r in should_fail if r.passed),
    )
