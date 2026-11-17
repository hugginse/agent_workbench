"""single vs multi（Supervisor）A/B 对比。

同一数据集、同一替身语义下各跑一遍，输出指标差值与 Markdown 表。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_workbench.harness.dataset import EvalTask
from agent_workbench.harness.metrics import AggregateMetrics, RunResult
from agent_workbench.harness.runner import Judge, run_dataset
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["ComparisonReport", "ModeComparison", "compare_modes", "format_comparison_markdown"]

# 对比表关注的指标字段（multi − single）
_DELTA_FIELDS: tuple[str, ...] = (
    "task_success_rate",
    "step_success_rate",
    "retry_rate",
    "mean_repairs",
    "mean_token_total",
    "token_total_sum",
    "latency_p50_ms",
    "latency_p95_ms",
    "hallucination_rate",
)

_LABELS: dict[str, str] = {
    "task_success_rate": "任务通过率",
    "step_success_rate": "逐步成功率",
    "retry_rate": "回修率",
    "mean_repairs": "平均回修",
    "mean_token_total": "平均 token",
    "token_total_sum": "token 合计",
    "latency_p50_ms": "延迟 p50 (ms)",
    "latency_p95_ms": "延迟 p95 (ms)",
    "hallucination_rate": "幻觉率",
}


class ModeComparison(BaseModel):
    """双模式聚合指标。"""

    single: AggregateMetrics
    multi: AggregateMetrics


class ComparisonReport(ModeComparison):
    """对比报告：双模式指标 + 差值（multi − single）。"""

    deltas: dict[str, float] = Field(
        default_factory=dict,
        description="关键指标 multi − single",
    )
    single_runs: list[RunResult] = Field(default_factory=list)
    multi_runs: list[RunResult] = Field(default_factory=list)


def _deltas(single: AggregateMetrics, multi: AggregateMetrics) -> dict[str, float]:
    """计算 multi − single 差值。"""
    s = single.model_dump()
    m = multi.model_dump()
    out: dict[str, float] = {}
    for key in _DELTA_FIELDS:
        sv = float(s.get(key) or 0.0)
        mv = float(m.get(key) or 0.0)
        out[key] = round(mv - sv, 4)
    return out


def compare_modes(
    tasks: Sequence[EvalTask],
    llm_factory: Callable[[], BaseChatModel],
    registry: ToolRegistry,
    *,
    max_steps: int = 8,
    max_repair_attempts: int = 2,
    judge: Judge | None = None,
) -> ComparisonReport:
    """在同一数据集上分别以 single / multi 执行并对比。

    Args:
        tasks: 评测任务。
        llm_factory: 每次调用返回新的模型实例（保证两模式起点一致、互不污染游标）。
        registry: 工具注册表（两模式共用结构）。
        max_steps: 工具轮次上限。
        max_repair_attempts: 回修上限（multi）。
        judge: 自定义判定。

    Returns:
        ComparisonReport。
    """
    single_runs, single_metrics = run_dataset(
        tasks,
        llm_factory(),
        registry,
        mode="single",
        max_steps=max_steps,
        max_repair_attempts=max_repair_attempts,
        judge=judge,
    )
    multi_runs, multi_metrics = run_dataset(
        tasks,
        llm_factory(),
        registry,
        mode="multi",
        max_steps=max_steps,
        max_repair_attempts=max_repair_attempts,
        judge=judge,
    )
    return ComparisonReport(
        single=single_metrics,
        multi=multi_metrics,
        deltas=_deltas(single_metrics, multi_metrics),
        single_runs=single_runs,
        multi_runs=multi_runs,
    )


def format_comparison_markdown(report: ComparisonReport) -> str:
    """渲染 Markdown 对比表（供 README / 终端阅读）。

    Args:
        report: 对比报告。

    Returns:
        Markdown 文本。
    """
    lines = [
        "| 指标 | single | multi | Δ (multi−single) |",
        "|---|---:|---:|---:|",
    ]
    s = report.single.model_dump()
    m = report.multi.model_dump()
    for key, label in _LABELS.items():
        sv = s.get(key)
        mv = m.get(key)
        dv = report.deltas.get(key, 0.0)
        if sv is None or mv is None:
            lines.append(f"| {label} | n/a | n/a | n/a |")
            continue
        if isinstance(sv, float) and abs(sv) < 10 and key.endswith("_rate"):
            lines.append(f"| {label} | {sv:.4f} | {mv:.4f} | {dv:+.4f} |")
        else:
            lines.append(f"| {label} | {sv} | {mv} | {dv:+} |")
    return "\n".join(lines)


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """序列化报告（含 runs）。"""
    return report.model_dump()
