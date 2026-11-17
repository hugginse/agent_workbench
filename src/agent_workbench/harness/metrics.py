"""评测指标：通过率、稳定性、幻觉（引用支撑）、成本与延迟分位。

通过判定（默认 judge，可替换）：
- 普通任务（should_fail=false）：completed 且 answer 非空 → 通过；
- 应失败任务（should_fail=true）：failed，或 completed 且（拒答 / 降级 / 空答）→ 通过。

链路稳定性：
- step_success_rate：无异常跑完的比例（completed / total）；
- retry_rate：发生过回修（repair_count>0）的任务占比；
- mean_repairs：平均回修次数。

幻觉：
- citation_support_rate：答案中引用标记被合法命中支撑的比例；
- hallucination_rate = 1 − 聚合支撑率（无支撑样本时 0）；
- 拒答答案记支撑率 1.0（未编造引用）；无引用且非拒答 → 不计入支撑率分母（None）。

成本与延迟：
- token_* 为启发式估算（见 memory.context），非账单数字；
- latency p50/p95：对 duration_ms 排序后的分位（确定性）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from agent_workbench.harness.dataset import EvalTask

__all__ = [
    "AggregateMetrics",
    "RunResult",
    "citation_support_rate",
    "default_judge",
    "percentile",
    "summarize_runs",
]

_ABSTAIN_MARKERS = ("拒答", "无证据", "无匹配", "无法给出", "没有足以支持")
_REF_PATTERN = re.compile(r"\[(\d+)\]")


def percentile(values: Sequence[float], p: float) -> float:
    """确定性分位数（线性插值，p ∈ [0, 100]）。

    Args:
        values: 非空数值序列。
        p: 百分位 0~100。

    Returns:
        分位值。

    Raises:
        ValueError: values 为空或 p 越界。
    """
    if not values:
        msg = "percentile 需要非空序列"
        raise ValueError(msg)
    if not 0.0 <= p <= 100.0:
        msg = f"p 必须在 [0, 100]，当前 {p}"
        raise ValueError(msg)
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def citation_support_rate(
    answer: str,
    allowed_refs: Sequence[str] | None = None,
    *,
    allowed_chunk_ids: Sequence[str] | None = None,
) -> float | None:
    """计算单条答案的引用支撑率。

    Args:
        answer: 答案正文。
        allowed_refs: 合法引用标记（如 ["[1]", "[2]"]）；来自检索命中。
        allowed_chunk_ids: 合法 chunk_id 列表。

    Returns:
        [0, 1] 支撑率；None 表示本条不纳入支撑率统计
        （无引用标记且非拒答，或无合法集合可对照时由调用方决定）。
    """
    text = answer or ""
    found_refs = _REF_PATTERN.findall(text)
    found_ids = [cid for cid in (allowed_chunk_ids or []) if cid and cid in text]
    allowed_ref_set = set(allowed_refs or [])
    allowed_id_set = set(allowed_chunk_ids or [])

    # 无任何引用标记
    if not found_refs and not found_ids:
        stripped = text.strip()
        if any(m in stripped for m in _ABSTAIN_MARKERS):
            return 1.0
        return None

    supported = 0
    total = 0
    for n in found_refs:
        ref = f"[{n}]"
        total += 1
        if ref in allowed_ref_set:
            supported += 1
        elif not allowed_ref_set:
            # 无合法集合：有引用但无法核验 → 视为未支撑
            pass
    for cid in found_ids:
        total += 1
        if cid in allowed_id_set:
            supported += 1

    if total == 0:
        return 1.0 if any(m in text for m in _ABSTAIN_MARKERS) else None
    return round(supported / total, 4)


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
    token_input: int = Field(default=0, description="估算输入 token")
    token_output: int = Field(default=0, description="估算输出 token")
    token_total: int = Field(default=0, description="估算总 token")
    citation_support_rate: float | None = Field(
        default=None,
        description="引用支撑率；None=不纳入幻觉统计",
    )


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
    # 成本
    token_input_sum: int = Field(default=0, description="输入 token 合计（估算）")
    token_output_sum: int = Field(default=0, description="输出 token 合计（估算）")
    token_total_sum: int = Field(default=0, description="总 token 合计（估算）")
    mean_token_total: float = Field(default=0.0, description="每任务平均总 token")
    # 延迟
    latency_p50_ms: float = Field(default=0.0, description="延迟 p50")
    latency_p95_ms: float = Field(default=0.0, description="延迟 p95")
    # 幻觉
    citation_support_rate: float | None = Field(
        default=None,
        description="聚合引用支撑率（仅统计非 None 样本）",
    )
    hallucination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="幻觉率 = 1 − 支撑率；无样本时为 0",
    )
    citation_sample_count: int = Field(
        default=0,
        description="纳入支撑率统计的样本数",
    )


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

    token_in = sum(r.token_input for r in runs)
    token_out = sum(r.token_output for r in runs)
    token_total = sum(r.token_total for r in runs)
    durations = [r.duration_ms for r in runs]
    p50 = round(percentile(durations, 50), 3)
    p95 = round(percentile(durations, 95), 3)

    support_samples = [r.citation_support_rate for r in runs if r.citation_support_rate is not None]
    if support_samples:
        support = round(sum(support_samples) / len(support_samples), 4)
        hallucination = round(1.0 - support, 4)
    else:
        support = None
        hallucination = 0.0

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
        token_input_sum=token_in,
        token_output_sum=token_out,
        token_total_sum=token_total,
        mean_token_total=round(token_total / len(runs), 4),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        citation_support_rate=support,
        hallucination_rate=hallucination,
        citation_sample_count=len(support_samples),
    )
