"""评测 runner：执行数据集任务并产出 RunResult / AggregateMetrics。

默认离线替身可复现；编排模式 single|multi（#31 扩展 CLI 对比）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from agent_workbench.harness.dataset import EvalTask, load_dataset
from agent_workbench.harness.metrics import (
    AggregateMetrics,
    RunResult,
    default_judge,
    summarize_runs,
)
from agent_workbench.orchestrator.builder import build_supervisor_graph
from agent_workbench.orchestrator.single_agent import create_single_agent_graph, initial_state
from agent_workbench.orchestrator.state import initial_supervisor_state
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["Judge", "RunnerMode", "run_dataset", "run_one"]

RunnerMode = Literal["single", "multi"]

Judge = Callable[[EvalTask, RunResult], bool]


def run_one(
    task: EvalTask,
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    mode: RunnerMode = "multi",
    max_steps: int = 8,
    max_repair_attempts: int = 2,
    judge: Judge | None = None,
) -> RunResult:
    """执行单条评测任务。

    Args:
        task: 数据集条目。
        llm: 聊天模型（替身或真实）。
        registry: 工具注册表。
        mode: single=单 Agent；multi=Supervisor。
        max_steps: 工具轮次上限。
        max_repair_attempts: 回修上限（multi）。
        judge: 自定义通过判定；None 用 default_judge。

    Returns:
        RunResult（含 passed）。
    """
    judge_fn = judge or default_judge
    started = time.perf_counter()
    status = "failed"
    answer = ""
    degraded = False
    repair_count = 0
    step_count = 0
    error: str | None = None

    try:
        if mode == "single":
            single_graph = create_single_agent_graph(llm, registry, max_steps=max_steps)
            state = single_graph.invoke(initial_state(task.task, max_steps=max_steps))
            step_count = int(state.get("step_count", 0))
            answer = _extract_single(state)
        else:
            supervisor_graph = build_supervisor_graph(
                llm,
                registry,
                checkpointer=None,
                max_steps=max_steps,
            )
            state = supervisor_graph.invoke(
                initial_supervisor_state(
                    task.task,
                    max_repair_attempts=max_repair_attempts,
                )
            )
            answer = str(state.get("final_answer") or "")
            if not answer:
                answer = _extract_single(state)
            degraded = bool(state.get("degraded", False))
            repair_count = int(state.get("repair_count", 0))
            step_count = repair_count
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}"

    duration_ms = (time.perf_counter() - started) * 1000.0
    result = RunResult(
        task_id=task.id,
        should_fail=task.should_fail,
        status=status,
        answer=answer,
        degraded=degraded,
        repair_count=repair_count,
        step_count=step_count,
        duration_ms=duration_ms,
        error=error,
        passed=False,
    )
    result.passed = judge_fn(task, result)
    return result


def run_dataset(
    tasks: Sequence[EvalTask],
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    mode: RunnerMode = "multi",
    max_steps: int = 8,
    max_repair_attempts: int = 2,
    judge: Judge | None = None,
) -> tuple[list[RunResult], AggregateMetrics]:
    """顺序执行数据集并汇总指标。

    Args:
        tasks: 任务列表。
        llm: 聊天模型。
        registry: 工具注册表。
        mode: 编排模式。
        max_steps: 工具轮次上限。
        max_repair_attempts: 回修上限。
        judge: 自定义判定。

    Returns:
        (逐条 RunResult, AggregateMetrics)。
    """
    runs: list[RunResult] = []
    for task in tasks:
        runs.append(
            run_one(
                task,
                llm,
                registry,
                mode=mode,
                max_steps=max_steps,
                max_repair_attempts=max_repair_attempts,
                judge=judge,
            )
        )
    return runs, summarize_runs(runs)


def run_default_dataset(
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    dataset_path: Any = None,
    mode: RunnerMode = "multi",
    max_steps: int = 8,
    max_repair_attempts: int = 2,
    judge: Judge | None = None,
) -> tuple[list[RunResult], AggregateMetrics]:
    """加载默认/指定数据集并执行。"""
    tasks = load_dataset(dataset_path)
    return run_dataset(
        tasks,
        llm,
        registry,
        mode=mode,
        max_steps=max_steps,
        max_repair_attempts=max_repair_attempts,
        judge=judge,
    )


def _extract_single(state: dict[str, Any]) -> str:
    """从单 Agent 终态提取答案（finish JSON 或末条 AI 文本）。"""
    import json

    from langchain_core.messages import AIMessage, ToolMessage

    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str) and "final_delivery" in content:
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("answer"):
                    return str(payload["answer"])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return str(msg.content)
    return ""
