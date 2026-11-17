"""execute 节点：按 assign 结果派发 SubAgent 执行子任务。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_workbench.agents import SubAgentSpec, get_subagent
from agent_workbench.orchestrator.prompts import build_execute_prompt
from agent_workbench.orchestrator.state import SubTask, SupervisorState, default_max_steps
from agent_workbench.tools.base import ToolError, ToolSpec
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["create_execute_node", "run_subagent"]

# 子任务内工具循环上限（与单 Agent 默认一致）
_DEFAULT_SUBAGENT_STEPS = default_max_steps


def _bound_tools(spec: SubAgentSpec, registry: ToolRegistry) -> list[dict[str, Any]]:
    """按 SubAgent 声明过滤并导出可绑定工具（OpenAI function 格式）。"""
    bound: list[dict[str, Any]] = []
    for name in spec.tool_names:
        if name not in registry:
            continue
        tool_spec: ToolSpec = registry.get(name).spec()
        bound.append(
            {
                "type": "function",
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.description,
                    "parameters": tool_spec.parameters,
                },
            }
        )
    return bound


def run_subagent(
    spec: SubAgentSpec,
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    subtask: SubTask,
    task_description: str,
    feedback: str | None = None,
    max_steps: int = _DEFAULT_SUBAGENT_STEPS,
) -> str:
    """在步数封顶内运行单个 SubAgent，返回结果正文。

    Args:
        spec: SubAgent 声明。
        llm: 聊天模型。
        registry: 工具注册表。
        subtask: 当前子任务。
        task_description: 总任务描述（上下文）。
        feedback: 回修反馈；首次执行为 None。
        max_steps: 子任务内工具循环上限，必须 >= 1。

    Returns:
        结果正文（超限或失败时为可读说明，不抛穿图）。
    """
    if max_steps < 1:
        msg = f"max_steps 必须 >= 1，当前 {max_steps}"
        raise ValueError(msg)

    tools = _bound_tools(spec, registry)
    bound = llm.bind_tools(tools)
    messages: list[Any] = [
        SystemMessage(content=spec.system_prompt),
        HumanMessage(content=build_execute_prompt(task_description, subtask, feedback)),
    ]

    for _ in range(max_steps):
        ai = bound.invoke(messages)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            content = str(getattr(ai, "content", "") or "")
            return content.strip() or "（子任务无输出）"
        messages.append(ai)
        for call in tool_calls:
            call_id = str(call.get("id", ""))
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            try:
                output = registry.execute(name, args)
            except ToolError as exc:
                output = f"工具错误: {exc}"
            messages.append(ToolMessage(content=output, tool_call_id=call_id))

    return f"子任务执行达到步数上限（{max_steps}），未产出最终结论"


def create_execute_node(
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    max_steps: int = _DEFAULT_SUBAGENT_STEPS,
) -> Callable[[SupervisorState], dict[str, Any]]:
    """创建 execute 节点。

    调度规则：
    - 首轮：执行尚无结果的子任务；
    - 回修（verdict.passed=false 且有 target）：仅重跑目标子任务；
    - 已有结果且非回修目标：跳过。

    Args:
        llm: 聊天模型。
        registry: 工具注册表。
        max_steps: 每个 SubAgent 的工具循环上限。

    Returns:
        接收 SupervisorState、返回 `{"results": ...}` 的节点函数。
    """
    if max_steps < 1:
        msg = f"max_steps 必须 >= 1，当前 {max_steps}"
        raise ValueError(msg)

    def execute_node(state: SupervisorState) -> dict[str, Any]:
        """派发并执行需要运行的子任务，合并写回 results。"""
        results = dict(state["results"])
        verdict = state["verdict"]
        repair_target: str | None = None
        if verdict is not None and not verdict.passed:
            repair_target = verdict.target_subtask_id

        for subtask in state["plan"]:
            if repair_target is not None:
                if subtask.id != repair_target:
                    continue
            elif subtask.id in results:
                continue

            agent_name = subtask.assignee or "executor"
            try:
                spec = get_subagent(agent_name)
            except KeyError:
                spec = get_subagent("executor")

            feedback = verdict.feedback if (repair_target == subtask.id and verdict) else None
            content = run_subagent(
                spec,
                llm,
                registry,
                subtask=subtask,
                task_description=state["task_description"],
                feedback=feedback,
                max_steps=max_steps,
            )
            results[subtask.id] = content

        return {"results": results}

    return execute_node
