"""execute 节点：按 assign 结果派发 SubAgent 执行子任务。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agent_workbench.agents import SubAgentSpec, get_subagent
from agent_workbench.memory.context import (
    DEFAULT_MAX_CONTEXT_MESSAGES,
    estimate_messages_tokens,
    merge_token_usage,
    prune_messages,
)
from agent_workbench.orchestrator.prompts import build_execute_prompt
from agent_workbench.orchestrator.state import SubTask, SupervisorState, default_max_steps
from agent_workbench.tools.base import ToolError, ToolSpec
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["SubAgentOutcome", "create_execute_node", "run_subagent"]

# 子任务内工具循环上限（与单 Agent 默认一致）
_DEFAULT_SUBAGENT_STEPS = default_max_steps


class SubAgentOutcome(BaseModel):
    """SubAgent 一次执行的结果与 token 用量。"""

    content: str = Field(description="结果正文")
    input_tokens: int = Field(default=0, description="近似输入 token 合计")
    output_tokens: int = Field(default=0, description="近似输出 token 合计")


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
    max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
) -> SubAgentOutcome:
    """在步数封顶内运行单个 SubAgent；送模型前裁剪上下文并累计 token。

    Args:
        spec: SubAgent 声明。
        llm: 聊天模型。
        registry: 工具注册表。
        subtask: 当前子任务。
        task_description: 总任务描述（上下文）。
        feedback: 回修反馈；首次执行为 None。
        max_steps: 子任务内工具循环上限，必须 >= 1。
        max_context_messages: 每次 invoke 前的消息条数上限。

    Returns:
        结果正文与近似 token 用量（超限/失败时正文为可读说明，不抛穿图）。
    """
    if max_steps < 1:
        msg = f"max_steps 必须 >= 1，当前 {max_steps}"
        raise ValueError(msg)

    tools = _bound_tools(spec, registry)
    bound = llm.bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(content=spec.system_prompt),
        HumanMessage(content=build_execute_prompt(task_description, subtask, feedback)),
    ]

    input_tokens = 0
    output_tokens = 0

    for _ in range(max_steps):
        context = prune_messages(messages, max_messages=max_context_messages)
        input_tokens += estimate_messages_tokens(context)
        ai = bound.invoke(context)
        output_tokens += estimate_messages_tokens([ai])
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            content = str(getattr(ai, "content", "") or "")
            return SubAgentOutcome(
                content=content.strip() or "（子任务无输出）",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
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

    return SubAgentOutcome(
        content=f"子任务执行达到步数上限（{max_steps}），未产出最终结论",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


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
        接收 SupervisorState、返回 results 与累计 token_usage 的节点函数。
    """
    if max_steps < 1:
        msg = f"max_steps 必须 >= 1，当前 {max_steps}"
        raise ValueError(msg)

    def execute_node(state: SupervisorState) -> dict[str, Any]:
        """派发并执行需要运行的子任务，合并 results 与 token_usage。"""
        results = dict(state["results"])
        usage = dict(state.get("token_usage") or {"input": 0, "output": 0, "total": 0})
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
            outcome = run_subagent(
                spec,
                llm,
                registry,
                subtask=subtask,
                task_description=state["task_description"],
                feedback=feedback,
                max_steps=max_steps,
            )
            results[subtask.id] = outcome.content
            usage = merge_token_usage(
                usage,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
            )

        return {"results": results, "token_usage": usage}

    return execute_node
