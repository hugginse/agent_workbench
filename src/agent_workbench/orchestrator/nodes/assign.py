"""assign 节点：为子任务分派 SubAgent 角色。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_workbench.agents import SUBAGENTS
from agent_workbench.orchestrator.state import SubTask, SupervisorState

__all__ = ["create_assign_node", "resolve_assignee"]

# 描述关键词 → 角色（未显式指定时的启发式）
_RESEARCH_HINTS = ("调研", "研究", "检索", "收集", "整理资料", "查找")
_REVIEW_HINTS = ("审查", "评审", "检查", "核对", "校验")


def resolve_assignee(subtask: SubTask) -> str:
    """解析子任务应由哪个 SubAgent 执行。

    优先保留已声明且合法的 assignee；否则按描述关键词启发式；
    默认 executor。

    Args:
        subtask: 子任务。

    Returns:
        角色名（researcher / executor / reviewer 之一）。
    """
    if subtask.assignee in SUBAGENTS:
        return subtask.assignee
    text = subtask.description
    if any(hint in text for hint in _REVIEW_HINTS):
        return "reviewer"
    if any(hint in text for hint in _RESEARCH_HINTS):
        return "researcher"
    return "executor"


def create_assign_node() -> Callable[[SupervisorState], dict[str, Any]]:
    """创建 assign 节点。

    返回：
        更新 plan 中各子任务 assignee 的节点函数。
    """

    def assign_node(state: SupervisorState) -> dict[str, Any]:
        """为 plan 内所有子任务写入 assignee（回修时同样覆盖目标）。"""
        assigned: list[SubTask] = [
            st.model_copy(update={"assignee": resolve_assignee(st)}) for st in state["plan"]
        ]
        return {"plan": assigned}

    return assign_node
