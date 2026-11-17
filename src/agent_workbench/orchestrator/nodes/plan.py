"""plan 节点：将任务结构化拆解为子任务列表。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from agent_workbench.orchestrator.prompts import build_plan_prompt
from agent_workbench.orchestrator.state import SubTask, SupervisorState

__all__ = ["PlanOutput", "create_plan_node"]

# 单次拆解上限（prompt 约定 1~8，此处再兜底）
_MAX_SUBTASKS = 8


class PlanOutput(BaseModel):
    """LLM 结构化输出 schema（plan 产物）。"""

    subtasks: list[SubTask] = Field(default_factory=list, description="子任务列表")


def _fallback_plan(task_description: str) -> list[SubTask]:
    """受控 fallback：整任务作为单个子任务，保证图可继续。"""
    return [SubTask(id="s1", description=task_description.strip() or "完成任务")]


def _normalize_subtasks(
    subtasks: list[SubTask],
    task_description: str,
) -> list[SubTask]:
    """规范化：去重 id、截断上限；空列表走 fallback。

    Args:
        subtasks: 模型产出的原始子任务。
        task_description: 原始任务（fallback 用）。

    Returns:
        可安全写入 state 的子任务列表（至少 1 个，id 唯一）。
    """
    cleaned: list[SubTask] = []
    seen: set[str] = set()
    for index, item in enumerate(subtasks, start=1):
        if len(cleaned) >= _MAX_SUBTASKS:
            break
        candidate = item.id.strip() or f"s{index}"
        while candidate in seen:
            candidate = f"{candidate}_{index}"
        seen.add(candidate)
        cleaned.append(item.model_copy(update={"id": candidate}))
    if not cleaned:
        return _fallback_plan(task_description)
    return cleaned


def create_plan_node(llm: BaseChatModel) -> Callable[[SupervisorState], dict[str, Any]]:
    """创建 plan 节点。

    结构化输出优先；解析失败 / 空结果走受控 fallback（单子任务），
    不向图外抛异常。

    Args:
        llm: 支持 with_structured_output 的聊天模型。

    Returns:
        接收 SupervisorState、返回 `{"plan": [...]}` 的节点函数。
    """
    structured = llm.with_structured_output(PlanOutput)

    def plan_node(state: SupervisorState) -> dict[str, Any]:
        """拆解任务并写回 plan 字段。"""
        task_description = state["task_description"]
        prompt = build_plan_prompt(task_description)
        try:
            raw = structured.invoke([HumanMessage(content=prompt)])
        except (OutputParserException, ValidationError, ValueError, TypeError, RuntimeError):
            raw = None

        subtasks: list[SubTask] = []
        if isinstance(raw, PlanOutput):
            subtasks = list(raw.subtasks)
        elif isinstance(raw, dict):
            try:
                subtasks = list(PlanOutput.model_validate(raw).subtasks)
            except (ValidationError, TypeError, ValueError):
                subtasks = []

        return {"plan": _normalize_subtasks(subtasks, task_description)}

    return plan_node
