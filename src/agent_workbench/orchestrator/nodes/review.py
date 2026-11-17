"""review 节点：结构化评审 + 三路径条件边（通过 / 回修 / 超限降级）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent_workbench.orchestrator.prompts import build_review_prompt
from agent_workbench.orchestrator.state import ReviewVerdict, SupervisorState

__all__ = ["ReviewTarget", "create_review_node", "create_review_router"]

# 条件边目标节点名（与 builder 注册一致）
ReviewTarget = Literal["assign", "aggregate"]


def _fallback_verdict(state: SupervisorState) -> ReviewVerdict:
    """解析失败时的保守 fallback：判失败并指定首个子任务回修。

    无子任务时 target 为 None，由条件边按 fail+无目标走降级，避免静默通过。
    """
    target = state["plan"][0].id if state["plan"] else None
    return ReviewVerdict(
        passed=False,
        feedback="评审输出解析失败，按未通过处理",
        target_subtask_id=target,
    )


def _sanitize_verdict(verdict: ReviewVerdict, state: SupervisorState) -> ReviewVerdict:
    """校验 target_subtask_id 必须存在于 plan；失败且无合法目标时补首个子任务。"""
    plan_ids = [st.id for st in state["plan"]]
    if verdict.passed:
        return verdict.model_copy(update={"target_subtask_id": None})
    if verdict.target_subtask_id in plan_ids:
        return verdict
    # 失败但目标非法/缺失 → 指向首个子任务（保证回修可执行）
    fallback_target = plan_ids[0] if plan_ids else None
    return verdict.model_copy(update={"target_subtask_id": fallback_target})


def create_review_node(llm: BaseChatModel) -> Callable[[SupervisorState], dict[str, Any]]:
    """创建 review 节点。

    结构化输出 ReviewVerdict；解析失败走保守 fallback（fail）。
    节点只写 verdict，不递增 repair_count（计数在回修入口 assign）。

    Args:
        llm: 支持 with_structured_output 的聊天模型。

    Returns:
        接收 SupervisorState、返回 `{"verdict": ...}` 的节点函数。
    """
    structured = llm.with_structured_output(ReviewVerdict)

    def review_node(state: SupervisorState) -> dict[str, Any]:
        """产出评审结论并写回 verdict。"""
        prompt = build_review_prompt(state["task_description"], state["results"])
        try:
            raw = structured.invoke([HumanMessage(content=prompt)])
        except (OutputParserException, ValidationError, ValueError, TypeError, RuntimeError):
            raw = None

        if isinstance(raw, ReviewVerdict):
            verdict = raw
        elif isinstance(raw, dict):
            try:
                verdict = ReviewVerdict.model_validate(raw)
            except (ValidationError, TypeError, ValueError):
                verdict = _fallback_verdict(state)
        else:
            verdict = _fallback_verdict(state)

        return {"verdict": _sanitize_verdict(verdict, state)}

    return review_node


def create_review_router() -> Callable[[SupervisorState], ReviewTarget]:
    """创建 review 后条件边（契约三路径）。

    - pass → aggregate
    - fail & repair_count < max → assign（定向回修）
    - fail & repair_count >= max → aggregate（降级交付）

    Returns:
        返回 `"assign"` 或 `"aggregate"` 的路由函数。
    """

    def route_after_review(state: SupervisorState) -> ReviewTarget:
        """按 verdict 与回修计数选择下一节点。"""
        verdict = state["verdict"]
        if verdict is None or verdict.passed:
            return "aggregate"
        if int(state["repair_count"]) >= int(state["max_repair_attempts"]):
            return "aggregate"
        return "assign"

    return route_after_review
