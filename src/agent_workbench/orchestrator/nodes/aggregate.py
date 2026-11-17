"""aggregate 节点：聚合交付（正常 / 降级）并作为完成终点。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agent_workbench.memory.context import estimate_tokens, merge_token_usage
from agent_workbench.orchestrator.prompts import build_aggregate_prompt
from agent_workbench.orchestrator.state import SupervisorState

__all__ = ["create_aggregate_node", "is_degraded_delivery", "join_results_deterministic"]

_DEGRADED_PREFIX = "【降级交付】回修已达上限，以下内容未经评审通过，仅供参考：\n\n"
_DEGRADED_NO_REVIEW = "【降级交付】缺少有效评审结论，以下内容仅供参考：\n\n"


def is_degraded_delivery(state: SupervisorState) -> bool:
    """判断本次聚合是否走降级路径。

    降级条件：verdict 缺失，或 verdict.passed 为 False
    （能走到 aggregate 的 fail 即已超限）。

    Args:
        state: 当前 Supervisor 状态。

    Returns:
        True 表示降级交付。
    """
    verdict = state["verdict"]
    if verdict is None:
        return True
    return not verdict.passed


def join_results_deterministic(state: SupervisorState) -> str:
    """按 plan 顺序拼接 results（LLM 失败时的受控 fallback）。

    Args:
        state: 含 plan / results 的状态。

    Returns:
        Markdown 风格的拼接正文；无产出时为占位说明。
    """
    if not state["plan"] and not state["results"]:
        return "（无子任务产出）"
    parts: list[str] = []
    for st in state["plan"]:
        content = state["results"].get(st.id, "（未产出）")
        parts.append(f"## {st.id} · {st.description}\n\n{content}".rstrip())
    # 兜底：results 中有但 plan 缺失的 id
    planned = {st.id for st in state["plan"]}
    for sid, content in state["results"].items():
        if sid not in planned:
            parts.append(f"## {sid}\n\n{content}".rstrip())
    return "\n\n".join(parts) if parts else "（无子任务产出）"


def create_aggregate_node(
    llm: BaseChatModel | None = None,
) -> Callable[[SupervisorState], dict[str, Any]]:
    """创建 aggregate 节点（图的完成终点，之后接 END）。

    - 正常路径：LLM 汇总，失败则确定性拼接；
    - 降级路径：拼接正文并加前缀说明，degraded=True。

    Args:
        llm: 可选聊天模型；None 时仅用确定性拼接。

    Returns:
        接收 SupervisorState、返回 final_answer/degraded 的节点函数。
    """

    def aggregate_node(state: SupervisorState) -> dict[str, Any]:
        """生成最终交付并标记是否降级；累计本节点 token_usage。"""
        degraded = is_degraded_delivery(state)
        body = join_results_deterministic(state)
        input_tokens = 0
        output_tokens = 0

        if llm is not None and not degraded:
            prompt = build_aggregate_prompt(state["task_description"], state["results"])
            input_tokens = estimate_tokens(prompt)
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                text = str(getattr(response, "content", "") or "").strip()
                if text:
                    body = text
                output_tokens = estimate_tokens(text)
            except (ValueError, TypeError, RuntimeError, OSError):
                # 汇总失败不阻断交付，保留确定性拼接
                body = join_results_deterministic(state)

        if degraded:
            verdict = state["verdict"]
            if verdict is None:
                final = _DEGRADED_NO_REVIEW + body
            else:
                notice = (
                    f"回修已达上限（{state['repair_count']}/{state['max_repair_attempts']}），"
                    f"评审意见：{verdict.feedback or '无'}"
                )
                final = f"【降级交付】{notice}，以下内容未经评审通过，仅供参考：\n\n{body}"
        else:
            final = body

        return {
            "final_answer": final,
            "degraded": degraded,
            "token_usage": merge_token_usage(
                state.get("token_usage"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        }

    return aggregate_node
