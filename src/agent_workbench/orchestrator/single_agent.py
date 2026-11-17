"""单 Agent ReAct 图：工具循环计数封顶 + 显式终止路径。

流程：
    START → agent → (有工具调用且未超限) → tools → agent
                  → (无工具调用 或 step_count >= max_steps) → END
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_workbench.orchestrator.state import SingleAgentState, default_max_steps
from agent_workbench.tools.base import ToolError, ToolSpec
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["create_single_agent_graph", "initial_state"]

ContinueTarget = Literal["tools", "end"]


def _to_bound_tool(spec: ToolSpec) -> dict[str, Any]:
    """将 ToolSpec 转为 bind_tools 可用的 OpenAI function 格式。"""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def create_single_agent_graph(
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    max_steps: int = default_max_steps,
) -> CompiledStateGraph[SingleAgentState, None, Any, Any]:
    """编译单 Agent ReAct 图。

    Args:
        llm: 聊天模型（真实或替身）；需支持 bind_tools。
        registry: 已注册可用工具。
        max_steps: 工具轮次硬上限，必须 >= 1。

    Returns:
        可 `invoke` 的编译图。

    Raises:
        ValueError: max_steps 非法。
    """
    if max_steps < 1:
        msg = f"max_steps 必须 >= 1，当前 {max_steps}"
        raise ValueError(msg)

    bound_tools = [_to_bound_tool(s) for s in registry.specs()]

    def agent_node(state: SingleAgentState) -> dict[str, Any]:
        """调用 LLM 生成下一条消息（可能携带 tool_calls）。"""
        bound = llm.bind_tools(bound_tools)
        response = bound.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(state: SingleAgentState) -> dict[str, Any]:
        """执行上一条 AIMessage 中的全部 tool_calls，步数 +1。"""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        results: list[ToolMessage] = []
        for call in tool_calls:
            call_id = str(call.get("id", ""))
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            try:
                output = registry.execute(name, args)
            except ToolError as exc:
                # 工具错误回传给模型，形成受控反馈而非崩溃
                output = f"工具错误: {exc}"
            results.append(ToolMessage(content=output, tool_call_id=call_id))
        return {
            "messages": results,
            "step_count": state["step_count"] + 1,
        }

    def should_continue(state: SingleAgentState) -> ContinueTarget:
        """条件边：无工具调用 → 结束；超限 → 结束；否则继续 tools。"""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "end"
        if state["step_count"] >= state["max_steps"]:
            # 显式终止路径：达到上限不再进入 tools
            return "end"
        return "tools"

    builder: StateGraph[SingleAgentState] = StateGraph(SingleAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def initial_state(user_input: str, *, max_steps: int = default_max_steps) -> dict[str, Any]:
    """构造图的初始状态。

    Args:
        user_input: 用户任务文本。
        max_steps: 工具轮次上限。

    Returns:
        SingleAgentState 初始字典。
    """
    return {
        "messages": [HumanMessage(content=user_input)],
        "step_count": 0,
        "max_steps": max_steps,
    }
