"""单 Agent ReAct 循环：工具调用、计数封顶与终止路径测试。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.orchestrator import create_single_agent_graph
from agent_workbench.orchestrator.single_agent import initial_state
from agent_workbench.tools import ToolRegistry
from agent_workbench.tools.builtins import FinishTool


def _registry_with_echo() -> ToolRegistry:
    """注册 finish 的最小注册表（参数简单、无副作用）。"""
    r = ToolRegistry()
    r.register(FinishTool())
    return r


def _tool_call_message(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """构造携带单个 tool_call 的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def _final_state(graph: Any, initial: dict[str, Any]) -> dict[str, Any]:
    """invoke 并返回终态 dict。"""
    result = graph.invoke(initial)
    assert isinstance(result, dict)
    return result


class TestTerminationPaths:
    """三条终止路径：直接完成 / 工具后完成 / 超限终止。"""

    def test_no_tool_call_ends_immediately(self) -> None:
        """模型不调用工具 → 立即 END，step_count 保持 0。"""
        llm = ScriptedChatModel(scripted_responses=["最终答案"])
        graph = create_single_agent_graph(llm, _registry_with_echo(), max_steps=5)
        state = _final_state(graph, initial_state("任务", max_steps=5))

        assert state["step_count"] == 0
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        assert last.content == "最终答案"
        assert not last.tool_calls

    def test_single_tool_then_final(self) -> None:
        """一次工具调用后给出最终答案 → 正常 END，step_count == 1。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _tool_call_message("finish", {"answer": "交付"}, "c1"),
                "完成",
            ]
        )
        graph = create_single_agent_graph(llm, _registry_with_echo(), max_steps=5)
        state = _final_state(graph, initial_state("任务", max_steps=5))

        assert state["step_count"] == 1
        # 存在工具结果消息
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "c1"
        assert "final_delivery" in tool_msgs[0].content
        # 最终消息为无 tool_calls 的 AI 消息
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        assert not last.tool_calls

    def test_bounded_loop_terminates_at_max_steps(self) -> None:
        """模型持续请求工具 → step_count 封顶为 max_steps，图必须结束。"""
        max_steps = 3
        # 永远要调用工具（循环复用同一条 tool_call 消息）
        always_tool = _tool_call_message("finish", {"answer": "x"}, "c_loop")
        llm = ScriptedChatModel(scripted_responses=[always_tool])
        graph = create_single_agent_graph(llm, _registry_with_echo(), max_steps=max_steps)
        state = _final_state(graph, initial_state("任务", max_steps=max_steps))

        assert state["step_count"] == max_steps
        # 终止时最后一条 AI 消息仍可能带 tool_calls，但图已 END（不再执行 tools）
        last_ai = None
        for m in state["messages"]:
            if isinstance(m, AIMessage):
                last_ai = m
        assert last_ai is not None
        assert last_ai.tool_calls, "超限终止前模型仍在请求工具"
        # 工具执行次数 = max_steps（每轮一条 ToolMessage）
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == max_steps

    def test_max_steps_must_be_positive(self) -> None:
        """max_steps < 1 必须在构造期拒绝。"""
        llm = ScriptedChatModel(scripted_responses=["x"])
        with pytest.raises(ValueError, match="max_steps"):
            create_single_agent_graph(llm, _registry_with_echo(), max_steps=0)


class TestToolExecution:
    """工具执行与错误回传。"""

    def test_unknown_tool_error_returned_to_model(self) -> None:
        """未注册工具 → ToolMessage 携带错误文本，循环不崩溃。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _tool_call_message("no_such_tool", {}, "c_bad"),
                "已处理错误",
            ]
        )
        graph = create_single_agent_graph(llm, _registry_with_echo(), max_steps=5)
        state = _final_state(graph, initial_state("任务", max_steps=5))

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "工具错误" in tool_msgs[0].content
        assert "未注册" in tool_msgs[0].content
        assert state["step_count"] == 1

    def test_invalid_tool_args_error_returned(self) -> None:
        """入参不合法 → ToolMessage 错误反馈，不抛穿图。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _tool_call_message("finish", {"answer": ""}, "c_args"),
                "修正后完成",
            ]
        )
        graph = create_single_agent_graph(llm, _registry_with_echo(), max_steps=5)
        state = _final_state(graph, initial_state("任务", max_steps=5))

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert "工具错误" in tool_msgs[0].content
        assert "校验失败" in tool_msgs[0].content


class TestGraphContract:
    """图结构契约：节点与显式路由存在。"""

    def test_graph_compiles_with_nodes(self) -> None:
        """编译图包含 agent/tools 节点。"""
        llm = ScriptedChatModel(scripted_responses=["ok"])
        graph = create_single_agent_graph(llm, _registry_with_echo(), max_steps=2)
        # get_graph 可渲染说明节点/边合法
        rendered = graph.get_graph()
        node_names = {n for n in rendered.nodes}
        assert "agent" in node_names
        assert "tools" in node_names
