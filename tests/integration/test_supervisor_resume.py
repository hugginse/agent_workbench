"""Supervisor 图装配与 SQLite 断点续跑集成测试。"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.orchestrator.builder import (
    build_supervisor_graph,
    open_sqlite_checkpointer,
)
from agent_workbench.orchestrator.state import initial_supervisor_state
from agent_workbench.tools.builtins import FinishTool, PythonSandboxTool
from agent_workbench.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(FinishTool())
    r.register(PythonSandboxTool())
    return r


def _passing_llm() -> ScriptedChatModel:
    """按调用顺序：plan → execute(×n) → review(pass) → aggregate。

    使用同一脚本序列：结构化 plan、执行文本、通过评审、汇总文本。
    """
    plan_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "PlanOutput",
                "args": {"subtasks": [{"id": "s1", "description": "生成一句总结"}]},
                "id": "c_plan",
            }
        ],
    )
    review_pass = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ReviewVerdict",
                "args": {"passed": True, "feedback": "合格", "target_subtask_id": None},
                "id": "c_rev",
            }
        ],
    )
    # plan → execute 文本 → review pass → aggregate 文本
    return ScriptedChatModel(
        scripted_responses=[
            plan_call,
            "子任务产出正文",
            review_pass,
            "最终汇总交付",
        ]
    )


@pytest.fixture
def make_checkpointer() -> Iterator[Callable[[], SqliteSaver]]:
    """创建可关闭的临时 SQLite checkpointer 工厂（Windows 文件锁安全）。"""
    fd, name = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    path = Path(name)
    savers: list[SqliteSaver] = []

    def factory() -> SqliteSaver:
        saver = open_sqlite_checkpointer(path)
        savers.append(saver)
        return saver

    yield factory

    for saver in savers:
        with contextlib.suppress(OSError):
            saver.conn.close()
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{name}{suffix}")
        with contextlib.suppress(PermissionError):
            p.unlink(missing_ok=True)


class TestGraphWiring:
    """图结构接线。"""

    def test_nodes_connected(self) -> None:
        """plan→assign→execute→review，aggregate 为终点。"""
        graph = build_supervisor_graph(_passing_llm(), _registry())
        rendered = graph.get_graph()
        names = set(rendered.nodes)
        for node in ("plan", "assign", "execute", "review", "aggregate"):
            assert node in names

    def test_end_to_end_pass_without_checkpointer(self) -> None:
        """无 checkpointer 也可跑通通过路径。"""
        graph = build_supervisor_graph(_passing_llm(), _registry())
        result = graph.invoke(initial_supervisor_state("写一句总结"))
        assert result["final_answer"]
        assert result["degraded"] is False
        assert result["verdict"] is not None
        assert result["verdict"].passed is True
        assert result["repair_count"] == 0

    def test_initial_state_has_final_fields(self) -> None:
        state = initial_supervisor_state("T")
        assert state["final_answer"] == ""
        assert state["degraded"] is False


class TestSqliteCheckpointerResume:
    """SQLite checkpointer：thread_id 持久化与断点续跑。"""

    def test_state_persisted_to_sqlite(self, make_checkpointer: Any) -> None:
        """invoke 后状态可在同一 checkpointer 上读回。"""
        checkpointer = make_checkpointer()
        graph = build_supervisor_graph(_passing_llm(), _registry(), checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "task-resume-1"}}
        graph.invoke(initial_supervisor_state("持久化检查"), config)

        snapshot = graph.get_state(config)
        assert snapshot.values.get("final_answer")

    def test_interrupt_then_resume_completes(self, make_checkpointer: Any) -> None:
        """模拟中断：interrupt_after=execute → 新图实例同 thread_id 续跑至完成。"""
        from langgraph.graph import END, START, StateGraph

        from agent_workbench.orchestrator.nodes import (
            create_aggregate_node,
            create_assign_node,
            create_execute_node,
            create_plan_node,
            create_review_node,
            create_review_router,
        )
        from agent_workbench.orchestrator.state import SupervisorState

        llm = _passing_llm()
        registry = _registry()
        saver = make_checkpointer()

        def make_builder() -> StateGraph[SupervisorState]:
            b: StateGraph[SupervisorState] = StateGraph(SupervisorState)
            b.add_node("plan", create_plan_node(llm))
            b.add_node("assign", create_assign_node())
            b.add_node("execute", create_execute_node(llm, registry, max_steps=4))
            b.add_node("review", create_review_node(llm))
            b.add_node("aggregate", create_aggregate_node(llm))
            b.add_edge(START, "plan")
            b.add_edge("plan", "assign")
            b.add_edge("assign", "execute")
            b.add_edge("execute", "review")
            b.add_conditional_edges(
                "review",
                create_review_router(),
                {"assign": "assign", "aggregate": "aggregate"},
            )
            b.add_edge("aggregate", END)
            return b

        # 中断版：execute 之后停住（等价于进程在 execute 后被杀）
        graph_a = make_builder().compile(checkpointer=saver, interrupt_after=["execute"])
        config = {"configurable": {"thread_id": "task-resume-2"}}
        partial = graph_a.invoke(initial_supervisor_state("断点续跑任务"), config)

        # 中断时：已有 plan/assign/execute 产物，尚未 review/aggregate
        assert partial["results"], "中断前应已有执行结果"
        assert partial.get("final_answer", "") == ""
        state_after = graph_a.get_state(config)
        assert state_after.next, "中断后应仍有后续节点"

        # 「重启进程」：新图实例、同一 SQLite、同一 thread_id，继续跑完
        graph_b = build_supervisor_graph(
            llm,
            registry,
            checkpointer=make_checkpointer(),
        )
        resumed = graph_b.invoke(None, config)

        assert resumed["final_answer"], "续跑后必须产出最终交付"
        assert resumed["verdict"] is not None and resumed["verdict"].passed
        assert resumed["degraded"] is False
        # 续跑不重复执行已完成的 execute（结果仍存在）
        assert resumed["results"]

    def test_resume_same_thread_shares_history(self, make_checkpointer: Any) -> None:
        """同 thread_id 第二次 get_state 能看到第一步写入的 plan。"""
        graph = build_supervisor_graph(
            _passing_llm(),
            _registry(),
            checkpointer=make_checkpointer(),
        )
        config = {"configurable": {"thread_id": "task-resume-3"}}
        graph.invoke(initial_supervisor_state("共享历史"), config)

        again = graph.get_state(config)
        assert again.values["plan"], "plan 应持久化在 checkpointer 中"
        assert again.values["final_answer"]

    def test_two_threads_isolated(self, make_checkpointer: Any) -> None:
        """不同 thread_id 状态互不覆盖。"""
        graph = build_supervisor_graph(
            _passing_llm(),
            _registry(),
            checkpointer=make_checkpointer(),
        )
        c1 = {"configurable": {"thread_id": "iso-1"}}
        c2 = {"configurable": {"thread_id": "iso-2"}}
        graph.invoke(initial_supervisor_state("任务一"), c1)
        graph.invoke(initial_supervisor_state("任务二"), c2)

        v1 = graph.get_state(c1).values
        v2 = graph.get_state(c2).values
        assert v1["task_description"] == "任务一"
        assert v2["task_description"] == "任务二"
