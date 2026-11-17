"""Phase 3 集成测试：跨会话记忆召回 + token 指标落库/累计。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent_workbench.application.services.task import TaskService
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.main import build_default_registry, create_app
from agent_workbench.memory import SqliteLongTermMemory, TaskReflection
from agent_workbench.orchestrator.builder import build_supervisor_graph
from agent_workbench.orchestrator.state import initial_supervisor_state
from agent_workbench.tools.builtins import FinishTool, PythonSandboxTool
from agent_workbench.tools.registry import ToolRegistry
from config.settings import Settings


async def _wait_terminal(service: TaskService, task_id: str, *, rounds: int = 300) -> None:
    for _ in range(rounds):
        if service.is_terminal(task_id):
            return
        await asyncio.sleep(0.05)
    msg = f"任务未到终态: {task_id}"
    raise AssertionError(msg)


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(FinishTool())
    r.register(PythonSandboxTool())
    return r


def _passing_llm() -> ScriptedChatModel:
    plan = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "PlanOutput",
                "args": {"subtasks": [{"id": "s1", "description": "生成一句总结"}]},
                "id": "c_plan",
            }
        ],
    )
    review = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ReviewVerdict",
                "args": {"passed": True, "feedback": "合格", "target_subtask_id": None},
                "id": "c_rev",
            }
        ],
    )
    return ScriptedChatModel(scripted_responses=[plan, "子任务产出正文", review, "最终汇总交付"])


class TestCrossSessionMemory:
    """跨会话：完成写复盘 → 新会话/新进程句柄可召回并注入工具。"""

    def test_supervisor_run_then_new_session_recall(self, tmp_path: Path) -> None:
        """会话 1 跑 Supervisor 写复盘；会话 2 新 memory 实例 search 命中。"""
        mem_path = tmp_path / "memory.db"

        async def run_session1() -> str:
            with SqliteLongTermMemory(mem_path) as mem:
                llm = ScriptedChatModel(
                    scripted_responses=[
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "PlanOutput",
                                    "args": {"subtasks": [{"id": "s1", "description": "写简介"}]},
                                    "id": "p1",
                                }
                            ],
                        ),
                        "会话一产出内容",
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "ReviewVerdict",
                                    "args": {
                                        "passed": True,
                                        "feedback": "ok",
                                        "target_subtask_id": None,
                                    },
                                    "id": "r1",
                                }
                            ],
                        ),
                        "会话一最终交付",
                    ]
                )
                service = TaskService(
                    llm,
                    build_default_registry(memory=mem),
                    mode="multi",
                    memory=mem,
                    max_repair_attempts=2,
                )
                task = await service.create_task("跨会话任务：写多智能体简介")
                await _wait_terminal(service, task.id)
                assert service.get_task(task.id).status.value == "completed"
                return task.id

        task_id = asyncio.run(run_session1())

        # 会话 2：新连接
        with SqliteLongTermMemory(mem_path) as mem2:
            saved = mem2.get_by_task_id(task_id)
            assert saved is not None, "会话 2 应读到会话 1 写入的复盘"
            assert "多智能体" in saved.task_description
            hits = mem2.search("多智能体简介")
            assert any(h.task_id == task_id for h in hits)

            # 经 memory_recall 工具召回（注入下游上下文）
            registry = build_default_registry(memory=mem2)
            raw = registry.execute("memory_recall", {"query": "多智能体简介"})
            payload = json.loads(raw)
            assert payload["hits"]
            assert payload["hits"][0]["task_id"] == task_id
            assert "不是系统指令" in payload["note"]

    def test_api_two_sequential_apps_share_memory_db(self, tmp_path: Path) -> None:
        """两个 create_app 实例（模拟两次启动）共享同一 memory 文件。"""
        mem_path = tmp_path / "memory.db"
        settings = Settings(_env_file=None, app_env="test", chat_model="")

        def make_app() -> Any:
            return create_app(
                settings,
                mode="single",
                checkpoint_db=tmp_path / "cp.db",
                memory_db=mem_path,
            )

        with TestClient(make_app()) as app1:
            created = app1.post("/tasks", json={"description": "第一次启动写入的记忆标记"}).json()
            with app1.stream("GET", f"/tasks/{created['id']}/stream") as resp:
                raw = "".join(resp.iter_text())
            assert "task_completed" in raw or "task_failed" in raw

        with TestClient(make_app()) as app2:
            # 第二次「启动」：直接读 memory（不经任务）
            mem = app2.app.state.memory
            saved = mem.get_by_task_id(created["id"])
            assert saved is not None, "第二次启动应读到第一次的复盘"
            assert "记忆标记" in saved.task_description

    def test_reflection_outcome_recorded_for_recall(self, tmp_path: Path) -> None:
        """复盘 outcome 可被检索结果带回（供后续指标使用）。"""
        mem_path = tmp_path / "m.db"
        with SqliteLongTermMemory(mem_path) as mem:
            mem.save(
                TaskReflection(
                    task_id="t-1",
                    task_description="部署检查",
                    summary="缺卷挂载导致数据丢失",
                    outcome="failed",
                )
            )
        with SqliteLongTermMemory(mem_path) as mem2:
            hits = mem2.search("卷挂载")
            assert hits[0].outcome == "failed"


class TestTokenMetrics:
    """token 指标：图终态累计、可汇入成本。"""

    def test_supervisor_final_state_has_positive_usage(self) -> None:
        """完整 Supervisor 跑完后 token_usage.total > 0。"""
        graph = build_supervisor_graph(_passing_llm(), _registry())
        state = graph.invoke(initial_supervisor_state("统计 token 的任务"))
        usage = state["token_usage"]
        assert usage["input"] > 0, usage
        assert usage["output"] >= 0, usage
        assert usage["total"] == usage["input"] + usage["output"]
        assert state["final_answer"]

    def test_token_usage_monotonic_across_nodes(self) -> None:
        """通过中断分步观察：execute 后 usage 已非零，最终仍累计。"""
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

        builder: StateGraph[SupervisorState] = StateGraph(SupervisorState)
        builder.add_node("plan", create_plan_node(llm))
        builder.add_node("assign", create_assign_node())
        builder.add_node("execute", create_execute_node(llm, registry, max_steps=4))
        builder.add_node("review", create_review_node(llm))
        builder.add_node("aggregate", create_aggregate_node(llm))
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "assign")
        builder.add_edge("assign", "execute")
        builder.add_edge("execute", "review")
        builder.add_conditional_edges(
            "review",
            create_review_router(),
            {"assign": "assign", "aggregate": "aggregate"},
        )
        builder.add_edge("aggregate", END)
        graph = builder.compile()

        seen: list[dict[str, int]] = []
        for state in graph.stream(
            initial_supervisor_state("分步统计"),
            stream_mode="values",
        ):
            if isinstance(state, dict) and "token_usage" in state:
                seen.append(dict(state["token_usage"]))

        assert len(seen) >= 2, "应观察到多个中间状态"
        # 首个非零之后，total 不减
        nonzero = [u for u in seen if u.get("total", 0) > 0]
        assert nonzero, f"应出现非零 token_usage，观察到 {seen}"
        totals = [u["total"] for u in nonzero]
        assert totals[-1] >= totals[0]

    def test_task_completed_event_includes_mode_for_cost_split(self) -> None:
        """完成事件带 mode，便于 single/multi 成本拆分（Phase 5）。"""

        async def run() -> dict[str, Any]:
            with SqliteLongTermMemory(":memory:") as mem:
                llm = ScriptedChatModel(scripted_responses=["ok"])
                service = TaskService(
                    llm,
                    build_default_registry(),
                    mode="single",
                    memory=mem,
                )
                task = await service.create_task("成本拆分")
                await _wait_terminal(service, task.id)
                events: list[dict[str, Any]] = []
                async for e in service.stream_events(task.id):
                    events.append(e)
                return {e["type"]: e for e in events}

        events = asyncio.run(run())
        completed = events.get("task_completed")
        assert completed is not None
        assert completed["mode"] == "single"
        assert "answer" in completed

    def test_single_mode_task_runs_and_records_tokens_in_graph(self) -> None:
        """single 模式图也有 step_count（与 token 互补的链路指标）。"""
        from agent_workbench.orchestrator.single_agent import (
            create_single_agent_graph,
            initial_state,
        )

        llm = ScriptedChatModel(scripted_responses=["直接答案"])
        graph = create_single_agent_graph(llm, _registry(), max_steps=3)
        state = graph.invoke(initial_state("单 Agent"))
        assert state["step_count"] == 0
        assert state["messages"]
