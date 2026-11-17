"""Phase 1 集成测试：任务端到端、工具调用断言、超限终止、SSE 过程可见。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from agent_workbench.application.services.task import TaskService
from agent_workbench.domain.schemas import TaskStatus
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.main import build_default_registry, create_app
from agent_workbench.orchestrator.single_agent import create_single_agent_graph, initial_state
from config.settings import Settings


def _finish_call(answer: str, call_id: str) -> AIMessage:
    """构造调用 finish 的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": "finish", "args": {"answer": answer}, "id": call_id}],
    )


def _always_tool_call(call_id: str) -> AIMessage:
    """构造无限循环用的 tool_call（参数合法以便工具可执行）。"""
    return _finish_call("循环交付", call_id)


async def _wait_terminal(service: TaskService, task_id: str, *, rounds: int = 200) -> None:
    """轮询直到任务终态（防挂死上限）。"""
    for _ in range(rounds):
        if service.is_terminal(task_id):
            return
        await asyncio.sleep(0.05)
    msg = f"任务未在预期轮次内到终态: {task_id}"
    raise AssertionError(msg)


async def _drain_events(service: TaskService, task_id: str) -> list[dict[str, Any]]:
    """拉取直到流结束的全部事件。"""
    events: list[dict[str, Any]] = []
    async for event in service.stream_events(task_id):
        events.append(event)
    return events


class TestEndToEndTask:
    """任务端到端：创建 → 执行 → 终态 → 交付正文。"""

    def test_task_completes_with_answer(self) -> None:
        """无工具路径：任务到 completed 且无错误。"""

        async def run() -> None:
            llm = ScriptedChatModel(scripted_responses=["端到端最终答案"])
            service = TaskService(llm, build_default_registry(), max_steps=5)
            task = await service.create_task("生成一句总结")
            await _wait_terminal(service, task.id)
            final = service.get_task(task.id)
            assert final.status is TaskStatus.COMPLETED
            assert final.error is None

        asyncio.run(run())

    def test_task_with_tool_call_finish(self) -> None:
        """工具路径：finish 被调用，终态 answer 来自 final_delivery。"""

        async def run() -> None:
            llm = ScriptedChatModel(
                scripted_responses=[
                    _finish_call("工具交付的正文", "c_e2e"),
                    "收到",
                ]
            )
            service = TaskService(llm, build_default_registry(), max_steps=5)
            task = await service.create_task("请用 finish 交付")
            await _wait_terminal(service, task.id)

            final = service.get_task(task.id)
            assert final.status is TaskStatus.COMPLETED

            events = await _drain_events(service, task.id)
            completed = [e for e in events if e["type"] == "task_completed"]
            assert completed, events
            assert completed[0]["answer"] == "工具交付的正文"
            assert completed[0]["step_count"] == 1

        asyncio.run(run())


class TestToolCallingAssertions:
    """图级工具调用集成断言。"""

    def test_python_sandbox_executed_in_loop(self) -> None:
        """模型请求 python_sandbox → 子进程真实执行 → ToolMessage 含输出。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "python_sandbox",
                            "args": {"code": "print(6 * 7)"},
                            "id": "c_sbx",
                        }
                    ],
                ),
                "答案是 42",
            ]
        )
        registry = build_default_registry()
        graph = create_single_agent_graph(llm, registry, max_steps=5)
        state = graph.invoke(initial_state("算 6*7", max_steps=5))

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "c_sbx"
        assert "42" in tool_msgs[0].content
        assert state["step_count"] == 1
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        assert last.content == "答案是 42"

    def test_finish_tool_emits_final_delivery_json(self) -> None:
        """finish 执行结果为结构化 JSON 交付包。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _finish_call("交付内容", "c_fin"),
                "done",
            ]
        )
        graph = create_single_agent_graph(llm, build_default_registry(), max_steps=3)
        state = graph.invoke(initial_state("交付", max_steps=3))

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        payload = json.loads(tool_msgs[0].content)
        assert payload["type"] == "final_delivery"
        assert payload["answer"] == "交付内容"
        assert payload["artifacts"] == []


class TestBoundedTermination:
    """超限终止：循环计数封顶且图必须结束。"""

    def test_loop_stops_at_max_steps_graph_level(self) -> None:
        """图级：持续 tool_call 在 max_steps 处终止，不无限循环。"""
        max_steps = 4
        llm = ScriptedChatModel(scripted_responses=[_always_tool_call("c_bounded")])
        graph = create_single_agent_graph(llm, build_default_registry(), max_steps=max_steps)
        state = graph.invoke(initial_state("永远调工具", max_steps=max_steps))

        assert state["step_count"] == max_steps
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == max_steps

    def test_service_level_bounded_termination(self) -> None:
        """服务级：超限后任务仍到终态，不挂起。"""
        max_steps = 3

        async def run() -> None:
            llm = ScriptedChatModel(scripted_responses=[_always_tool_call("c_svc")])
            service = TaskService(llm, build_default_registry(), max_steps=max_steps)
            task = await service.create_task("服务级超限")
            await _wait_terminal(service, task.id)

            final = service.get_task(task.id)
            assert final.status is TaskStatus.COMPLETED, final.error

            events = await _drain_events(service, task.id)
            completed = [e for e in events if e["type"] == "task_completed"]
            assert completed, events
            assert completed[0]["step_count"] == max_steps

        asyncio.run(run())


class TestApiEndToEndSse:
    """API 端到端 + SSE 过程可见（P1 验收）。"""

    @pytest.fixture
    def client(self) -> Iterator[TestClient]:
        """隔离配置的 TestClient。"""
        settings = Settings(_env_file=None, app_env="test", chat_model="")
        app = create_app(settings, max_steps=5)
        with TestClient(app) as c:
            yield c

    def test_full_flow_with_sse(self, client: TestClient) -> None:
        """POST → SSE 见过程事件 → GET 终态 completed。"""
        created = client.post("/tasks", json={"description": "端到端 SSE"}).json()
        task_id = created["id"]
        assert created["status"] == "running"

        with client.stream("GET", f"/tasks/{task_id}/stream") as resp:
            assert resp.status_code == 200
            raw = "".join(resp.iter_text())

        events = []
        for line in raw.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
        types = [e["type"] for e in events]
        assert types[0] == "task_started"
        assert "task_completed" in types or "task_failed" in types

        final = client.get(f"/tasks/{task_id}").json()
        assert final["status"] in ("completed", "failed")
        assert final["id"] == task_id

    def test_tool_call_task_via_api(self, client: TestClient) -> None:
        """默认替身不调工具也能走完整 API 闭环。"""
        created = client.post("/tasks", json={"description": "最小闭环"}).json()
        with client.stream("GET", f"/tasks/{created['id']}/stream") as resp:
            raw = "".join(resp.iter_text())
        assert "task_completed" in raw or "task_failed" in raw
        final = client.get(f"/tasks/{created['id']}").json()
        assert final["status"] == "completed"
