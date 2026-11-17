"""任务完成路径写入长期复盘的集成测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent_workbench.application.services.task import TaskService
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.main import build_default_registry, create_app
from agent_workbench.memory import SqliteLongTermMemory, TaskReflection
from config.settings import Settings


async def _wait_async(service: TaskService, task_id: str, *, rounds: int = 200) -> None:
    for _ in range(rounds):
        if service.is_terminal(task_id):
            return
        await asyncio.sleep(0.05)
    msg = f"任务未到终态: {task_id}"
    raise AssertionError(msg)


class TestReflectionOnCompletion:
    """TaskService 终态写复盘。"""

    def test_completed_writes_reflection(self) -> None:
        """成功完成 → memory 中有 outcome=completed 的复盘。"""

        async def run() -> None:
            with SqliteLongTermMemory(":memory:") as mem:
                llm = ScriptedChatModel(scripted_responses=["交付正文ABC"])
                service = TaskService(
                    llm,
                    build_default_registry(),
                    max_steps=3,
                    mode="single",
                    memory=mem,
                )
                task = await service.create_task("生成一段总结")
                await _wait_async(service, task.id)

                saved = mem.get_by_task_id(task.id)
                assert saved is not None, "完成后应写入复盘"
                assert saved.outcome == "completed"
                assert saved.task_description == "生成一段总结"
                assert "正常完成" in saved.summary
                assert "交付正文ABC" in saved.summary

        asyncio.run(run())

    def test_failed_writes_reflection(self) -> None:
        """执行异常 → outcome=failed 复盘仍写入。"""

        async def run() -> None:
            with SqliteLongTermMemory(":memory:") as mem:

                class BoomModel(ScriptedChatModel):
                    def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
                        raise RuntimeError("模拟图崩溃")

                service = TaskService(
                    BoomModel(scripted_responses=["x"]),
                    build_default_registry(),
                    mode="single",
                    memory=mem,
                )
                task = await service.create_task("会失败的任务")
                await _wait_async(service, task.id)

                assert service.get_task(task.id).status.value == "failed"
                saved = mem.get_by_task_id(task.id)
                assert saved is not None
                assert saved.outcome == "failed"
                assert "RuntimeError" in saved.summary

        asyncio.run(run())

    def test_no_memory_skips_write(self) -> None:
        """memory=None 时不写复盘、不报错。"""

        async def run() -> None:
            llm = ScriptedChatModel(scripted_responses=["ok"])
            service = TaskService(llm, build_default_registry(), mode="single", memory=None)
            task = await service.create_task("无记忆任务")
            await _wait_async(service, task.id)
            assert service.get_task(task.id).status.value == "completed"

        asyncio.run(run())

    def test_degraded_outcome_mapped(self) -> None:
        """Supervisor 降级完成 → outcome=degraded。"""

        async def run() -> None:
            with SqliteLongTermMemory(":memory:") as mem:
                # 多步脚本：plan → execute → review fail → … 最终仍会 aggregate
                plan = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "PlanOutput",
                            "args": {"subtasks": [{"id": "s1", "description": "子任务"}]},
                            "id": "c1",
                        }
                    ],
                )
                fail_review = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReviewVerdict",
                            "args": {
                                "passed": False,
                                "feedback": "不合格",
                                "target_subtask_id": "s1",
                            },
                            "id": "c2",
                        }
                    ],
                )
                # 循环脚本：plan、执行文本、反复 fail 评审直到超限、汇总
                llm = ScriptedChatModel(
                    scripted_responses=[
                        plan,
                        "执行结果",
                        fail_review,
                        "再执行",
                        fail_review,
                        "再执行2",
                        fail_review,
                        "降级汇总",
                    ]
                )
                service = TaskService(
                    llm,
                    build_default_registry(),
                    mode="multi",
                    memory=mem,
                    max_repair_attempts=1,
                )
                task = await service.create_task("会降级的任务")
                await _wait_async(service, task.id, rounds=400)

                final = service.get_task(task.id)
                assert final.status.value == "completed"
                saved = mem.get_by_task_id(task.id)
                assert saved is not None
                assert saved.outcome == "degraded"

        asyncio.run(run())


class TestReflectionViaApi:
    """API 完成后可跨「会话」召回（同一 memory 文件）。"""

    def test_api_completion_persists_reflection(self, tmp_path: Path) -> None:
        mem_path = tmp_path / "memory.db"
        settings = Settings(_env_file=None, app_env="test", chat_model="")
        app = create_app(
            settings,
            mode="single",
            checkpoint_db=tmp_path / "cp.db",
            memory_db=mem_path,
        )
        with TestClient(app) as client:
            created = client.post("/tasks", json={"description": "API 复盘任务XYZ"}).json()
            with client.stream("GET", f"/tasks/{created['id']}/stream") as resp:
                raw = "".join(resp.iter_text())
            assert "task_completed" in raw or "task_failed" in raw

        # 新「会话」打开同一 memory 文件
        with SqliteLongTermMemory(mem_path) as mem:
            saved = mem.get_by_task_id(created["id"])
            assert saved is not None, "API 完成后复盘应持久化"
            assert saved.task_description == "API 复盘任务XYZ"
            hits = mem.search("API 复盘任务XYZ")
            assert any(h.task_id == created["id"] for h in hits)

    def test_reflection_json_shape(self, tmp_path: Path) -> None:
        """复盘字段完整可 JSON 往返。"""
        r = TaskReflection(
            task_id="t",
            task_description="d",
            summary="s",
            outcome="completed",
        )
        back = TaskReflection.model_validate_json(r.model_dump_json())
        assert back.task_id == "t"
        assert json.loads(r.model_dump_json())["outcome"] == "completed"
