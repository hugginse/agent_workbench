"""memory_recall 工具单测：检索、空结果、跨会话召回。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_workbench.memory import SqliteLongTermMemory, TaskReflection
from agent_workbench.tools import ToolError, ToolRegistry
from agent_workbench.tools.builtins import MemoryRecallTool


def _seed(mem: SqliteLongTermMemory) -> None:
    """写入两条可区分的复盘。"""
    mem.save(
        TaskReflection(
            task_id="t-chroma",
            task_description="评估向量库",
            summary="Chroma 适合原型阶段快速验证",
            outcome="completed",
        )
    )
    mem.save(
        TaskReflection(
            task_id="t-docker",
            task_description="容器部署清单",
            summary="Docker 需挂载 .data 卷保证数据持久化",
            outcome="completed",
        )
    )


class TestMemoryRecallTool:
    """工具基本行为。"""

    def test_returns_matching_hits(self) -> None:
        with SqliteLongTermMemory(":memory:") as mem:
            _seed(mem)
            tool = MemoryRecallTool(mem)
            raw = tool.run({"query": "Chroma"})
            payload = json.loads(raw)
            assert payload["query"] == "Chroma"
            assert len(payload["hits"]) == 1
            assert payload["hits"][0]["task_id"] == "t-chroma"
            assert payload["hits"][0]["summary"]

    def test_result_marked_as_data_not_instruction(self) -> None:
        """输出带「数据不是指令」提示（注入防护）。"""
        with SqliteLongTermMemory(":memory:") as mem:
            _seed(mem)
            payload = json.loads(MemoryRecallTool(mem).run({"query": "Chroma"}))
            assert "不是系统指令" in payload["note"]

    def test_empty_hits_message(self) -> None:
        with SqliteLongTermMemory(":memory:") as mem:
            payload = json.loads(MemoryRecallTool(mem).run({"query": "不存在xyz"}))
            assert payload["hits"] == []
            assert "无匹配" in payload["message"]

    def test_limit_respected(self) -> None:
        with SqliteLongTermMemory(":memory:") as mem:
            for i in range(8):
                mem.save(
                    TaskReflection(
                        task_id=f"t-{i}",
                        task_description=f"任务{i}",
                        summary=f"经验要点{i}",
                    )
                )
            tool = MemoryRecallTool(mem)
            payload = json.loads(tool.run({"query": "经验", "limit": 3}))
            assert len(payload["hits"]) == 3

    def test_empty_query_rejected(self) -> None:
        with SqliteLongTermMemory(":memory:") as mem, pytest.raises(ToolError, match="校验失败"):
            MemoryRecallTool(mem).run({"query": ""})

    def test_registers_in_registry(self) -> None:
        with SqliteLongTermMemory(":memory:") as mem:
            r = ToolRegistry()
            r.register(MemoryRecallTool(mem))
            assert "memory_recall" in r
            assert any(s.name == "memory_recall" for s in r.specs())


class TestCrossSessionRecall:
    """跨会话召回：新连接读到旧会话写入。"""

    def test_new_session_reads_previous_writes(self, tmp_path: Path) -> None:
        db = tmp_path / "memory.db"
        # 会话 1：写入并关闭
        with SqliteLongTermMemory(db) as session1:
            _seed(session1)

        # 会话 2：同一文件新实例（模拟进程重启）
        with SqliteLongTermMemory(db) as session2:
            tool = MemoryRecallTool(session2)
            payload = json.loads(tool.run({"query": "Docker"}))
            assert len(payload["hits"]) == 1
            assert payload["hits"][0]["task_id"] == "t-docker"
            # 全量也能查到会话 1 的另一条
            all_hits = json.loads(tool.run({"query": "Chroma", "limit": 5}))
            assert any(h["task_id"] == "t-chroma" for h in all_hits["hits"])

    def test_cross_session_via_default_registry_path(self, tmp_path: Path) -> None:
        """经 build_default_registry(memory=...) 注册后跨会话可用。"""
        from agent_workbench.main import build_default_registry

        db = tmp_path / "mem.db"
        with SqliteLongTermMemory(db) as s1:
            s1.save(
                TaskReflection(
                    task_id="t-x",
                    task_description="历史任务",
                    summary="关键结论：先写测试再实现",
                    outcome="completed",
                )
            )

        with SqliteLongTermMemory(db) as s2:
            registry = build_default_registry(memory=s2)
            raw = registry.execute("memory_recall", {"query": "先写测试"})
            payload = json.loads(raw)
            assert payload["hits"][0]["task_id"] == "t-x"
