"""领域 schema（Task / AgentMessage / ToolCall / TaskResult）单测。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_workbench.domain.schemas import (
    AgentMessage,
    MessageRole,
    Task,
    TaskResult,
    TaskStatus,
    ToolCall,
)


class TestTask:
    """Task schema 校验与默认值。"""

    def test_minimal_task(self) -> None:
        """仅提供 description 即可创建，默认 pending 且时间为 UTC。"""
        t = Task(description="调研向量数据库方案")
        assert t.id
        assert t.status is TaskStatus.PENDING
        assert t.created_at.tzinfo is not None
        assert t.updated_at.tzinfo is not None
        assert t.error is None

    def test_empty_description_rejected(self) -> None:
        """空描述必须被拒绝。"""
        with pytest.raises(ValidationError):
            Task(description="")

    def test_status_roundtrip(self) -> None:
        """状态序列化为字符串后可还原。"""
        t = Task(description="x", status=TaskStatus.RUNNING)
        assert t.status == "running"
        assert Task.model_validate_json(t.model_dump_json()).status is TaskStatus.RUNNING

    def test_unique_ids(self) -> None:
        """缺省 id 每次不同。"""
        assert Task(description="a").id != Task(description="b").id


class TestToolCall:
    """ToolCall schema。"""

    def test_valid_call(self) -> None:
        """合法调用可创建。"""
        call = ToolCall(id="c1", name="retrieve", arguments={"query": "q", "top_k": 3})
        assert call.name == "retrieve"
        assert call.arguments["top_k"] == 3

    def test_missing_id_or_name_rejected(self) -> None:
        """id / name 不得为空。"""
        with pytest.raises(ValidationError):
            ToolCall(id="", name="retrieve")
        with pytest.raises(ValidationError):
            ToolCall(id="c1", name="")


class TestAgentMessage:
    """AgentMessage schema。"""

    def test_user_message(self) -> None:
        """用户消息默认无工具调用。"""
        m = AgentMessage(role=MessageRole.USER, content="你好")
        assert m.tool_calls == []
        assert m.agent_name is None
        assert m.timestamp.tzinfo is not None

    def test_assistant_message_with_tool_calls(self) -> None:
        """assistant 消息可携带工具调用。"""
        m = AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            agent_name="executor",
            tool_calls=[ToolCall(id="c1", name="finish", arguments={})],
        )
        assert m.tool_calls[0].name == "finish"
        assert m.agent_name == "executor"

    def test_tool_result_message(self) -> None:
        """tool 角色消息通过 tool_call_id 关联回调用。"""
        m = AgentMessage(
            role=MessageRole.TOOL,
            content="检索结果",
            tool_call_id="c1",
        )
        assert m.tool_call_id == "c1"

    def test_role_enum_values(self) -> None:
        """角色值与字符串枚举一致。"""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestTaskResult:
    """TaskResult schema。"""

    def test_success_result(self) -> None:
        """成功结果：completed + answer。"""
        r = TaskResult(task_id="t1", status=TaskStatus.COMPLETED, answer="结论")
        assert r.status is TaskStatus.COMPLETED
        assert r.artifacts == []
        assert r.error is None

    def test_failed_result(self) -> None:
        """失败结果：携带用户可见 error。"""
        r = TaskResult(task_id="t1", status=TaskStatus.FAILED, error="超限终止")
        assert r.error == "超限终止"

    def test_empty_task_id_rejected(self) -> None:
        """task_id 不得为空。"""
        with pytest.raises(ValidationError):
            TaskResult(task_id="", status=TaskStatus.COMPLETED)

    def test_json_roundtrip(self) -> None:
        """JSON 序列化往返保持字段。"""
        r = TaskResult(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            answer="ok",
            artifacts=["a.md"],
        )
        back = TaskResult.model_validate_json(r.model_dump_json())
        assert back == r


def test_domain_imports_no_framework() -> None:
    """领域 schema 源码不得 import Web / 编排框架（架构红线）。"""
    import ast
    from pathlib import Path

    domain_dir = Path(__file__).resolve().parents[2] / "src" / "agent_workbench" / "domain"
    forbidden = frozenset(
        {"fastapi", "langchain", "langchain_core", "langgraph", "chromadb", "sqlite3"}
    )
    checked = 0
    for path in domain_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                assert root not in forbidden, f"{path.name} 禁止 import {name}"
                checked += 1
    assert checked > 0, "应至少检查到领域层的 import 语句"
