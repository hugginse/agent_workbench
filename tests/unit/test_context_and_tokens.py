"""上下文裁剪与 token 统计单测。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.memory.context import (
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    merge_token_usage,
    prune_messages,
)
from agent_workbench.orchestrator.nodes import create_execute_node, create_plan_node
from agent_workbench.orchestrator.state import SubTask, initial_supervisor_state
from agent_workbench.tools.builtins import FinishTool, PythonSandboxTool
from agent_workbench.tools.registry import ToolRegistry


class TestEstimateTokens:
    """确定性 token 估算。"""

    def test_empty_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_ascii_approx(self) -> None:
        # 8 个 ASCII → ceil(8/4)=2
        assert estimate_tokens("abcdefgh") == 2

    def test_cjk_one_per_char(self) -> None:
        assert estimate_tokens("多智能体") == 4

    def test_mixed(self) -> None:
        # 中文 2 + "abcd" 4 字符 → 1
        assert estimate_tokens("中中abcd") == 3

    def test_deterministic(self) -> None:
        text = "固定输入 deterministic 123"
        assert estimate_tokens(text) == estimate_tokens(text)

    def test_message_tokens_includes_tool_calls(self) -> None:
        m = AIMessage(
            content="hi",
            tool_calls=[{"name": "finish", "args": {"answer": "很长的交付内容"}, "id": "c1"}],
        )
        plain = AIMessage(content="hi")
        assert estimate_message_tokens(m) > estimate_message_tokens(plain)

    def test_messages_sum(self) -> None:
        msgs = [HumanMessage(content="abcd"), HumanMessage(content="efgh")]
        assert estimate_messages_tokens(msgs) == 2


class TestPruneMessages:
    """裁剪：保 system + 最近条数。"""

    def _seq(self, n: int) -> list:
        msgs: list = [SystemMessage(content="系统提示")]
        for i in range(n):
            msgs.append(HumanMessage(content=f"消息{i}"))
        return msgs

    def test_no_prune_when_under_limit(self) -> None:
        msgs = self._seq(3)
        assert prune_messages(msgs, max_messages=10) == msgs

    def test_keeps_system_and_tail(self) -> None:
        msgs = self._seq(30)
        out = prune_messages(msgs, max_messages=5)
        assert len(out) == 5
        assert isinstance(out[0], SystemMessage)
        # 尾部保留
        assert out[-1].content == "消息29"

    def test_does_not_mutate_input(self) -> None:
        msgs = self._seq(30)
        original_len = len(msgs)
        prune_messages(msgs, max_messages=5)
        assert len(msgs) == original_len

    def test_without_system_keeps_tail(self) -> None:
        msgs = [HumanMessage(content=f"m{i}") for i in range(20)]
        out = prune_messages(msgs, max_messages=4, keep_system=False)
        assert len(out) == 4
        assert out[-1].content == "m19"

    def test_max_messages_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_messages"):
            prune_messages([HumanMessage(content="x")], max_messages=1)


class TestMergeTokenUsage:
    """usage 合并。"""

    def test_from_none(self) -> None:
        assert merge_token_usage(None, input_tokens=3, output_tokens=2) == {
            "input": 3,
            "output": 2,
            "total": 5,
        }

    def test_accumulates(self) -> None:
        base = {"input": 10, "output": 5, "total": 15}
        out = merge_token_usage(base, input_tokens=4, output_tokens=1)
        assert out == {"input": 14, "output": 6, "total": 20}

    def test_does_not_mutate_base(self) -> None:
        base = {"input": 1, "output": 1, "total": 2}
        merge_token_usage(base, input_tokens=10, output_tokens=10)
        assert base["input"] == 1


class TestNodeTokenAccounting:
    """节点把 token 计入 state。"""

    def test_plan_node_adds_usage(self) -> None:
        llm = ScriptedChatModel(
            scripted_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "PlanOutput",
                            "args": {"subtasks": [{"id": "s1", "description": "d"}]},
                            "id": "c",
                        }
                    ],
                )
            ]
        )
        state = initial_supervisor_state("任务")
        update = create_plan_node(llm)(state)
        usage = update["token_usage"]
        assert usage["input"] > 0
        assert usage["output"] >= 0
        assert usage["total"] == usage["input"] + usage["output"]
        # 入参未被改
        assert state["token_usage"]["input"] == 0

    def test_execute_node_accumulates_usage(self) -> None:
        llm = ScriptedChatModel(scripted_responses=["产出正文"])
        r = ToolRegistry()
        r.register(FinishTool())
        r.register(PythonSandboxTool())
        state = initial_supervisor_state("T")
        state["plan"] = [SubTask(id="s1", description="生成摘要", assignee="executor")]
        update = create_execute_node(llm, r)(state)
        assert update["results"]["s1"] == "产出正文"
        assert update["token_usage"]["input"] > 0
        assert update["token_usage"]["total"] >= update["token_usage"]["input"]

    def test_execute_accumulates_across_calls(self) -> None:
        llm = ScriptedChatModel(scripted_responses=["a", "b"])
        r = ToolRegistry()
        r.register(FinishTool())
        r.register(PythonSandboxTool())
        state = initial_supervisor_state("T")
        state["plan"] = [
            SubTask(id="s1", description="x", assignee="executor"),
            SubTask(id="s2", description="y", assignee="executor"),
        ]
        state["token_usage"] = {"input": 100, "output": 50, "total": 150}
        update = create_execute_node(llm, r)(state)
        assert update["token_usage"]["input"] > 100
        assert update["token_usage"]["output"] >= 50

    def test_run_subagent_returns_outcome_with_tokens(self) -> None:
        from agent_workbench.agents import EXECUTOR
        from agent_workbench.orchestrator.nodes import run_subagent

        llm = ScriptedChatModel(scripted_responses=["结果"])
        r = ToolRegistry()
        r.register(FinishTool())
        r.register(PythonSandboxTool())
        outcome = run_subagent(
            EXECUTOR,
            llm,
            r,
            subtask=SubTask(id="s1", description="计算"),
            task_description="T",
        )
        assert outcome.content == "结果"
        assert outcome.input_tokens > 0
        assert outcome.output_tokens >= 0

    def test_prune_applied_in_subagent_loop(self) -> None:
        """超长历史在 invoke 前被裁剪（spy 消息条数）。"""
        from agent_workbench.agents import EXECUTOR
        from agent_workbench.orchestrator.nodes import run_subagent

        # 每次都要求工具，迫使多轮累积消息
        always = AIMessage(
            content="",
            tool_calls=[{"name": "python_sandbox", "args": {"code": "print(1)"}, "id": "c1"}],
        )
        llm = ScriptedChatModel(scripted_responses=[always])
        seen_sizes: list[int] = []
        original = llm._generate  # type: ignore[method-assign]

        def spy(messages, **kwargs):  # type: ignore[no-untyped-def]
            seen_sizes.append(len(messages))
            return original(messages, **kwargs)

        llm._generate = spy  # type: ignore[method-assign]

        r = ToolRegistry()
        r.register(PythonSandboxTool())
        run_subagent(
            EXECUTOR,
            llm,
            r,
            subtask=SubTask(id="s1", description="x"),
            task_description="T",
            max_steps=6,
            max_context_messages=4,
        )
        assert seen_sizes, "应有 invoke"
        assert all(size <= 4 for size in seen_sizes), seen_sizes
