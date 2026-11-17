"""SubAgent 声明与 assign / execute 节点单测。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from agent_workbench.agents import (
    EXECUTOR,
    RESEARCHER,
    REVIEWER,
    SubAgentSpec,
    SubTaskResult,
    get_subagent,
)
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.orchestrator.nodes import (
    create_assign_node,
    create_execute_node,
    resolve_assignee,
)
from agent_workbench.orchestrator.prompts import build_execute_prompt
from agent_workbench.orchestrator.state import (
    ReviewVerdict,
    SubTask,
    initial_supervisor_state,
)
from agent_workbench.tools.builtins import FinishTool, PythonSandboxTool
from agent_workbench.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    """注册 finish + python_sandbox。"""
    r = ToolRegistry()
    r.register(FinishTool())
    r.register(PythonSandboxTool())
    return r


class TestSubAgentSpecs:
    """三个 SubAgent 声明。"""

    def test_three_agents_declared(self) -> None:
        """researcher / executor / reviewer 均已声明。"""
        assert set(__import__("agent_workbench.agents", fromlist=["SUBAGENTS"]).SUBAGENTS) >= {
            "researcher",
            "executor",
            "reviewer",
        }

    def test_specs_have_prompt_and_tools(self) -> None:
        """声明含 system_prompt 与工具子集类型正确。"""
        for spec in (RESEARCHER, EXECUTOR, REVIEWER):
            assert isinstance(spec, SubAgentSpec)
            assert spec.system_prompt
            assert isinstance(spec.tool_names, tuple)
            assert spec.output_schema is SubTaskResult

    def test_tool_subsets_differ(self) -> None:
        """reviewer 无工具；researcher/executor 有 python_sandbox。"""
        assert REVIEWER.tool_names == ()
        assert "python_sandbox" in RESEARCHER.tool_names
        assert "python_sandbox" in EXECUTOR.tool_names

    def test_get_subagent_unknown_raises(self) -> None:
        """未声明角色抛 KeyError。"""
        try:
            get_subagent("nope")
        except KeyError as exc:
            assert "nope" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("应抛出 KeyError")

    def test_get_subagent_roundtrip(self) -> None:
        """按名取回同一声明。"""
        assert get_subagent("researcher") is RESEARCHER


class TestResolveAssignee:
    """assignee 解析。"""

    def test_explicit_assignee_preserved(self) -> None:
        """已声明且合法的 assignee 保留。"""
        st = SubTask(id="s1", description="随便", assignee="reviewer")
        assert resolve_assignee(st) == "reviewer"

    def test_invalid_assignee_overridden(self) -> None:
        """非法 assignee 被启发式覆盖。"""
        st = SubTask(id="s1", description="撰写报告", assignee="ghost")
        assert resolve_assignee(st) == "executor"

    def test_research_keyword(self) -> None:
        """调研类描述 → researcher。"""
        st = SubTask(id="s1", description="调研向量数据库方案")
        assert resolve_assignee(st) == "researcher"

    def test_review_keyword(self) -> None:
        """审查类描述 → reviewer。"""
        st = SubTask(id="s1", description="审查初稿格式")
        assert resolve_assignee(st) == "reviewer"

    def test_default_executor(self) -> None:
        """无关键词 → executor。"""
        st = SubTask(id="s1", description="生成一段摘要文本")
        assert resolve_assignee(st) == "executor"


class TestAssignNode:
    """assign 节点。"""

    def test_fills_assignees(self) -> None:
        """为 plan 中每个子任务写入 assignee。"""
        state = initial_supervisor_state("T")
        state["plan"] = [
            SubTask(id="s1", description="调研背景"),
            SubTask(id="s2", description="生成摘要"),
        ]
        update = create_assign_node()(state)
        plan = update["plan"]
        assert plan[0].assignee == "researcher"
        assert plan[1].assignee == "executor"

    def test_does_not_mutate_input(self) -> None:
        """不修改入参 state 的 plan 列表。"""
        state = initial_supervisor_state("T")
        state["plan"] = [SubTask(id="s1", description="生成摘要")]
        create_assign_node()(state)
        assert state["plan"][0].assignee is None


class TestExecuteNode:
    """execute 节点：派发、结果合并、回修重跑。"""

    def test_executes_unfinished_subtasks(self) -> None:
        """无结果的子任务被执行并写入 results。"""
        llm = ScriptedChatModel(scripted_responses=["调研结论A"])
        state = initial_supervisor_state("总任务")
        state["plan"] = [
            SubTask(id="s1", description="调研背景", assignee="researcher"),
            SubTask(id="s2", description="生成摘要", assignee="executor"),
        ]
        node = create_execute_node(llm, _registry(), max_steps=3)
        update = node(state)
        assert update["results"]["s1"] == "调研结论A"
        assert update["results"]["s2"] == "调研结论A"

    def test_skips_already_completed(self) -> None:
        """已有结果且非回修目标的子任务不重跑。"""
        llm = ScriptedChatModel(scripted_responses=["新结果"])
        state = initial_supervisor_state("T")
        state["plan"] = [SubTask(id="s1", description="已完成", assignee="executor")]
        state["results"] = {"s1": "旧结果"}
        update = create_execute_node(llm, _registry())(state)
        assert update["results"]["s1"] == "旧结果"

    def test_repair_reruns_only_target(self) -> None:
        """回修时仅重跑 verdict 目标，其它结果保留。"""
        llm = ScriptedChatModel(scripted_responses=["修正后的s2"])
        state = initial_supervisor_state("T")
        state["plan"] = [
            SubTask(id="s1", description="a", assignee="executor"),
            SubTask(id="s2", description="b", assignee="executor"),
        ]
        state["results"] = {"s1": "原s1", "s2": "原s2"}
        state["verdict"] = ReviewVerdict(
            passed=False,
            feedback="补充数据",
            target_subtask_id="s2",
        )
        update = create_execute_node(llm, _registry())(state)
        assert update["results"]["s1"] == "原s1"
        assert update["results"]["s2"] == "修正后的s2"

    def test_tool_call_executes_and_returns_final(self) -> None:
        """子任务内工具调用真实执行，最终返回文本。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "python_sandbox", "args": {"code": "print(1+1)"}, "id": "c1"}
                    ],
                ),
                "计算结果是2",
            ]
        )
        state = initial_supervisor_state("算加法")
        state["plan"] = [SubTask(id="s1", description="计算 1+1", assignee="executor")]
        update = create_execute_node(llm, _registry(), max_steps=5)(state)
        assert update["results"]["s1"] == "计算结果是2"

    def test_bounded_steps_returns_limit_message(self) -> None:
        """持续请求工具 → 达到 max_steps 返回上限说明。"""
        always = AIMessage(
            content="",
            tool_calls=[{"name": "python_sandbox", "args": {"code": "print(0)"}, "id": "c_loop"}],
        )
        llm = ScriptedChatModel(scripted_responses=[always])
        state = initial_supervisor_state("T")
        state["plan"] = [SubTask(id="s1", description="x", assignee="executor")]
        update = create_execute_node(llm, _registry(), max_steps=2)(state)
        assert "步数上限" in update["results"]["s1"]

    def test_researcher_does_not_see_finish_tool(self) -> None:
        """工具按声明过滤：researcher 绑定不含 finish。"""
        bound_names: list[str] = []

        class SpyModel(ScriptedChatModel):
            """记录 bind_tools 入参的替身。"""

            def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
                for t in tools:
                    fn = t.get("function", {}) if isinstance(t, dict) else {}
                    if fn.get("name"):
                        bound_names.append(str(fn["name"]))
                return super().bind_tools(tools, **kwargs)

        llm = SpyModel(scripted_responses=["ok"])
        state = initial_supervisor_state("T")
        state["plan"] = [SubTask(id="s1", description="调研资料", assignee="researcher")]
        create_execute_node(llm, _registry())(state)
        assert "python_sandbox" in bound_names
        assert "finish" not in bound_names

    def test_unknown_agent_falls_back_to_executor(self) -> None:
        """非法 assignee 回落到 executor，不抛异常。"""
        llm = ScriptedChatModel(scripted_responses=["兜底执行"])
        state = initial_supervisor_state("T")
        state["plan"] = [SubTask(id="s1", description="x", assignee="ghost")]
        update = create_execute_node(llm, _registry())(state)
        assert update["results"]["s1"] == "兜底执行"


class TestExecutePrompt:
    """execute Prompt 可单独测试。"""

    def test_contains_subtask_and_task(self) -> None:
        """包含总任务与子任务描述。"""
        st = SubTask(id="s9", description="写出要点")
        prompt = build_execute_prompt("总目标XYZ", st, None)
        assert "总目标XYZ" in prompt
        assert "s9" in prompt
        assert "写出要点" in prompt
        assert "回修反馈" not in prompt

    def test_contains_feedback_when_provided(self) -> None:
        """回修时包含反馈段。"""
        st = SubTask(id="s1", description="d")
        prompt = build_execute_prompt("T", st, "请补充数据来源")
        assert "回修反馈" in prompt
        assert "请补充数据来源" in prompt


class TestSystemPromptIsolation:
    """SubAgent system prompt 注入执行消息。"""

    def test_system_message_uses_spec_prompt(self) -> None:
        """run_subagent 首条消息为 spec.system_prompt。"""
        from agent_workbench.orchestrator.nodes import run_subagent

        captured: list[object] = []
        llm = ScriptedChatModel(scripted_responses=["done"])
        original = llm._generate  # type: ignore[method-assign]

        def spy(messages, **kwargs):  # type: ignore[no-untyped-def]
            if messages and isinstance(messages[0], SystemMessage):
                captured.append(messages[0])
            return original(messages, **kwargs)

        llm._generate = spy  # type: ignore[method-assign]
        run_subagent(
            RESEARCHER,
            llm,
            _registry(),
            subtask=SubTask(id="s1", description="调研"),
            task_description="T",
        )
        assert captured
        assert "调研员" in str(captured[0].content)
