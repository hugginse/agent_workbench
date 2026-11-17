"""plan 节点与 Prompt 单测：结构化拆解、fallback、规范化。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.orchestrator.nodes import PlanOutput, create_plan_node
from agent_workbench.orchestrator.prompts import build_plan_prompt
from agent_workbench.orchestrator.state import SubTask, initial_supervisor_state


def _plan_call(subtasks: list[dict[str, str]], call_id: str = "c_plan") -> AIMessage:
    """构造携带 PlanOutput 结构化 tool_call 的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": "PlanOutput", "args": {"subtasks": subtasks}, "id": call_id}],
    )


class TestBuildPlanPrompt:
    """Prompt 单独可测。"""

    def test_contains_task_description(self) -> None:
        """提示词包含原始任务文本。"""
        prompt = build_plan_prompt("对比三种向量库")
        assert "对比三种向量库" in prompt
        assert "子任务" in prompt

    def test_strips_whitespace(self) -> None:
        """任务描述两侧空白被去掉。"""
        prompt = build_plan_prompt("  任务A  ")
        assert "任务A" in prompt
        assert "  任务A  " not in prompt


class TestPlanNodeStructured:
    """结构化输出主路径。"""

    def test_decomposes_into_subtasks(self) -> None:
        """脚本 tool_calls → 解析为 SubTask 列表写入 plan。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _plan_call(
                    [
                        {"id": "s1", "description": "收集资料"},
                        {"id": "s2", "description": "撰写对比"},
                    ]
                )
            ]
        )
        node = create_plan_node(llm)
        state = initial_supervisor_state("写向量库对比报告")
        update = node(state)

        plan = update["plan"]
        assert isinstance(plan, list)
        assert len(plan) == 2
        assert plan[0].id == "s1"
        assert plan[0].description == "收集资料"
        assert plan[1].id == "s2"
        # 不修改入参 state
        assert state["plan"] == []

    def test_prompt_sent_to_model(self) -> None:
        """节点将含任务描述的 HumanMessage 传给模型。"""
        llm = ScriptedChatModel(scripted_responses=[_plan_call([{"id": "s1", "description": "x"}])])
        captured: list[object] = []

        original_invoke = llm._generate  # type: ignore[method-assign]

        def spy(messages: list[object], **kwargs: object):  # type: ignore[no-untyped-def]
            captured.extend(messages)
            return original_invoke(messages, **kwargs)  # type: ignore[arg-type]

        llm._generate = spy  # type: ignore[method-assign]
        node = create_plan_node(llm)
        node(initial_supervisor_state("独一无二的任务XYZ"))

        assert any(
            isinstance(m, HumanMessage) and "独一无二的任务XYZ" in str(m.content) for m in captured
        )


class TestPlanNodeFallback:
    """解析失败 / 空结果的受控 fallback。"""

    def test_unparseable_text_falls_back_to_single_subtask(self) -> None:
        """纯文本响应（structured 返回 None）→ 单子任务 fallback。"""
        llm = ScriptedChatModel(scripted_responses=["这不是结构化输出"])
        node = create_plan_node(llm)
        update = node(initial_supervisor_state("原始任务描述"))

        plan = update["plan"]
        assert len(plan) == 1
        assert plan[0].id == "s1"
        assert plan[0].description == "原始任务描述"

    def test_empty_subtasks_falls_back(self) -> None:
        """结构化但 subtasks 为空 → fallback。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "PlanOutput", "args": {"subtasks": []}, "id": "c0"}],
                )
            ]
        )
        node = create_plan_node(llm)
        plan = node(initial_supervisor_state("T"))["plan"]
        assert len(plan) == 1
        assert plan[0].description == "T"

    def test_fallback_does_not_raise(self) -> None:
        """任何异常路径都不向图外抛出。"""
        llm = ScriptedChatModel(scripted_responses=[])  # 默认话术
        node = create_plan_node(llm)
        plan = node(initial_supervisor_state("兜底"))["plan"]
        assert plan  # 至少一个子任务


class TestNormalizeSubtasks:
    """id 去重与数量上限。"""

    def test_duplicate_ids_rewritten(self) -> None:
        """重复 id 被改写为唯一。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _plan_call(
                    [
                        {"id": "s1", "description": "a"},
                        {"id": "s1", "description": "b"},
                    ]
                )
            ]
        )
        plan = create_plan_node(llm)(initial_supervisor_state("T"))["plan"]
        ids = [s.id for s in plan]
        assert len(ids) == len(set(ids))

    def test_max_eight_subtasks(self) -> None:
        """超过 8 个子任务被截断。"""
        many = [{"id": f"s{i}", "description": f"d{i}"} for i in range(1, 12)]
        llm = ScriptedChatModel(scripted_responses=[_plan_call(many)])
        plan = create_plan_node(llm)(initial_supervisor_state("T"))["plan"]
        assert len(plan) == 8

    def test_blank_id_assigned(self) -> None:
        """空白 id 自动补全。"""
        llm = ScriptedChatModel(scripted_responses=[_plan_call([{"id": "  ", "description": "x"}])])
        plan = create_plan_node(llm)(initial_supervisor_state("T"))["plan"]
        assert plan[0].id


class TestPlanOutputSchema:
    """结构化 schema 自身。"""

    def test_plan_output_accepts_subtasks(self) -> None:
        """PlanOutput 可容纳 SubTask。"""
        out = PlanOutput(subtasks=[SubTask(id="s1", description="d")])
        assert out.subtasks[0].id == "s1"

    def test_plan_output_default_empty(self) -> None:
        """默认空列表。"""
        assert PlanOutput().subtasks == []
