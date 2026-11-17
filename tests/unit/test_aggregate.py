"""aggregate 节点与完成路径单测：正常交付 / 降级交付 / END 语义。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.orchestrator.nodes import (
    create_aggregate_node,
    create_review_router,
    is_degraded_delivery,
    join_results_deterministic,
)
from agent_workbench.orchestrator.prompts import build_aggregate_prompt
from agent_workbench.orchestrator.state import (
    ReviewVerdict,
    SubTask,
    initial_supervisor_state,
)


def _passed_state(*, with_llm_answer: bool = True) -> dict:
    """构造评审通过、具备产出的状态。"""
    state = initial_supervisor_state("写报告")
    state["plan"] = [
        SubTask(id="s1", description="收集资料", assignee="researcher"),
        SubTask(id="s2", description="撰写正文", assignee="executor"),
    ]
    state["results"] = {"s1": "要点A", "s2": "正文B"}
    state["verdict"] = ReviewVerdict(passed=True, feedback="合格")
    return state


def _degraded_state() -> dict:
    """构造超限降级状态。"""
    state = _passed_state()
    state["verdict"] = ReviewVerdict(
        passed=False,
        feedback="仍缺引用",
        target_subtask_id="s1",
    )
    state["repair_count"] = 2
    state["max_repair_attempts"] = 2
    return state


class TestAggregatePrompt:
    """aggregate Prompt 可单独测试。"""

    def test_contains_task_and_results(self) -> None:
        prompt = build_aggregate_prompt("总目标", {"s1": "内容1"})
        assert "总目标" in prompt
        assert "s1" in prompt
        assert "内容1" in prompt

    def test_empty_results_placeholder(self) -> None:
        assert "尚无产出" in build_aggregate_prompt("T", {})


class TestJoinDeterministic:
    """确定性拼接 fallback。"""

    def test_joins_in_plan_order(self) -> None:
        state = _passed_state()
        text = join_results_deterministic(state)
        assert text.index("s1") < text.index("s2")
        assert "要点A" in text
        assert "正文B" in text

    def test_missing_result_placeholder(self) -> None:
        state = _passed_state()
        state["results"] = {"s1": "只有s1"}
        text = join_results_deterministic(state)
        assert "未产出" in text

    def test_empty_state(self) -> None:
        state = initial_supervisor_state("T")
        assert "无子任务产出" in join_results_deterministic(state)

    def test_orphan_result_included(self) -> None:
        """plan 外的 results id 仍被拼入，不丢数据。"""
        state = initial_supervisor_state("T")
        state["results"] = {"sx": "孤儿结果"}
        text = join_results_deterministic(state)
        assert "孤儿结果" in text


class TestDegradedFlag:
    """降级判定。"""

    def test_pass_not_degraded(self) -> None:
        assert is_degraded_delivery(_passed_state()) is False

    def test_fail_is_degraded(self) -> None:
        assert is_degraded_delivery(_degraded_state()) is True

    def test_missing_verdict_is_degraded(self) -> None:
        state = _passed_state()
        state["verdict"] = None
        assert is_degraded_delivery(state) is True


class TestAggregateNode:
    """aggregate 节点产出 final_answer。"""

    def test_pass_uses_llm_answer(self) -> None:
        """通过路径：LLM 汇总文本作为 final_answer。"""
        llm = ScriptedChatModel(scripted_responses=["这是汇总后的最终交付"])
        state = _passed_state()
        update = create_aggregate_node(llm)(state)
        assert update["final_answer"] == "这是汇总后的最终交付"
        assert update["degraded"] is False
        assert state["final_answer"] == ""  # 不修改入参

    def test_pass_without_llm_uses_deterministic(self) -> None:
        """无 LLM 时用确定性拼接。"""
        state = _passed_state()
        update = create_aggregate_node(None)(state)
        assert "要点A" in update["final_answer"]
        assert update["degraded"] is False

    def test_pass_llm_failure_falls_back(self) -> None:
        """LLM 抛错时回退确定性拼接，不向上抛。"""

        class BoomModel(ScriptedChatModel):
            """始终抛错的替身。"""

            def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
                msg = "boom"
                raise ValueError(msg)

        state = _passed_state()
        update = create_aggregate_node(BoomModel(scripted_responses=["x"]))(state)
        assert "要点A" in update["final_answer"]
        assert update["degraded"] is False

    def test_degraded_adds_notice(self) -> None:
        """降级路径带前缀说明与评审意见。"""
        state = _degraded_state()
        update = create_aggregate_node(None)(state)
        assert update["degraded"] is True
        assert "降级交付" in update["final_answer"]
        assert "仍缺引用" in update["final_answer"]
        assert "2/2" in update["final_answer"]
        assert "要点A" in update["final_answer"]

    def test_degraded_without_verdict_notice(self) -> None:
        """无 verdict 的降级使用另一前缀。"""
        state = _passed_state()
        state["verdict"] = None
        update = create_aggregate_node(None)(state)
        assert update["degraded"] is True
        assert "缺少有效评审结论" in update["final_answer"]


class TestCompletionRouting:
    """完成路由：两条终止路径均汇入 aggregate（END 前一跳）。"""

    def test_pass_routes_to_aggregate(self) -> None:
        state = _passed_state()
        assert create_review_router()(state) == "aggregate"

    def test_degraded_fail_routes_to_aggregate(self) -> None:
        state = _degraded_state()
        assert create_review_router()(state) == "aggregate"

    def test_aggregate_is_terminal_producer(self) -> None:
        """aggregate 产出 final_answer，作为图完成标志。"""
        llm = ScriptedChatModel(scripted_responses=["完成"])
        update = create_aggregate_node(llm)(_passed_state())
        assert update["final_answer"]
        assert "degraded" in update

    def test_prompt_sent_as_human_message(self) -> None:
        """汇总时向模型发送 HumanMessage。"""
        seen: list[object] = []
        llm = ScriptedChatModel(scripted_responses=["ok"])
        original = llm._generate  # type: ignore[method-assign]

        def spy(messages, **kwargs):  # type: ignore[no-untyped-def]
            seen.extend(messages)
            return original(messages, **kwargs)

        llm._generate = spy  # type: ignore[method-assign]
        create_aggregate_node(llm)(_passed_state())
        assert any(isinstance(m, HumanMessage) for m in seen)
