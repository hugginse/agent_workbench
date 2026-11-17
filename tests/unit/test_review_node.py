"""review 节点与三路径条件边单测：通过 / 回修 / 超限降级。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.orchestrator.nodes import (
    create_assign_node,
    create_review_node,
    create_review_router,
)
from agent_workbench.orchestrator.prompts import build_review_prompt
from agent_workbench.orchestrator.state import (
    ReviewVerdict,
    SubTask,
    initial_supervisor_state,
)


def _verdict_call(verdict: ReviewVerdict, call_id: str = "c_r") -> AIMessage:
    """构造携带 ReviewVerdict 结构化 tool_call 的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ReviewVerdict",
                "args": verdict.model_dump(),
                "id": call_id,
            }
        ],
    )


def _state_with_results(*, repair_count: int = 0, max_attempts: int = 2) -> dict:
    """构造含 plan/results 的评审输入状态。"""
    state = initial_supervisor_state("写报告", max_repair_attempts=max_attempts)
    state["plan"] = [
        SubTask(id="s1", description="收集资料", assignee="researcher"),
        SubTask(id="s2", description="撰写正文", assignee="executor"),
    ]
    state["results"] = {"s1": "资料要点", "s2": "正文草稿"}
    state["repair_count"] = repair_count
    return state


class TestReviewPrompt:
    """review Prompt 单独可测。"""

    def test_contains_task_and_results(self) -> None:
        """包含总任务与各子任务产出。"""
        prompt = build_review_prompt("写报告", {"s1": "要点A"})
        assert "写报告" in prompt
        assert "s1" in prompt
        assert "要点A" in prompt

    def test_empty_results_placeholder(self) -> None:
        """无产出时有占位说明。"""
        prompt = build_review_prompt("T", {})
        assert "尚无产出" in prompt


class TestReviewNode:
    """review 节点结构化输出与 fallback。"""

    def test_pass_verdict_written(self) -> None:
        """脚本通过结论写入 verdict.passed=True，且清空 target。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _verdict_call(ReviewVerdict(passed=True, feedback="合格", target_subtask_id="s1"))
            ]
        )
        state = _state_with_results()
        update = create_review_node(llm)(state)
        assert update["verdict"].passed is True
        assert update["verdict"].target_subtask_id is None
        # 不修改入参
        assert state["verdict"] is None

    def test_fail_verdict_keeps_valid_target(self) -> None:
        """失败结论保留 plan 内合法 target。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _verdict_call(
                    ReviewVerdict(
                        passed=False,
                        feedback="缺数据",
                        target_subtask_id="s2",
                    )
                )
            ]
        )
        update = create_review_node(llm)(_state_with_results())
        assert update["verdict"].passed is False
        assert update["verdict"].target_subtask_id == "s2"
        assert update["verdict"].feedback == "缺数据"

    def test_fail_with_invalid_target_repaired(self) -> None:
        """非法 target 被改写为首个子任务 id。"""
        llm = ScriptedChatModel(
            scripted_responses=[
                _verdict_call(ReviewVerdict(passed=False, feedback="x", target_subtask_id="ghost"))
            ]
        )
        update = create_review_node(llm)(_state_with_results())
        assert update["verdict"].target_subtask_id == "s1"

    def test_unparseable_output_falls_back_to_fail(self) -> None:
        """解析失败 → 保守 fail（不静默通过）。"""
        llm = ScriptedChatModel(scripted_responses=["无法解析的评审文本"])
        update = create_review_node(llm)(_state_with_results())
        assert update["verdict"].passed is False
        assert update["verdict"].target_subtask_id == "s1"
        assert "解析失败" in update["verdict"].feedback

    def test_fail_without_plan_falls_back(self) -> None:
        """空 plan 时 fail fallback target 为 None（路由将降级）。"""
        llm = ScriptedChatModel(scripted_responses=["nope"])
        state = initial_supervisor_state("T")
        update = create_review_node(llm)(state)
        assert update["verdict"].passed is False
        assert update["verdict"].target_subtask_id is None


class TestThreePaths:
    """契约三路径：通过 / 回修 / 超限降级。"""

    def test_path_pass_routes_to_aggregate(self) -> None:
        """路径 1：review.pass → aggregate。"""
        route = create_review_router()
        state = _state_with_results()
        state["verdict"] = ReviewVerdict(passed=True, feedback="")
        assert route(state) == "aggregate"

    def test_path_fail_routes_to_assign_when_budget_left(self) -> None:
        """路径 2：fail & repair_count < max → assign。"""
        route = create_review_router()
        state = _state_with_results(repair_count=0, max_attempts=2)
        state["verdict"] = ReviewVerdict(passed=False, feedback="改", target_subtask_id="s1")
        assert route(state) == "assign"

    def test_path_fail_routes_to_aggregate_when_exhausted(self) -> None:
        """路径 3：fail & repair_count >= max → aggregate（超限降级）。"""
        route = create_review_router()
        state = _state_with_results(repair_count=2, max_attempts=2)
        state["verdict"] = ReviewVerdict(passed=False, feedback="仍不行", target_subtask_id="s1")
        assert route(state) == "aggregate"

    def test_no_verdict_routes_to_aggregate(self) -> None:
        """verdict 缺失时保守走 aggregate，不进入死循环。"""
        route = create_review_router()
        state = _state_with_results()
        assert route(state) == "aggregate"

    def test_default_max_attempts_is_two(self) -> None:
        """默认回修上限为 2。"""
        state = initial_supervisor_state("T")
        assert state["max_repair_attempts"] == 2


class TestRepairCountIncrement:
    """回修计数在 assign 回修入口递增（配合三路径封顶）。"""

    def test_first_assign_does_not_increment(self) -> None:
        """首轮 plan→assign（verdict 为 None）不计数。"""
        state = _state_with_results()
        state["verdict"] = None
        update = create_assign_node()(state)
        assert "repair_count" not in update
        assert state["repair_count"] == 0

    def test_repair_assign_increments(self) -> None:
        """回修进入 assign 时 repair_count +1。"""
        state = _state_with_results(repair_count=0)
        state["verdict"] = ReviewVerdict(passed=False, feedback="x", target_subtask_id="s1")
        update = create_assign_node()(state)
        assert update["repair_count"] == 1

    def test_full_loop_bounded_degrades(self) -> None:
        """模拟连续失败：回修两次后第三次必须降级（终止路径）。"""
        route = create_review_router()
        state = _state_with_results(repair_count=0, max_attempts=2)
        fail = ReviewVerdict(passed=False, feedback="bad", target_subtask_id="s1")
        assign = create_assign_node()

        # 第 1 次失败 → 回修
        state["verdict"] = fail
        assert route(state) == "assign"
        state.update(assign(state))  # count 0→1

        # 第 2 次失败 → 仍可回修
        state["verdict"] = fail
        assert route(state) == "assign"
        state.update(assign(state))  # count 1→2

        # 第 3 次失败 → 超限降级
        state["verdict"] = fail
        assert route(state) == "aggregate"
        assert state["repair_count"] == 2

    def test_loop_never_exceeds_max_attempts(self) -> None:
        """计数永不超过 max_repair_attempts。"""
        assign = create_assign_node()
        state = _state_with_results(repair_count=0, max_attempts=2)
        fail = ReviewVerdict(passed=False, feedback="b", target_subtask_id="s1")
        for _ in range(10):
            state["verdict"] = fail
            route = create_review_router()
            if route(state) == "aggregate":
                break
            state.update(assign(state))
            assert state["repair_count"] <= state["max_repair_attempts"]
        assert state["repair_count"] == 2
