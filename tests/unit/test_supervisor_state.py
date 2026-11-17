"""Supervisor 状态 schema 与回修计数单测。"""

from __future__ import annotations

import pytest

from agent_workbench.orchestrator.state import (
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    MAX_REPAIR_ATTEMPTS_LIMIT,
    ReviewVerdict,
    SubTask,
    SupervisorState,
    initial_supervisor_state,
    next_repair_count,
)


class TestInitialSupervisorState:
    """初始状态构造与默认值。"""

    def test_defaults(self) -> None:
        """默认 repair_count=0、上限 2、plan/results 空、verdict None。"""
        state = initial_supervisor_state("调研方案对比")
        assert state["task_description"] == "调研方案对比"
        assert state["plan"] == []
        assert state["results"] == {}
        assert state["verdict"] is None
        assert state["repair_count"] == 0
        assert state["max_repair_attempts"] == DEFAULT_MAX_REPAIR_ATTEMPTS
        assert state["token_usage"] == {"input": 0, "output": 0, "total": 0}

    def test_empty_description_rejected(self) -> None:
        """空任务描述必须拒绝。"""
        with pytest.raises(ValueError, match="task_description"):
            initial_supervisor_state("   ")

    @pytest.mark.parametrize("limit", [1, 2, 3, 4, 5])
    def test_repair_limit_in_range(self, limit: int) -> None:
        """1..5 的回修上限合法。"""
        state = initial_supervisor_state("t", max_repair_attempts=limit)
        assert state["max_repair_attempts"] == limit

    @pytest.mark.parametrize("limit", [0, -1, 6, 99])
    def test_repair_limit_out_of_range_rejected(self, limit: int) -> None:
        """超出 [1,5] 的回修上限必须拒绝。"""
        with pytest.raises(ValueError, match="max_repair_attempts"):
            initial_supervisor_state("t", max_repair_attempts=limit)

    def test_limit_constant_matches_contract(self) -> None:
        """契约常量：默认 2、硬上限 5。"""
        assert DEFAULT_MAX_REPAIR_ATTEMPTS == 2
        assert MAX_REPAIR_ATTEMPTS_LIMIT == 5


class TestSubTaskAndVerdict:
    """SubTask / ReviewVerdict 模型。"""

    def test_sub_task_minimal(self) -> None:
        """id + description 即可，assignee 可空。"""
        st = SubTask(id="s1", description="收集资料")
        assert st.assignee is None

    def test_sub_task_rejects_empty_id(self) -> None:
        """空 id 被拒绝。"""
        with pytest.raises(ValueError):
            SubTask(id="", description="x")

    def test_verdict_pass(self) -> None:
        """通过结论无回修目标。"""
        v = ReviewVerdict(passed=True, feedback="合格")
        assert v.passed is True
        assert v.target_subtask_id is None

    def test_verdict_fail_with_target(self) -> None:
        """失败结论可带定向回修目标。"""
        v = ReviewVerdict(passed=False, feedback="缺引用", target_subtask_id="s2")
        assert v.target_subtask_id == "s2"


class TestRepairCounter:
    """回修计数与封顶语义。"""

    def test_next_repair_count_increments(self) -> None:
        """每次 review.fail 计数 +1。"""
        state = initial_supervisor_state("t")
        assert next_repair_count(state) == 1
        state2: SupervisorState = {**state, "repair_count": 1}
        assert next_repair_count(state2) == 2

    def test_can_continue_below_limit(self) -> None:
        """repair_count < max 时仍可回修（条件边输入）。"""
        state = initial_supervisor_state("t", max_repair_attempts=2)
        assert state["repair_count"] < state["max_repair_attempts"]
        state = {**state, "repair_count": 1}
        assert state["repair_count"] < state["max_repair_attempts"]

    def test_must_degrade_at_limit(self) -> None:
        """repair_count >= max 时必须走降级路径。"""
        state = initial_supervisor_state("t", max_repair_attempts=2)
        state = {**state, "repair_count": 2}
        assert state["repair_count"] >= state["max_repair_attempts"]

    def test_state_is_typed_mapping(self) -> None:
        """关键字段均在状态中（编排表要求）。"""
        state = initial_supervisor_state("t")
        for key in (
            "plan",
            "results",
            "verdict",
            "repair_count",
            "token_usage",
            "max_repair_attempts",
            "task_description",
        ):
            assert key in state
