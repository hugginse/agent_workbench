"""编排状态定义：单 Agent 循环 + Supervisor 回修闭环。"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

__all__ = [
    "DEFAULT_MAX_REPAIR_ATTEMPTS",
    "MAX_REPAIR_ATTEMPTS_LIMIT",
    "ReviewVerdict",
    "SingleAgentState",
    "SubTask",
    "SupervisorState",
    "default_max_steps",
    "initial_supervisor_state",
    "next_repair_count",
]

# 单 Agent 工具循环默认上限
default_max_steps = 8

# Supervisor 回修：默认 2，硬上限 5（契约红线）
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
MAX_REPAIR_ATTEMPTS_LIMIT = 5


class SingleAgentState(TypedDict):
    """单 Agent ReAct 循环状态。

    Attributes:
        messages: 消息历史（经 add_messages 归并）。
        step_count: 已执行的工具轮次数（每进入 tools 节点 +1）。
        max_steps: 工具轮次硬上限，达到后必须走终止路径。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int
    max_steps: int


class SubTask(BaseModel):
    """Supervisor 拆解出的一个子任务。

    Attributes:
        id: 子任务标识（须与执行/评审结果对应，模型不得编造执行外 id）。
        description: 子任务描述。
        assignee: 期望执行角色（如 researcher / executor）；plan 阶段可空。
    """

    id: str = Field(min_length=1, description="子任务标识")
    description: str = Field(min_length=1, description="子任务描述")
    assignee: str | None = Field(default=None, description="期望执行角色")


class ReviewVerdict(BaseModel):
    """评审结论（驱动条件边：通过 / 定向回修 / 超限降级）。

    Attributes:
        passed: 是否通过。
        feedback: 评审意见（回修时作为定向输入）。
        target_subtask_id: 回修目标子任务；整体通过或降级时为 None。
    """

    passed: bool = Field(description="是否通过")
    feedback: str = Field(default="", description="评审意见")
    target_subtask_id: str | None = Field(default=None, description="回修目标子任务 id")


class SupervisorState(TypedDict):
    """Supervisor 多智能体图状态。

    Attributes:
        task_description: 原始任务描述（plan 输入）。
        plan: 子任务列表（plan 节点产出，可被回修更新）。
        results: 子任务 id → 执行结果正文。
        verdict: 最近一次评审结论；尚未评审为 None。
        repair_count: 已发生的回修次数（review.fail 时递增）。
        max_repair_attempts: 回修上限（默认 2，硬上限 5）。
        token_usage: 累计 token：input / output / total。
    """

    task_description: str
    plan: list[SubTask]
    results: dict[str, str]
    verdict: ReviewVerdict | None
    repair_count: int
    max_repair_attempts: int
    token_usage: dict[str, int]


def _validate_repair_limit(max_repair_attempts: int) -> int:
    """校验回修上限：必须在 [1, MAX_REPAIR_ATTEMPTS_LIMIT]。

    Raises:
        ValueError: 超出契约允许范围。
    """
    if not 1 <= max_repair_attempts <= MAX_REPAIR_ATTEMPTS_LIMIT:
        msg = (
            f"max_repair_attempts 必须在 1..{MAX_REPAIR_ATTEMPTS_LIMIT}，当前 {max_repair_attempts}"
        )
        raise ValueError(msg)
    return max_repair_attempts


def initial_supervisor_state(
    task_description: str,
    *,
    max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
) -> SupervisorState:
    """构造 Supervisor 初始状态。

    Args:
        task_description: 原始任务描述（非空）。
        max_repair_attempts: 回修上限，默认 2，上限 5。

    Returns:
        初始化完毕的 SupervisorState。

    Raises:
        ValueError: 描述为空或回修上限非法。
    """
    if not task_description.strip():
        msg = "task_description 不得为空"
        raise ValueError(msg)
    limit = _validate_repair_limit(max_repair_attempts)
    return {
        "task_description": task_description,
        "plan": [],
        "results": {},
        "verdict": None,
        "repair_count": 0,
        "max_repair_attempts": limit,
        "token_usage": {"input": 0, "output": 0, "total": 0},
    }


def next_repair_count(state: SupervisorState) -> int:
    """计算下一次 repair_count（在 review.fail 边调用，不修改入参）。

    计数只增不减；达到上限后由条件边走降级路径，本函数仍返回 +1
    供状态写入，终止判断以 `repair_count >= max_repair_attempts` 为准。

    Args:
        state: 当前 Supervisor 状态。

    Returns:
        repair_count + 1。
    """
    return int(state["repair_count"]) + 1
