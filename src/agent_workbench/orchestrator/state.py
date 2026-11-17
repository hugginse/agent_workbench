"""单 Agent 图状态定义。"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

__all__ = ["SingleAgentState", "default_max_steps"]

# 工具循环默认上限（单 Agent；Supervisor 回修上限见 Phase 2）
default_max_steps = 8


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
