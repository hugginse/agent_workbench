"""Agent 消息与工具调用领域 schema。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = ["AgentMessage", "MessageRole", "ToolCall"]


class MessageRole(StrEnum):
    """消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """一次工具调用请求。

    Attributes:
        id: 调用标识（与 Tool 结果对应）。
        name: 工具名（须已在 ToolRegistry 注册，模型不得编造）。
        arguments: 工具入参（已按 schema 校验的结构化参数）。
    """

    id: str = Field(min_length=1, description="工具调用标识")
    name: str = Field(min_length=1, description="工具名")
    arguments: dict[str, object] = Field(default_factory=dict, description="结构化入参")


class AgentMessage(BaseModel):
    """Agent 之间流转的一条消息。

    Attributes:
        role: 角色（system / user / assistant / tool）。
        content: 消息正文。
        agent_name: 产出该消息的 SubAgent 名（如 researcher）；系统消息可为 None。
        tool_calls: assistant 消息携带的工具调用列表。
        tool_call_id: tool 结果消息对应的调用 id。
        timestamp: 消息时间（UTC）。
    """

    role: MessageRole = Field(description="消息角色")
    content: str = Field(default="", description="消息正文")
    agent_name: str | None = Field(default=None, description="产出消息的 SubAgent 名")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="工具调用列表")
    tool_call_id: str | None = Field(default=None, description="对应的工具调用 id")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="消息时间 UTC",
    )
