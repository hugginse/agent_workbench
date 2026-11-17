"""finish 内置工具：结构化交付最终结果。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agent_workbench.tools.base import BaseTool

__all__ = ["FinishArgs", "FinishTool"]


class FinishArgs(BaseModel):
    """finish 工具入参。

    Attributes:
        answer: 最终交付正文（非空）。
        artifacts: 产出物标识列表（路径或产物 id）。
    """

    answer: str = Field(min_length=1, description="最终交付正文")
    artifacts: list[str] = Field(default_factory=list, description="产出物标识列表")


class FinishTool(BaseTool):
    """结构化交付工具：Agent 完成任务时调用，输出 JSON 交付包。"""

    name = "finish"
    description = (
        "任务完成时调用，提交结构化最终交付。"
        "answer 为完整交付正文；artifacts 为产出物标识列表。"
        "调用后表示任务结束，不得再发起其他工具调用。"
    )
    args_schema = FinishArgs

    def execute(self, arguments: BaseModel) -> str:
        """将交付内容序列化为 JSON 文本。

        Args:
            arguments: 已校验的 `FinishArgs`。

        Returns:
            含 type/answer/artifacts 的 JSON 字符串（UTF-8，中文不转义）。
        """
        assert isinstance(arguments, FinishArgs)
        payload: dict[str, Any] = {
            "type": "final_delivery",
            "answer": arguments.answer,
            "artifacts": arguments.artifacts,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
