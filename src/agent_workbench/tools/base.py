"""工具定义基类与公共类型。

约定：Pydantic schema 定义入参 → 注册表校验 → 执行；模型只产生
「工具名 + JSON 入参」，引用与参数合法性由本层把关。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

__all__ = ["BaseTool", "ToolError", "ToolSpec"]


class ToolError(Exception):
    """工具体系错误：未注册、重复注册或参数校验失败。"""


class ToolSpec(BaseModel):
    """工具的对外描述（供 LLM function-calling / SubAgent 装配）。

    Attributes:
        name: 工具名。
        description: 工具说明（含前置条件，可作 Skill 轻量描述）。
        parameters: JSON Schema 形式的入参定义。
    """

    name: str
    description: str
    parameters: dict[str, Any]


class BaseTool(ABC):
    """工具基类：名称、描述、参数 schema 与执行入口。

    子类必须提供类属性 `name`、`description`、`args_schema`，
    并实现 `execute`（入参已通过 schema 校验）。
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    def validate_args(self, arguments: dict[str, Any]) -> BaseModel:
        """按 `args_schema` 校验原始入参。

        Args:
            arguments: 模型或调用方给出的原始字典参数。

        Returns:
            校验通过的模型实例。

        Raises:
            ToolError: 校验失败时包装 `ValidationError` 并保留原因。
        """
        try:
            return self.args_schema.model_validate(arguments)
        except ValidationError as exc:
            msg = f"工具 {self.name} 入参校验失败: {exc.error_count()} 处错误"
            raise ToolError(msg) from exc

    @abstractmethod
    def execute(self, arguments: BaseModel) -> str:
        """执行已校验的工具逻辑。

        Args:
            arguments: 通过 `args_schema` 校验的入参模型。

        Returns:
            工具执行结果文本。
        """

    def run(self, arguments: dict[str, Any]) -> str:
        """校验入参并执行（调用方主入口）。

        Args:
            arguments: 原始字典参数。

        Returns:
            工具执行结果文本。

        Raises:
            ToolError: 入参不合法。
        """
        validated = self.validate_args(arguments)
        return self.execute(validated)

    def spec(self) -> ToolSpec:
        """导出工具描述与 JSON Schema（用于注册与模型装配）。"""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_schema.model_json_schema(),
        )
