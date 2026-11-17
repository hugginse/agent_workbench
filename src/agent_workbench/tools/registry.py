"""ToolRegistry：工具注册、查询、schema 校验与执行分发。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_workbench.tools.base import BaseTool, ToolError, ToolSpec

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """工具注册表。

    职责：注册（拒绝非法/重复）→ 按名获取 → 校验入参 → 执行。
    不持有全局单例；由调用方构造并注入。
    """

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具实例。

        Args:
            tool: 满足 BaseTool 契约的工具。

        Raises:
            ToolError: 名称为空、描述为空、args_schema 缺失或名称已存在。
        """
        name = getattr(tool, "name", "")
        description = getattr(tool, "description", "")
        args_schema = getattr(tool, "args_schema", None)

        if not isinstance(name, str) or not name.strip():
            msg = "工具注册失败：name 必须为非空字符串"
            raise ToolError(msg)
        if not isinstance(description, str) or not description.strip():
            msg = f"工具注册失败：{name} 的 description 必须为非空字符串"
            raise ToolError(msg)
        if not (isinstance(args_schema, type) and issubclass(args_schema, BaseModel)):
            msg = f"工具注册失败：{name} 的 args_schema 必须是 Pydantic BaseModel 子类"
            raise ToolError(msg)
        if name in self._tools:
            msg = f"工具注册失败：{name} 已存在"
            raise ToolError(msg)

        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """按名称注销工具。

        Raises:
            ToolError: 名称未注册。
        """
        if name not in self._tools:
            msg = f"工具未注册：{name}"
            raise ToolError(msg)
        del self._tools[name]

    def get(self, name: str) -> BaseTool:
        """按名称获取工具。

        Raises:
            ToolError: 名称未注册。
        """
        tool = self._tools.get(name)
        if tool is None:
            msg = f"工具未注册：{name}"
            raise ToolError(msg)
        return tool

    def __contains__(self, name: object) -> bool:
        """支持 `name in registry` 判断。"""
        return isinstance(name, str) and name in self._tools

    @property
    def names(self) -> list[str]:
        """已注册工具名（排序后返回）。"""
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """导出全部工具描述（供 LLM / SubAgent 装配）。"""
        return [self._tools[n].spec() for n in self.names]

    def validate(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        """校验指定工具的入参（不执行）。

        Raises:
            ToolError: 工具未注册或入参不合法。
        """
        return self.get(name).validate_args(arguments)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """校验并执行工具。

        Args:
            name: 已注册工具名。
            arguments: 原始字典入参。

        Returns:
            工具执行结果文本。

        Raises:
            ToolError: 工具未注册或入参不合法。
        """
        return self.get(name).run(arguments)
