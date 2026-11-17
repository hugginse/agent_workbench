"""ToolRegistry 与 BaseTool 单测：注册 / 校验 / 拒绝非法工具。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from agent_workbench.tools import BaseTool, ToolError, ToolRegistry, ToolSpec


class EchoArgs(BaseModel):
    """echo 工具入参。"""

    text: str = Field(min_length=1, description="回显文本")


class EchoTool(BaseTool):
    """测试用回显工具。"""

    name = "echo"
    description = "原样返回文本"
    args_schema = EchoArgs

    def execute(self, arguments: BaseModel) -> str:
        """返回校验后的 text。"""
        assert isinstance(arguments, EchoArgs)
        return arguments.text


class CountArgs(BaseModel):
    """计数工具入参（用于多工具场景）。"""

    n: int = Field(ge=0, description="计数上限")


class CountTool(BaseTool):
    """测试用计数工具。"""

    name = "count"
    description = "输出 0..n-1"
    args_schema = CountArgs

    def execute(self, arguments: BaseModel) -> str:
        """生成计数序列。"""
        assert isinstance(arguments, CountArgs)
        return ",".join(str(i) for i in range(arguments.n))


class NoSchemaTool(BaseTool):
    """故意缺失 args_schema 的非法工具。"""

    name = "bad"
    description = "缺少 schema"
    # 不定义 args_schema —— 注册时应被拒绝

    def execute(self, arguments: BaseModel) -> str:  # pragma: no cover - 不可达
        """占位实现。"""
        return ""


class EmptyNameTool(BaseTool):
    """名称为空的非法工具。"""

    name = ""
    description = "空名"
    args_schema = EchoArgs

    def execute(self, arguments: BaseModel) -> str:  # pragma: no cover
        """占位实现。"""
        return ""


@pytest.fixture
def registry() -> ToolRegistry:
    """已注册 echo 与 count 的注册表。"""
    r = ToolRegistry()
    r.register(EchoTool())
    r.register(CountTool())
    return r


class TestRegistration:
    """注册成功与拒绝路径。"""

    def test_register_and_contains(self) -> None:
        """注册后可查询。"""
        r = ToolRegistry()
        assert "echo" not in r
        r.register(EchoTool())
        assert "echo" in r
        assert r.names == ["echo"]

    def test_register_duplicate_rejected(self) -> None:
        """重复注册同名工具必须拒绝。"""
        r = ToolRegistry()
        r.register(EchoTool())
        with pytest.raises(ToolError, match="已存在"):
            r.register(EchoTool())

    def test_register_empty_name_rejected(self) -> None:
        """空 name 必须拒绝。"""
        r = ToolRegistry()
        with pytest.raises(ToolError, match="name"):
            r.register(EmptyNameTool())

    def test_register_missing_schema_rejected(self) -> None:
        """args_schema 非 BaseModel 子类（或缺失）必须拒绝。"""
        r = ToolRegistry()
        with pytest.raises(ToolError, match="args_schema"):
            r.register(NoSchemaTool())

    def test_unregister(self) -> None:
        """注销后不可再获取。"""
        r = ToolRegistry()
        r.register(EchoTool())
        r.unregister("echo")
        assert "echo" not in r
        with pytest.raises(ToolError, match="未注册"):
            r.get("echo")


class TestValidation:
    """入参 schema 校验。"""

    def test_valid_args_pass(self, registry: ToolRegistry) -> None:
        """合法参数校验通过并返回模型。"""
        result = registry.validate("echo", {"text": "你好"})
        assert isinstance(result, EchoArgs)
        assert result.text == "你好"

    def test_invalid_args_rejected(self, registry: ToolRegistry) -> None:
        """缺字段 / 空字符串必须被拒绝。"""
        with pytest.raises(ToolError, match="校验失败"):
            registry.validate("echo", {})
        with pytest.raises(ToolError, match="校验失败"):
            registry.validate("echo", {"text": ""})

    def test_wrong_type_rejected(self, registry: ToolRegistry) -> None:
        """类型不符必须被拒绝。"""
        with pytest.raises(ToolError, match="校验失败"):
            registry.validate("count", {"n": "not-int"})


class TestExecution:
    """执行分发与拒绝路径。"""

    def test_execute_registered_tool(self, registry: ToolRegistry) -> None:
        """已注册工具可执行并返回结果。"""
        assert registry.execute("echo", {"text": "abc"}) == "abc"
        assert registry.execute("count", {"n": 3}) == "0,1,2"

    def test_execute_unregistered_rejected(self, registry: ToolRegistry) -> None:
        """未注册工具必须拒绝执行。"""
        with pytest.raises(ToolError, match="未注册"):
            registry.execute("nope", {})

    def test_execute_invalid_args_rejected(self, registry: ToolRegistry) -> None:
        """执行路径同样做 schema 校验。"""
        with pytest.raises(ToolError, match="校验失败"):
            registry.execute("echo", {"text": ""})

    def test_get_returns_same_instance(self, registry: ToolRegistry) -> None:
        """get 返回注册时的同一实例。"""
        assert registry.get("echo") is registry.get("echo")


class TestSpec:
    """工具描述导出。"""

    def test_spec_shape(self) -> None:
        """spec 含 name / description / JSON Schema。"""
        spec = EchoTool().spec()
        assert isinstance(spec, ToolSpec)
        assert spec.name == "echo"
        assert spec.description
        assert spec.parameters["type"] == "object"
        assert "text" in spec.parameters["properties"]

    def test_registry_specs_sorted(self, registry: ToolRegistry) -> None:
        """specs 按工具名排序，便于稳定断言。"""
        specs = registry.specs()
        assert [s.name for s in specs] == ["count", "echo"]

    def test_specs_do_not_leak_secrets(self) -> None:
        """导出的参数 schema 只含 JSON Schema 元数据，不含内部字段。"""
        params: dict[str, Any] = EchoTool().spec().parameters
        allowed = {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "title",
            "description",
        }
        assert set(params).issubset(allowed)
