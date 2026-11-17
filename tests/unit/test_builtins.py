"""内置工具 finish / python_sandbox 单测。"""

from __future__ import annotations

import json

import pytest

from agent_workbench.tools import ToolError, ToolRegistry
from agent_workbench.tools.builtins import FinishTool, PythonSandboxTool


class TestFinishTool:
    """finish：结构化交付。"""

    def test_execute_returns_json_delivery(self) -> None:
        """合法入参返回含 answer/artifacts 的 JSON。"""
        tool = FinishTool()
        raw = tool.run({"answer": "结论如下", "artifacts": ["a.md"]})
        payload = json.loads(raw)
        assert payload["type"] == "final_delivery"
        assert payload["answer"] == "结论如下"
        assert payload["artifacts"] == ["a.md"]

    def test_artifacts_default_empty(self) -> None:
        """artifacts 可省略，默认空列表。"""
        raw = FinishTool().run({"answer": "仅正文"})
        payload = json.loads(raw)
        assert payload["artifacts"] == []

    def test_empty_answer_rejected(self) -> None:
        """空 answer 必须被 schema 拒绝。"""
        with pytest.raises(ToolError, match="校验失败"):
            FinishTool().run({"answer": ""})

    def test_registers_in_registry(self) -> None:
        """可注册进 ToolRegistry 并导出 spec。"""
        r = ToolRegistry()
        r.register(FinishTool())
        assert "finish" in r
        assert r.specs()[0].name == "finish"


class TestPythonSandboxTool:
    """python_sandbox：超时 + 无网络 + 基本执行。"""

    @pytest.fixture
    def tool(self) -> PythonSandboxTool:
        """沙箱工具实例。"""
        return PythonSandboxTool()

    def test_runs_simple_code(self, tool: PythonSandboxTool) -> None:
        """本地计算可执行并返回 stdout。"""
        out = tool.run({"code": "print(1 + 1)"})
        assert out.strip() == "2"

    def test_syntax_error_returns_message(self, tool: PythonSandboxTool) -> None:
        """语法错误以文本返回，便于模型修正（非裸异常抛出）。"""
        out = tool.run({"code": "def broken(: pass"})
        assert "执行失败" in out
        assert "SyntaxError" in out

    def test_timeout_enforced(self, tool: PythonSandboxTool) -> None:
        """超时必须终止并返回超时说明。"""
        out = tool.run(
            {
                "code": "import time; time.sleep(30)",
                "timeout_seconds": 1.0,
            }
        )
        assert "超时" in out

    def test_network_blocked_socket(self, tool: PythonSandboxTool) -> None:
        """socket 连接必须被沙箱拒绝。"""
        code = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('example.com', 80), timeout=1)\n"
            "    print('NETWORK_OPEN')\n"
            "except OSError as e:\n"
            "    print(f'BLOCKED:{e}')\n"
        )
        out = tool.run({"code": code, "timeout_seconds": 10.0})
        assert "NETWORK_OPEN" not in out
        assert "BLOCKED" in out
        assert "禁止网络" in out

    def test_network_blocked_urllib(self, tool: PythonSandboxTool) -> None:
        """urllib 出网路径同样被拦截。"""
        code = (
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://example.com', timeout=2)\n"
            "    print('NETWORK_OPEN')\n"
            "except Exception as e:\n"
            "    print(f'BLOCKED:{type(e).__name__}')\n"
        )
        out = tool.run({"code": code, "timeout_seconds": 15.0})
        assert "NETWORK_OPEN" not in out
        assert "BLOCKED" in out

    def test_timeout_out_of_range_rejected(self, tool: PythonSandboxTool) -> None:
        """timeout 超出 (0, 30] 被 schema 拒绝。"""
        with pytest.raises(ToolError, match="校验失败"):
            tool.validate_args({"code": "pass", "timeout_seconds": 0})
        with pytest.raises(ToolError, match="校验失败"):
            tool.validate_args({"code": "pass", "timeout_seconds": 999})

    def test_empty_code_rejected(self, tool: PythonSandboxTool) -> None:
        """空代码被拒绝。"""
        with pytest.raises(ToolError, match="校验失败"):
            tool.run({"code": ""})

    def test_registers_in_registry(self) -> None:
        """可注册进 ToolRegistry。"""
        r = ToolRegistry()
        r.register(PythonSandboxTool())
        assert "python_sandbox" in r
