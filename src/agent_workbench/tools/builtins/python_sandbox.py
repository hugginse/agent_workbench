"""python_sandbox 内置工具：无网络、带超时的子进程执行 Python 代码。"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from agent_workbench.tools.base import BaseTool

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "PythonSandboxArgs",
    "PythonSandboxTool",
]

DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 30.0

# 子进程入口：先禁用网络能力，再执行用户代码（-I 隔离模式）。
_SANDBOX_BOOTSTRAP = r"""
import socket as _socket
import sys as _sys


class _DeniedSocket(_socket.socket):
    def connect(self, address, *args, **kwargs):  # noqa: ANN001
        raise OSError("沙箱禁止网络访问")

    def connect_ex(self, address, *args, **kwargs):  # noqa: ANN001
        raise OSError("沙箱禁止网络访问")


def _deny(*_args: object, **_kwargs: object) -> None:
    raise OSError("沙箱禁止网络访问")


_socket.socket = _DeniedSocket
_socket.create_connection = _deny  # type: ignore[assignment]
_socket.getaddrinfo = _deny  # type: ignore[assignment]
"""

# 出网相关环境变量一律剥离（纵深防御；主防线为 socket 补丁）。
_NETWORK_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGSMITH_API_KEY",
)


class PythonSandboxArgs(BaseModel):
    """python_sandbox 入参。

    Attributes:
        code: 待执行的 Python 源码（非空）。
        timeout_seconds: 超时秒数，范围 (0, 30]。
    """

    code: str = Field(min_length=1, description="待执行 Python 源码")
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=MAX_TIMEOUT_SECONDS,
        description="超时秒数",
    )


class PythonSandboxTool(BaseTool):
    """在隔离子进程中执行 Python 代码。

    安全边界：
    - `subprocess` + 硬超时，超时强制终止；
    - 子进程 `python -I` 隔离模式；
    - 启动阶段打上 socket 补丁，禁用出网；
    - 剥离代理与常见 API Key 环境变量。
    """

    name = "python_sandbox"
    description = (
        "在无网络、带超时的隔离子进程中执行 Python 代码并返回 stdout/stderr。"
        "适用于计算、文本处理等本地逻辑；禁止依赖外网。"
        f"超时上限 {MAX_TIMEOUT_SECONDS:g} 秒。"
    )
    args_schema = PythonSandboxArgs

    def execute(self, arguments: BaseModel) -> str:
        """执行代码并返回结果文本（失败信息也以文本返回，供模型修正）。

        Args:
            arguments: 已校验的 `PythonSandboxArgs`。

        Returns:
            成功时为 stdout（若有 stderr 则附在后）；失败/超时为可读错误说明。
        """
        assert isinstance(arguments, PythonSandboxArgs)
        return self._run_code(arguments.code, arguments.timeout_seconds)

    def _run_code(self, code: str, timeout_seconds: float) -> str:
        """在临时目录写入脚本并以受限环境启动子进程。

        Args:
            code: 用户 Python 源码。
            timeout_seconds: 超时秒数。

        Returns:
            结果或错误的可读文本。
        """
        with tempfile.TemporaryDirectory(prefix="aw_sandbox_") as tmp:
            script_path = Path(tmp) / "main.py"
            # bootstrap + 用户代码：bootstrap 先禁网，再执行业务代码
            script_path.write_text(
                f"{_SANDBOX_BOOTSTRAP}\n{code}\n",
                encoding="utf-8",
            )
            env = self._sanitized_env()
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    env=env,
                    cwd=tmp,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return f"执行超时（{timeout_seconds:g}s），子进程已终止"
            except OSError as exc:
                # 无法启动解释器等系统错误
                return f"沙箱启动失败: {exc}"

        return self._format_result(completed.returncode, completed.stdout, completed.stderr)

    @staticmethod
    def _sanitized_env() -> dict[str, str]:
        """构建剥离出网与密钥变量的子进程环境。"""
        env = {k: v for k, v in os.environ.items() if k not in _NETWORK_ENV_KEYS}
        # 保证子进程能定位解释器与系统库
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMP", "TEMP"):
            if key in os.environ and key not in env:
                env[key] = os.environ[key]
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def _format_result(returncode: int, stdout: str, stderr: str) -> str:
        """按退出码整理 stdout/stderr 为单一文本。"""
        stdout = stdout or ""
        stderr = stderr or ""
        if returncode == 0:
            if stderr.strip():
                return f"{stdout}\n[stderr]\n{stderr}".strip()
            return stdout.strip() or "(无输出)"
        return f"执行失败(exit={returncode})\n{stderr.strip() or stdout.strip()}"
