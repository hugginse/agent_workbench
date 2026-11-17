"""应用入口：`uv run uvicorn agent_workbench.main:app --reload`。

装配在 `create_app` 内完成；模块导入仅构造默认替身/配置，不建网络连接。
默认 Supervisor（multi）图 + SQLite checkpointer；`mode="single"` 供评测基线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_workbench.api.endpoints.tasks import router as tasks_router
from agent_workbench.application.services.task import TaskMode, TaskService
from agent_workbench.infrastructure.llm import get_chat_model
from agent_workbench.orchestrator.builder import open_sqlite_checkpointer
from agent_workbench.orchestrator.state import default_max_steps
from agent_workbench.tools import ToolRegistry
from agent_workbench.tools.builtins import FinishTool, PythonSandboxTool
from config.settings import Settings, get_settings

__all__ = ["build_default_registry", "create_app"]

# 默认断点库位置（已 .gitignore）
DEFAULT_CHECKPOINT_DB = Path(".data") / "checkpoints.db"


def build_default_registry() -> ToolRegistry:
    """构建内置工具注册表（finish + python_sandbox）。"""
    registry = ToolRegistry()
    registry.register(FinishTool())
    registry.register(PythonSandboxTool())
    return registry


def create_app(
    settings: Settings | None = None,
    *,
    max_steps: int = default_max_steps,
    mode: TaskMode = "multi",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    checkpoint_db: str | Path | None = None,
) -> FastAPI:
    """创建 FastAPI 应用（依赖注入，无全局可变单例）。

    Args:
        settings: 显式配置；None 时从环境读取。
        max_steps: 工具轮次上限。
        mode: `multi`=Supervisor（API 默认）；`single`=单 Agent（评测）。
        checkpointer: 显式检查点；为 None 且 multi 时按 checkpoint_db 打开。
        checkpoint_db: SQLite 路径；默认 `.data/checkpoints.db`。

    Returns:
        已挂载任务端点的 FastAPI 实例。
    """
    if settings is None:
        settings = get_settings()

    llm = get_chat_model(settings)
    registry = build_default_registry()

    resolved_checkpointer: BaseCheckpointSaver[Any] | None = None
    if mode == "multi":
        if checkpointer is not None:
            resolved_checkpointer = checkpointer
        else:
            db_path = Path(checkpoint_db) if checkpoint_db is not None else DEFAULT_CHECKPOINT_DB
            db_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_checkpointer = open_sqlite_checkpointer(db_path)

    task_service = TaskService(
        llm,
        registry,
        max_steps=max_steps,
        mode=mode,
        checkpointer=resolved_checkpointer,
    )

    app = FastAPI(
        title="agent_workbench",
        version="0.1.0",
        description="多智能体任务执行与评测平台",
    )
    app.state.settings = settings
    app.state.task_service = task_service
    app.include_router(tasks_router)
    return app


app = create_app()
