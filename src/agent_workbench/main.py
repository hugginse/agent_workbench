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
from agent_workbench.memory.long_term import LongTermMemoryRepository, SqliteLongTermMemory
from agent_workbench.orchestrator.builder import open_sqlite_checkpointer
from agent_workbench.orchestrator.state import default_max_steps
from agent_workbench.tools import ToolRegistry
from agent_workbench.tools.builtins import (
    FinishTool,
    MemoryRecallTool,
    PythonSandboxTool,
    RetrieveTool,
)
from config.settings import Settings, get_settings

__all__ = ["build_default_registry", "create_app"]

# 默认断点 / 长期记忆库位置（已 .gitignore）
DEFAULT_CHECKPOINT_DB = Path(".data") / "checkpoints.db"
DEFAULT_MEMORY_DB = Path(".data") / "memory.db"


def build_default_registry(
    memory: LongTermMemoryRepository | None = None,
    *,
    vector_store: Any = None,
    embedder: Any = None,
) -> ToolRegistry:
    """构建内置工具注册表。

    Args:
        memory: 非 None 时注册 memory_recall（供跨会话复盘检索）。
        vector_store: 非 None 时注册 retrieve。
        embedder: retrieve 使用的 Embedding；None 时工具内用 FakeEmbeddings。

    Returns:
        含 finish + python_sandbox（+ 可选 memory_recall / retrieve）的注册表。
    """
    registry = ToolRegistry()
    registry.register(FinishTool())
    registry.register(PythonSandboxTool())
    if memory is not None:
        registry.register(MemoryRecallTool(memory))
    if vector_store is not None:
        registry.register(RetrieveTool(vector_store, embedder))
    return registry


def create_app(
    settings: Settings | None = None,
    *,
    max_steps: int = default_max_steps,
    mode: TaskMode = "multi",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    checkpoint_db: str | Path | None = None,
    memory: LongTermMemoryRepository | None = None,
    memory_db: str | Path | None = None,
) -> FastAPI:
    """创建 FastAPI 应用（依赖注入，无全局可变单例）。

    Args:
        settings: 显式配置；None 时从环境读取。
        max_steps: 工具轮次上限。
        mode: `multi`=Supervisor（API 默认）；`single`=单 Agent（评测）。
        checkpointer: 显式检查点；为 None 且 multi 时按 checkpoint_db 打开。
        checkpoint_db: SQLite 路径；默认 `.data/checkpoints.db`。
        memory: 显式长期记忆；为 None 时按 memory_db 打开。
        memory_db: 长期记忆 SQLite 路径；默认 `.data/memory.db`。

    Returns:
        已挂载任务端点的 FastAPI 实例。
    """
    if settings is None:
        settings = get_settings()

    llm = get_chat_model(settings)

    if memory is None:
        mem_path = Path(memory_db) if memory_db is not None else DEFAULT_MEMORY_DB
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        memory = SqliteLongTermMemory(mem_path)
    registry = build_default_registry(memory=memory)

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
        memory=memory,
    )

    app = FastAPI(
        title="agent_workbench",
        version="0.1.0",
        description="多智能体任务执行与评测平台",
    )
    app.state.settings = settings
    app.state.task_service = task_service
    app.state.memory = memory
    app.include_router(tasks_router)
    return app


app = create_app()
