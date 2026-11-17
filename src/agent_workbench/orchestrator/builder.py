"""Supervisor 图装配：节点接线 + SQLite checkpointer 断点续跑。

流程：
    START → plan → assign → execute → review →
      pass                    → aggregate → END
      fail & repair < max     → assign（定向回修）
      fail & repair >= max    → aggregate（降级交付）→ END
"""

from __future__ import annotations

from pathlib import Path
from sqlite3 import Connection, connect
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_workbench.orchestrator.nodes import (
    create_aggregate_node,
    create_assign_node,
    create_execute_node,
    create_plan_node,
    create_review_node,
    create_review_router,
)
from agent_workbench.orchestrator.state import SupervisorState, default_max_steps
from agent_workbench.tools.registry import ToolRegistry

__all__ = [
    "build_supervisor_graph",
    "open_sqlite_checkpointer",
]


def open_sqlite_checkpointer(db_path: str | Path) -> SqliteSaver:
    """打开 SQLite checkpointer（文件可跨进程/重启复用）。

    Args:
        db_path: SQLite 文件路径；`:memory:` 为内存库（仅测试）。

    Returns:
        已就绪的 SqliteSaver。
    """
    conn: Connection = connect(str(db_path), check_same_thread=False)
    # 确保 WAL 等基础设置不影响断点读写
    conn.execute("PRAGMA journal_mode=WAL;")
    return SqliteSaver(conn)


def build_supervisor_graph(
    llm: BaseChatModel,
    registry: ToolRegistry,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_steps: int = default_max_steps,
) -> CompiledStateGraph[SupervisorState, None, Any, Any]:
    """编译 Supervisor 图。

    Args:
        llm: 聊天模型（plan/review/aggregate/execute 共用）。
        registry: 工具注册表。
        checkpointer: 检查点存储；None 表示无断点（不持久化）。
        max_steps: 每个 SubAgent 工具循环上限。

    Returns:
        可 invoke 的编译图；传入 checkpointer 后支持 thread_id 断点续跑。
    """
    builder: StateGraph[SupervisorState] = StateGraph(SupervisorState)

    # LangGraph add_node 重载与工厂返回的 Callable 签名不完全对齐，
    # 在装配边界收窄为 Any（节点函数本身仍由 mypy 按工厂定义检查）。
    builder.add_node("plan", cast(Any, create_plan_node(llm)))
    builder.add_node("assign", cast(Any, create_assign_node()))
    builder.add_node(
        "execute",
        cast(Any, create_execute_node(llm, registry, max_steps=max_steps)),
    )
    builder.add_node("review", cast(Any, create_review_node(llm)))
    builder.add_node("aggregate", cast(Any, create_aggregate_node(llm)))

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "assign")
    builder.add_edge("assign", "execute")
    builder.add_edge("execute", "review")

    # 三路径条件边：pass/超限 → aggregate；可回修 → assign
    builder.add_conditional_edges(
        "review",
        create_review_router(),
        {"assign": "assign", "aggregate": "aggregate"},
    )
    # 完成终点
    builder.add_edge("aggregate", END)

    return builder.compile(checkpointer=checkpointer)
