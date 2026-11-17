"""任务用例服务：创建、执行单 Agent 图、进度事件与查询。

编排与工具执行经依赖注入传入；本模块不含 FastAPI 类型。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent_workbench.domain.schemas import Task, TaskStatus
from agent_workbench.orchestrator.single_agent import create_single_agent_graph, initial_state
from agent_workbench.orchestrator.state import default_max_steps
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["TaskNotFoundError", "TaskService"]

_TERMINAL = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED})


class TaskNotFoundError(KeyError):
    """任务不存在。"""


class _Runtime:
    """单个任务的运行时：任务快照 + 进度事件 + 条件变量。"""

    __slots__ = ("cond", "events", "task")

    def __init__(self, task: Task) -> None:
        self.task = task
        self.events: list[dict[str, Any]] = []
        self.cond = asyncio.Condition()


class TaskService:
    """任务应用服务（内存态；后续可换持久化实现）。"""

    def __init__(
        self,
        llm: BaseChatModel,
        registry: ToolRegistry,
        *,
        max_steps: int = default_max_steps,
    ) -> None:
        """注入聊天模型与工具注册表。

        Args:
            llm: 聊天模型（真实或替身）。
            registry: 可用工具。
            max_steps: 单 Agent 工具轮次上限。
        """
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._runtime: dict[str, _Runtime] = {}
        # 持有后台任务引用，避免被 GC 提前回收（RUF006）
        self._background: set[asyncio.Task[None]] = set()

    async def create_task(self, description: str) -> Task:
        """创建任务并后台启动执行。

        Args:
            description: 任务描述（非空，由 API 层已校验）。

        Returns:
            已创建的 Task（status=running）。
        """
        task = Task(description=description, status=TaskStatus.RUNNING)
        self._runtime[task.id] = _Runtime(task)
        bg = asyncio.create_task(self._execute(task.id))
        self._background.add(bg)
        bg.add_done_callback(self._background.discard)
        return task

    def get_task(self, task_id: str) -> Task:
        """按 id 查询任务。

        Raises:
            TaskNotFoundError: 任务不存在。
        """
        return self._require(task_id).task

    def is_terminal(self, task_id: str) -> bool:
        """任务是否已到终态（completed / failed）。"""
        return self._require(task_id).task.status in _TERMINAL

    async def stream_events(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """异步产出进度事件：先重放已有事件，再增量推送，终态后结束。

        在条件锁内取批量事件，避免「读事件 → 等待」窗口丢通知的竞态。

        Raises:
            TaskNotFoundError: 任务不存在。
        """
        rt = self._require(task_id)
        cursor = 0
        while True:
            async with rt.cond:
                while cursor >= len(rt.events) and rt.task.status not in _TERMINAL:
                    await rt.cond.wait()
                batch = rt.events[cursor:]
                cursor = len(rt.events)
                terminal = rt.task.status in _TERMINAL
                # 终态后确保没有漏掉的尾部事件
                if terminal and cursor < len(rt.events):  # pragma: no cover — 防御
                    batch = rt.events[cursor:]
                    cursor = len(rt.events)
            for event in batch:
                yield event
            if terminal:
                return

    def _require(self, task_id: str) -> _Runtime:
        rt = self._runtime.get(task_id)
        if rt is None:
            msg = f"任务不存在: {task_id}"
            raise TaskNotFoundError(msg)
        return rt

    async def _append_event(self, task_id: str, event: dict[str, Any]) -> None:
        """追加事件并唤醒等待中的订阅者。"""
        rt = self._require(task_id)
        async with rt.cond:
            rt.events.append(event)
            rt.cond.notify_all()

    async def _set_terminal(self, task_id: str, *, status: TaskStatus, error: str | None) -> None:
        """写入终态任务快照并唤醒订阅者。"""
        rt = self._require(task_id)
        async with rt.cond:
            rt.task = rt.task.model_copy(update={"status": status, "error": error})
            rt.cond.notify_all()

    async def _execute(self, task_id: str) -> None:
        """在后台执行单 Agent 图并写回终态与事件。"""
        rt = self._require(task_id)
        description = rt.task.description
        await self._append_event(
            task_id,
            {"type": "task_started", "task_id": task_id, "description": description},
        )
        try:
            graph = create_single_agent_graph(
                self._llm,
                self._registry,
                max_steps=self._max_steps,
            )
            initial = initial_state(description, max_steps=self._max_steps)
            # 图执行为同步阻塞，放入线程池避免卡住事件循环
            final_state: dict[str, Any] = await asyncio.to_thread(graph.invoke, initial)
            answer = self._extract_answer(final_state)
            step_count = int(final_state.get("step_count", 0))
            await self._append_event(
                task_id,
                {
                    "type": "task_completed",
                    "task_id": task_id,
                    "answer": answer,
                    "step_count": step_count,
                },
            )
            await self._set_terminal(task_id, status=TaskStatus.COMPLETED, error=None)
        except Exception as exc:
            # 用户可见错误不携带堆栈/内部路径
            error_msg = f"任务执行失败: {type(exc).__name__}"
            await self._append_event(
                task_id,
                {"type": "task_failed", "task_id": task_id, "error": error_msg},
            )
            await self._set_terminal(task_id, status=TaskStatus.FAILED, error=error_msg)

    @staticmethod
    def _extract_answer(final_state: dict[str, Any]) -> str:
        """从终态消息中提取交付正文。

        优先 finish 工具的 final_delivery JSON，其次最后一条非空 AI 文本。
        """
        messages = final_state.get("messages") or []
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content
            if not isinstance(content, str) or "final_delivery" not in content:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                answer = payload.get("answer")
                if isinstance(answer, str) and answer:
                    return answer
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return str(msg.content)
        return ""
