"""任务用例服务：创建、执行编排图、进度事件与查询。

默认 Supervisor（multi）图；`mode="single"` 保留单 Agent 基线供评测。
编排与工具执行经依赖注入传入；本模块不含 FastAPI 类型。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field

from agent_workbench.domain.schemas import Task, TaskStatus
from agent_workbench.memory.context import estimate_tokens
from agent_workbench.memory.long_term import LongTermMemoryRepository, TaskReflection
from agent_workbench.observability.tracing import NoopSpanStore, SpanRecord, SpanStore
from agent_workbench.orchestrator.builder import build_supervisor_graph
from agent_workbench.orchestrator.single_agent import create_single_agent_graph, initial_state
from agent_workbench.orchestrator.state import (
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    default_max_steps,
    initial_supervisor_state,
)
from agent_workbench.tools.registry import ToolRegistry

__all__ = ["TaskMetricSample", "TaskNotFoundError", "TaskService"]

_TERMINAL = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED})

TaskMode = Literal["single", "multi"]


class TaskNotFoundError(KeyError):
    """任务不存在。"""


class TaskMetricSample(BaseModel):
    """单任务指标采样（终态时写入，供 GET /metrics）。"""

    task_id: str
    status: str
    duration_ms: float = Field(ge=0.0)
    token_total: int = Field(default=0, ge=0)


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
        mode: TaskMode = "multi",
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
        memory: LongTermMemoryRepository | None = None,
        span_store: SpanStore | None = None,
    ) -> None:
        """注入聊天模型与工具注册表。

        Args:
            llm: 聊天模型（真实或替身）。
            registry: 可用工具。
            max_steps: 单 Agent / SubAgent 工具轮次上限。
            mode: `multi`=Supervisor（默认）；`single`=单 Agent 基线（评测）。
            checkpointer: Supervisor 断点存储；single 模式忽略。
            max_repair_attempts: Supervisor 回修上限（默认 2）。
            memory: 长期记忆；非 None 时任务终态写入复盘。
            span_store: trace 存储；None 时用 Noop（不写库）。
        """
        if mode not in ("single", "multi"):
            msg = f"mode 必须为 single|multi，当前 {mode}"
            raise ValueError(msg)
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._mode: TaskMode = mode
        self._checkpointer = checkpointer if mode == "multi" else None
        self._max_repair_attempts = max_repair_attempts
        self._memory = memory
        self._span_store: SpanStore = span_store if span_store is not None else NoopSpanStore()
        self._runtime: dict[str, _Runtime] = {}
        self._metric_samples: list[TaskMetricSample] = []
        # 持有后台任务引用，避免被 GC 提前回收（RUF006）
        self._background: set[asyncio.Task[None]] = set()

    @property
    def mode(self) -> TaskMode:
        """当前编排模式。"""
        return self._mode

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

    def list_metric_samples(self) -> list[TaskMetricSample]:
        """返回全部终态任务指标采样副本。"""
        return list(self._metric_samples)

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
        """在后台执行编排图并写回终态、事件与指标采样。"""
        rt = self._require(task_id)
        description = rt.task.description
        started = time.perf_counter()
        await self._append_event(
            task_id,
            {
                "type": "task_started",
                "task_id": task_id,
                "description": description,
                "mode": self._mode,
            },
        )
        try:
            _final_state, extras = await asyncio.to_thread(self._run_graph, task_id, description)
            answer = str(extras.get("answer", ""))
            event: dict[str, Any] = {
                "type": "task_completed",
                "task_id": task_id,
                "answer": answer,
                "mode": self._mode,
            }
            event.update(extras)
            await self._append_event(task_id, event)
            await self._set_terminal(task_id, status=TaskStatus.COMPLETED, error=None)
            degraded = bool(extras.get("degraded", False))
            self._write_reflection(
                task_id,
                description,
                outcome="degraded" if degraded else "completed",
                summary=self._build_success_summary(extras, answer),
            )
            self._record_sample(
                task_id,
                status="completed",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                token_total=int(extras.get("token_total", 0)),
            )
        except Exception as exc:
            # 用户可见错误不携带堆栈/内部路径
            error_msg = f"任务执行失败: {type(exc).__name__}"
            await self._append_event(
                task_id,
                {"type": "task_failed", "task_id": task_id, "error": error_msg, "mode": self._mode},
            )
            await self._set_terminal(task_id, status=TaskStatus.FAILED, error=error_msg)
            self._write_reflection(
                task_id,
                description,
                outcome="failed",
                summary=f"执行失败（{type(exc).__name__}）；任务：{description[:120]}",
            )
            self._record_sample(
                task_id,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                token_total=0,
            )

    def _record_sample(
        self,
        task_id: str,
        *,
        status: str,
        duration_ms: float,
        token_total: int,
    ) -> None:
        """追加指标采样（旁路，失败不影响任务）。"""
        try:
            self._metric_samples.append(
                TaskMetricSample(
                    task_id=task_id,
                    status=status,
                    duration_ms=max(0.0, duration_ms),
                    token_total=max(0, token_total),
                )
            )
        except (ValueError, TypeError):
            return

    @staticmethod
    def _build_success_summary(extras: dict[str, Any], answer: str) -> str:
        """构造成功/降级复盘摘要（确定性，不额外调模型）。"""
        parts: list[str] = []
        if extras.get("degraded"):
            parts.append(f"降级交付（回修 {extras.get('repair_count', 0)} 次）")
        else:
            parts.append("正常完成")
        snippet = answer.strip().replace("\n", " ")[:160]
        if snippet:
            parts.append(f"结论：{snippet}")
        return "；".join(parts) if parts else "任务完成"

    def _write_reflection(
        self,
        task_id: str,
        description: str,
        *,
        outcome: str,
        summary: str,
    ) -> None:
        """写入长期复盘（旁路：失败不影响任务终态）。"""
        if self._memory is None:
            return
        try:
            self._memory.save(
                TaskReflection(
                    task_id=task_id,
                    task_description=description,
                    summary=summary,
                    outcome=outcome,
                )
            )
        except (ValueError, OSError):
            # 复盘写入失败不回滚任务状态
            return

    def _record_span(
        self,
        task_id: str,
        name: str,
        *,
        started_at: datetime,
        duration_ms: float,
        status: str = "ok",
        detail: str = "",
    ) -> None:
        """旁路写入 span（异常吞掉，不影响任务）。"""
        try:
            self._span_store.record(
                SpanRecord(
                    task_id=task_id,
                    name=name,
                    status=status,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    detail=detail,
                )
            )
        except (ValueError, OSError):
            return

    def _run_graph(self, task_id: str, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """同步执行图（在线程池中调用），返回终态与附加信息，并记录节点 span。"""
        if self._mode == "single":
            single_graph = create_single_agent_graph(
                self._llm,
                self._registry,
                max_steps=self._max_steps,
            )
            initial = initial_state(description, max_steps=self._max_steps)
            started = datetime.now(UTC)
            t0 = time.perf_counter()
            single_final: dict[str, Any] = single_graph.invoke(initial)
            duration = (time.perf_counter() - t0) * 1000.0
            self._record_span(
                task_id,
                "single_agent",
                started_at=started,
                duration_ms=duration,
                detail=f"mode=single steps={single_final.get('step_count', 0)}",
            )
            answer = self._extract_single_answer(single_final)
            extras: dict[str, Any] = {
                "answer": answer,
                "step_count": int(single_final.get("step_count", 0)),
                "token_total": estimate_tokens(answer) + estimate_tokens(description),
            }
            return single_final, extras

        supervisor_graph = build_supervisor_graph(
            self._llm,
            self._registry,
            checkpointer=self._checkpointer,
            max_steps=self._max_steps,
        )
        initial_sv = initial_supervisor_state(
            description,
            max_repair_attempts=self._max_repair_attempts,
        )
        run_config: RunnableConfig | None = None
        if self._checkpointer is not None:
            run_config = RunnableConfig(configurable={"thread_id": task_id})

        # stream 记录节点 span；values 终态作 final
        started_all = datetime.now(UTC)
        t0 = time.perf_counter()
        supervisor_final: dict[str, Any] = {}
        last_node_at = t0
        for stream_mode, chunk in supervisor_graph.stream(
            initial_sv,
            config=run_config,
            stream_mode=["updates", "values"],
        ):
            now = time.perf_counter()
            if stream_mode == "updates" and isinstance(chunk, dict):
                for node_name in chunk:
                    self._record_span(
                        task_id,
                        str(node_name),
                        started_at=datetime.fromtimestamp(
                            started_all.timestamp() + (last_node_at - t0),
                            tz=UTC,
                        ),
                        duration_ms=(now - last_node_at) * 1000.0,
                        detail="mode=multi",
                    )
                    last_node_at = now
            elif stream_mode == "values" and isinstance(chunk, dict):
                supervisor_final = chunk

        self._record_span(
            task_id,
            "supervisor_total",
            started_at=started_all,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            detail="mode=multi",
        )

        answer = str(supervisor_final.get("final_answer") or "")
        if not answer:
            answer = self._extract_single_answer(supervisor_final)
        usage = supervisor_final.get("token_usage") or {}
        token_total = int(usage.get("total", 0))
        if token_total <= 0:
            token_total = estimate_tokens(answer) + estimate_tokens(description)
        extras = {
            "answer": answer,
            "degraded": bool(supervisor_final.get("degraded", False)),
            "repair_count": int(supervisor_final.get("repair_count", 0)),
            "step_count": int(supervisor_final.get("repair_count", 0)),
            "token_total": token_total,
        }
        return supervisor_final, extras

    @staticmethod
    def _extract_single_answer(final_state: dict[str, Any]) -> str:
        """从消息中提取交付正文（single 模式 / supervisor 兜底）。

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
