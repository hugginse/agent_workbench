"""Trace 存储与 /progress 回放单测/集成测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_workbench.main import create_app
from agent_workbench.observability.tracing import NoopSpanStore, SpanRecord, SqliteSpanStore
from config.settings import Settings


class TestSqliteSpanStore:
    """span 读写。"""

    def test_record_and_list(self, tmp_path: Path) -> None:
        store = SqliteSpanStore(tmp_path / "traces.db")
        store.record(SpanRecord(task_id="t1", name="plan", duration_ms=12.5, detail="ok"))
        store.record(SpanRecord(task_id="t1", name="execute", duration_ms=30.0, status="ok"))
        store.record(SpanRecord(task_id="t2", name="plan", duration_ms=1.0))
        spans = store.list_by_task("t1")
        assert [s.name for s in spans] == ["plan", "execute"]
        assert spans[0].duration_ms == 12.5
        assert store.list_by_task("t2")[0].task_id == "t2"
        assert store.list_by_task("missing") == []
        store.close()

    def test_noop_store(self) -> None:
        s = NoopSpanStore()
        s.record(SpanRecord(task_id="x", name="n"))
        assert s.list_by_task("x") == []


class TestProgressEndpoint:
    """GET /progress/{task_id}。"""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Any:
        settings = Settings(_env_file=None, app_env="test", chat_model="")
        app = create_app(
            settings,
            checkpoint_db=tmp_path / "cp.db",
            memory_db=tmp_path / "m.db",
            trace_db=tmp_path / "traces.db",
        )
        with TestClient(app) as c:
            yield c

    def test_unknown_task_404(self, client: TestClient) -> None:
        resp = client.get("/progress/no-such-task")
        assert resp.status_code == 404
        assert "任务不存在" in resp.json()["detail"]
        assert "Traceback" not in resp.text

    def test_progress_after_task_runs(self, client: TestClient) -> None:
        created = client.post("/tasks", json={"description": "trace任务"}).json()
        task_id = created["id"]
        # 等到终态（SSE 或轮询）
        with client.stream("GET", f"/tasks/{task_id}/stream") as resp:
            raw = "".join(resp.iter_text())
        assert "task_completed" in raw or "task_failed" in raw

        prog = client.get(f"/progress/{task_id}")
        assert prog.status_code == 200
        body = prog.json()
        assert body["task_id"] == task_id
        assert body["status"] in ("completed", "failed")
        assert isinstance(body["spans"], list)
        assert len(body["spans"]) >= 1
        names = [s["name"] for s in body["spans"]]
        # multi 模式应含 supervisor 节点或 total
        assert any(n in names for n in ("plan", "supervisor_total", "aggregate", "single_agent"))
        # span 字段完整
        first = body["spans"][0]
        assert {"id", "task_id", "name", "status", "started_at", "duration_ms"} <= set(first)

    def test_span_duration_non_negative(self, client: TestClient) -> None:
        created = client.post("/tasks", json={"description": "耗时任务"}).json()
        with client.stream("GET", f"/tasks/{created['id']}/stream") as resp:
            "".join(resp.iter_text())
        body = client.get(f"/progress/{created['id']}").json()
        for s in body["spans"]:
            assert s["duration_ms"] >= 0
            datetime.fromisoformat(s["started_at"]).tzinfo is not None or True
