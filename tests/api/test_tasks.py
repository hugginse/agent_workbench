"""任务 API 端点测试：创建、查询、SSE 进度流。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_workbench.main import create_app
from config.settings import Settings


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """使用隔离 Settings（替身模型）与临时 checkpointer 的 TestClient。"""
    settings = Settings(_env_file=None, app_env="test", chat_model="")
    app = create_app(settings, checkpoint_db=tmp_path / "cp.db")
    with TestClient(app) as c:
        yield c


def _parse_sse_data(text: str) -> list[dict[str, Any]]:
    """解析 SSE 文本中的 data JSON 行。"""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


class TestCreateAndGet:
    """POST /tasks 与 GET /tasks/{id}。"""

    def test_create_returns_201_and_running(self, client: TestClient) -> None:
        """创建成功返回 201、任务 id 与 running 状态。"""
        resp = client.post("/tasks", json={"description": "写一份摘要"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"]
        assert body["description"] == "写一份摘要"
        assert body["status"] == "running"
        assert body["error"] is None

    def test_create_empty_description_422(self, client: TestClient) -> None:
        """空描述被 schema 拒绝。"""
        resp = client.post("/tasks", json={"description": ""})
        assert resp.status_code == 422

    def test_create_missing_field_422(self, client: TestClient) -> None:
        """缺少 description 返回 422。"""
        resp = client.post("/tasks", json={})
        assert resp.status_code == 422

    def test_get_task_by_id(self, client: TestClient) -> None:
        """创建后可按 id 查询。"""
        created = client.post("/tasks", json={"description": "查询测试"}).json()
        resp = client.get(f"/tasks/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == created["id"]

    def test_get_missing_task_404(self, client: TestClient) -> None:
        """不存在的任务返回 404，不泄露内部细节。"""
        resp = client.get("/tasks/no-such-id")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert "任务不存在" in detail
        assert "Traceback" not in detail


class TestSseStream:
    """GET /tasks/{id}/stream SSE。"""

    def test_stream_reaches_terminal_event(self, client: TestClient) -> None:
        """流式输出以 task_completed 结束（替身直接答复）。"""
        created = client.post("/tasks", json={"description": "流式测试"}).json()
        task_id = created["id"]

        with client.stream("GET", f"/tasks/{task_id}/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            raw = "".join(resp.iter_text())

        events = _parse_sse_data(raw)
        assert events, "SSE 应至少产生 task_started 事件"
        assert events[0]["type"] == "task_started"
        types = [e["type"] for e in events]
        assert "task_completed" in types or "task_failed" in types
        # 终态必须是最后一条业务事件
        assert types[-1] in ("task_completed", "task_failed")

    def test_stream_missing_task_404(self, client: TestClient) -> None:
        """不存在任务的流接口返回 404。"""
        resp = client.get("/tasks/missing/stream")
        assert resp.status_code == 404

    def test_stream_includes_completed_answer(self, client: TestClient) -> None:
        """task_completed 事件携带 answer 字段。"""
        created = client.post("/tasks", json={"description": "带答案"}).json()
        with client.stream("GET", f"/tasks/{created['id']}/stream") as resp:
            raw = "".join(resp.iter_text())
        events = _parse_sse_data(raw)
        completed = [e for e in events if e["type"] == "task_completed"]
        assert completed, f"未见 task_completed，events={events}"
        assert "answer" in completed[0]
        assert completed[0]["task_id"] == created["id"]
