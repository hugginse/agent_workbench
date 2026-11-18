"""反馈端点与静态任务面板 API 测试。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_workbench.main import create_app
from config.settings import Settings


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, app_env="test", chat_model="")
    app = create_app(
        settings,
        checkpoint_db=tmp_path / "cp.db",
        memory_db=tmp_path / "m.db",
        trace_db=tmp_path / "traces.db",
    )
    with TestClient(app) as c:
        yield c


def _finish_task(client: TestClient) -> str:
    created = client.post("/tasks", json={"description": "反馈用任务"}).json()
    with client.stream("GET", f"/tasks/{created['id']}/stream") as resp:
        raw = "".join(resp.iter_text())
    assert "task_completed" in raw or "task_failed" in raw
    return str(created["id"])


class TestFeedback:
    """POST/GET /feedback。"""

    def test_submit_and_get_up(self, client: TestClient) -> None:
        task_id = _finish_task(client)
        resp = client.post(
            "/feedback",
            json={"task_id": task_id, "rating": "up"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["rating"] == "up"
        assert body["count_up"] == 1

        got = client.get(f"/feedback/{task_id}")
        assert got.status_code == 200
        assert got.json()["rating"] == "up"

    def test_submit_down(self, client: TestClient) -> None:
        task_id = _finish_task(client)
        resp = client.post(
            "/feedback",
            json={"task_id": task_id, "rating": "down"},
        )
        assert resp.status_code == 200
        assert resp.json()["rating"] == "down"
        assert resp.json()["count_down"] == 1

    def test_unknown_task_404(self, client: TestClient) -> None:
        resp = client.post(
            "/feedback",
            json={"task_id": "missing-task", "rating": "up"},
        )
        assert resp.status_code == 404

    def test_invalid_rating_422(self, client: TestClient) -> None:
        task_id = _finish_task(client)
        resp = client.post(
            "/feedback",
            json={"task_id": task_id, "rating": "meh"},
        )
        assert resp.status_code == 422

    def test_get_missing_feedback_404(self, client: TestClient) -> None:
        task_id = _finish_task(client)
        assert client.get(f"/feedback/{task_id}").status_code == 404


class TestStaticUi:
    """静态任务面板。"""

    def test_ui_index_served(self, client: TestClient) -> None:
        resp = client.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "任务面板" in resp.text

    def test_ui_consumes_api_paths(self, client: TestClient) -> None:
        """面板 HTML 引用现有 API 路径。"""
        text = client.get("/ui/").text
        for path in ("/tasks", "/metrics", "/feedback"):
            assert path in text
