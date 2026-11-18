"""GET /metrics 与指标聚合单测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_workbench.application.services.task import TaskMetricSample
from agent_workbench.main import create_app
from agent_workbench.observability.metrics import compute_service_metrics
from config.settings import Settings


class TestComputeServiceMetrics:
    """聚合纯函数。"""

    def test_empty(self) -> None:
        m = compute_service_metrics([])
        assert m.task_count == 0
        assert m.success_rate == 0.0

    def test_rates_and_latency(self) -> None:
        samples = [
            TaskMetricSample(task_id="a", status="completed", duration_ms=10.0, token_total=100),
            TaskMetricSample(task_id="b", status="completed", duration_ms=30.0, token_total=200),
            TaskMetricSample(task_id="c", status="failed", duration_ms=50.0, token_total=0),
        ]
        m = compute_service_metrics(samples)
        assert m.task_count == 3
        assert m.completed_count == 2
        assert m.failed_count == 1
        assert m.success_rate == pytest.approx(2 / 3, abs=1e-4)
        assert m.token_total_sum == 300
        assert m.token_mean == 100.0
        assert m.latency_p50_ms > 0
        assert m.latency_p95_ms >= m.latency_p50_ms


class TestMetricsEndpoint:
    """HTTP 端点。"""

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

    def test_empty_metrics(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_count"] == 0
        assert body["success_rate"] == 0.0

    def test_metrics_after_task(self, client: TestClient) -> None:
        created = client.post("/tasks", json={"description": "指标任务"}).json()
        with client.stream("GET", f"/tasks/{created['id']}/stream") as resp:
            raw = "".join(resp.iter_text())
        assert "task_completed" in raw or "task_failed" in raw

        body = client.get("/metrics").json()
        assert body["task_count"] >= 1
        assert 0.0 <= body["success_rate"] <= 1.0
        assert body["latency_p50_ms"] >= 0
        assert body["latency_p95_ms"] >= body["latency_p50_ms"]
        assert body["token_total_sum"] >= 0
        # 成功后 completed+failed == task_count
        assert body["completed_count"] + body["failed_count"] == body["task_count"]
