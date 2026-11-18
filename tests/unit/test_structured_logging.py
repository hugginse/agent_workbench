"""结构化 JSON 日志与 X-Request-ID 关联单测。"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agent_workbench.main import create_app
from agent_workbench.observability.logging import (
    REQUEST_ID_HEADER,
    JsonFormatter,
    clear_request_id,
    configure_logging,
    get_logger,
    set_request_id,
)
from config.settings import Settings


@pytest.fixture(autouse=True)
def _clean_request_id() -> Iterator[None]:
    """每个测试后清空 contextvar。"""
    yield
    clear_request_id()


class TestJsonFormatter:
    """JsonFormatter 字段与脱敏。"""

    def _record(self, **extra: object) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_basic_fields(self) -> None:
        out = JsonFormatter().format(self._record())
        data = json.loads(out)
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["msg"] == "hello world"
        assert "ts" in data

    def test_request_id_attached(self) -> None:
        set_request_id("req-abc")
        data = json.loads(JsonFormatter().format(self._record()))
        assert data["request_id"] == "req-abc"

    def test_masks_api_key(self) -> None:
        data = json.loads(JsonFormatter().format(self._record(api_key="sk-secret-value")))
        assert data["api_key"] == "***"
        assert "sk-secret" not in json.dumps(data)

    def test_masks_authorization(self) -> None:
        data = json.loads(JsonFormatter().format(self._record(authorization="Bearer xyz")))
        assert data["authorization"] == "***"

    def test_extra_passthrough(self) -> None:
        data = json.loads(JsonFormatter().format(self._record(path="/tasks")))
        assert data["path"] == "/tasks"


class TestConfigureLogging:
    """configure_logging 输出 JSON 行。"""

    def test_writes_json_to_stream(self) -> None:
        stream = io.StringIO()
        configure_logging("INFO", stream=stream, force=True)
        log = get_logger("agent_workbench.test")
        log.info("事件发生", extra={"task_id": "t1"})
        configure_logging("INFO", force=True)  # 恢复默认

        line = stream.getvalue().strip().splitlines()[-1]
        data = json.loads(line)
        assert data["msg"] == "事件发生"
        assert data["task_id"] == "t1"


class TestRequestIdMiddleware:
    """API 层 X-Request-ID。"""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Iterator[TestClient]:

        settings = Settings(_env_file=None, app_env="test", chat_model="")
        app = create_app(
            settings,
            checkpoint_db=tmp_path / "cp.db",
            memory_db=tmp_path / "m.db",
        )
        with TestClient(app) as c:
            yield c

    def test_generates_request_id(self, client: TestClient) -> None:
        resp = client.get("/docs")
        rid = resp.headers.get(REQUEST_ID_HEADER)
        assert rid
        assert len(rid) >= 8

    def test_echoes_incoming_request_id(self, client: TestClient) -> None:
        resp = client.get("/docs", headers={REQUEST_ID_HEADER: "my-fixed-id-1"})
        assert resp.headers.get(REQUEST_ID_HEADER) == "my-fixed-id-1"

    def test_request_id_in_access_log_json(self, client: TestClient) -> None:
        stream = io.StringIO()
        configure_logging("INFO", stream=stream, force=True)
        try:
            client.get("/docs", headers={REQUEST_ID_HEADER: "log-rid-42"})
        finally:
            configure_logging("INFO", force=True)

        texts = stream.getvalue()
        assert "http_request" in texts
        found = False
        for line in texts.splitlines():
            if "http_request" not in line:
                continue
            data = json.loads(line)
            if data.get("request_id") == "log-rid-42":
                found = True
                assert data.get("path") == "/docs"
        assert found, "访问日志应带 request_id=log-rid-42"


# Path 仅用于类型注解时提前导入
from pathlib import Path  # noqa: E402
