"""结构化 JSON 日志与请求关联（X-Request-ID / contextvar）。

- 输出单行 JSON：ts / level / logger / msg / request_id / extra
- 敏感键（authorization、api_key 等）在 extra 中自动脱敏
- 不用 print 做应用日志（契约）
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "REQUEST_ID_HEADER",
    "JsonFormatter",
    "clear_request_id",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "set_request_id",
]

REQUEST_ID_HEADER = "X-Request-ID"

# 进程内请求关联（async 安全的 contextvar）
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# 日志 extra 中需脱敏的键（大小写不敏感子串）
_SENSITIVE_KEYS = (
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "cookie",
)


def get_request_id() -> str | None:
    """读取当前上下文的 request id。"""
    return _request_id.get()


def set_request_id(value: str) -> Token[str | None]:
    """设置当前上下文 request id，返回 token 以便还原。"""
    return _request_id.set(value)


def clear_request_id(token: Token[str | None] | None = None) -> None:
    """清除 request id（或按 token 回滚）。"""
    if token is not None:
        _request_id.reset(token)
    else:
        _request_id.set(None)


def _mask_sensitive(key: str, value: Any) -> Any:
    """敏感键返回掩码，其它值原样返回。"""
    lowered = key.lower()
    if any(s in lowered for s in _SENSITIVE_KEYS):
        if value is None or value == "":
            return value
        return "***"
    return value


class JsonFormatter(logging.Formatter):
    """将 LogRecord 格式化为单行 JSON。

    字段：ts / level / logger / msg / name（额外）/ request_id / exc_info
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化为 JSON 字符串。

        Args:
            record: 标准 LogRecord。

        Returns:
            单行 JSON 文本。
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid

        # 额外字段（skip 内部键）
        skip = set(logging.makeLogRecord({}).__dict__) | {
            "message",
            "asctime",
            "request_id",
        }
        for key, value in record.__dict__.items():
            if key in skip or key.startswith("_"):
                continue
            payload[key] = _mask_sensitive(key, value)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    stream: Any = None,
    force: bool = True,
) -> None:
    """配置根/应用日志为 JSON 输出。

    Args:
        level: 日志级别名（INFO/DEBUG/…）。
        stream: 输出流；默认 sys.stdout。
        force: 是否清理已有 handlers（测试中常用 True）。
    """
    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """按名称获取 logger（调用方负责是否 configure）。"""
    return logging.getLogger(name)
