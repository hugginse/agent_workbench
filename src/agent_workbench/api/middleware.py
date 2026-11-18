"""HTTP 中间件：X-Request-ID 读取/生成、回传头与访问日志。"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from agent_workbench.observability.logging import (
    REQUEST_ID_HEADER,
    clear_request_id,
    get_logger,
    set_request_id,
)

__all__ = ["RequestIdMiddleware"]

logger = get_logger("agent_workbench.http")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """为每个请求绑定 X-Request-ID（入站可复用，否则生成 UUID4）。

    - 响应头回写同一 id；
    - 在 contextvar 中设置，供 JsonFormatter 附带 request_id；
    - 结束后清理 contextvar。
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """处理单个请求的关联 id。

        Args:
            request: 入站请求。
            call_next: 下游处理。

        Returns:
            响应（含 X-Request-ID 头）。
        """
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            # 在清理 contextvar 之前写访问日志，保证带 request_id
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                },
            )
            return response
        finally:
            clear_request_id(token)
