# agent_workbench 运行镜像（uv + Python 3.11）
# 构建上下文为仓库根；.env 不进镜像，由 compose env_file 注入。

FROM ghcr.io/astral-sh/uv:0.13.5 AS uv-bin

FROM python:3.11-slim AS runtime

# 避免 uv/pyc 交互；时区可选
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY --from=uv-bin /uv /usr/local/bin/uv

WORKDIR /app

# 先拷依赖清单，利用层缓存
COPY pyproject.toml uv.lock .python-version README.md ./
COPY config ./config
COPY src ./src
COPY static ./static

# 同步运行依赖（不含 dev 组）
RUN uv sync --frozen --no-dev --no-install-project \
    && uv sync --frozen --no-dev

# 非 root 运行
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 容器内监听 0.0.0.0；数据写 /app/.data（由 compose 卷挂载）
CMD ["uv", "run", "uvicorn", "agent_workbench.main:app", "--host", "0.0.0.0", "--port", "8000"]
