# agent_workbench 运维手册（Runbook）

> 面向部署、日常运维与安全检查。架构见 [`architecture.md`](architecture.md)；开发契约见根目录 `CLAUDE.md`。

---

## 1. 部署方式

### 1.1 本地（开发）

```bash
uv sync
cp .env.example .env    # 按需填写；全空则用确定性替身
uv run uvicorn agent_workbench.main:app --reload
```

- 服务：`http://127.0.0.1:8000`
- API 文档：`/docs` · 任务面板：`/ui/`

### 1.2 Docker Compose（单机）

```bash
cp .env.example .env    # 填写 CHAT_MODEL / API_KEY / BASE_URL（可选）
docker compose up --build
docker compose down      # 停止（.data 保留在宿主机）
```

| 项 | 说明 |
|---|---|
| 端口 | `8000:8000` |
| 卷 | `./.data:/app/.data`（checkpoints / memory / traces） |
| 配置 | `env_file: .env` 运行时注入，**不打进镜像** |
| 健康检查 | `GET /docs`，30s 间隔 |

---

## 2. 数据目录（`.data/`）

| 文件 | 用途 | 备份建议 |
|---|---|---|
| `checkpoints.db` | LangGraph 断点续跑 | 重要任务需保留 |
| `memory.db` | 长期任务复盘 | 按需 |
| `traces.db` | node span（`GET /progress`） | 排障时保留 |
| `chroma/` | 向量索引（若启用 Chroma） | **换 embedding 必须清空重建** |

**清理向量数据：** 删除数据时必须同时清理向量索引与元数据（契约），勿只删一半。

---

## 3. 常用运维操作

### 3.1 质量门禁（发布前）

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

### 3.2 观测端点

| 端点 | 用途 |
|---|---|
| `GET /metrics` | 成功率 / 延迟 p50·p95 / token 成本（估算） |
| `GET /progress/{task_id}` | 节点 span 回放 |
| `GET /tasks/{id}` · SSE `/tasks/{id}/stream` | 任务状态与过程 |
| `POST/GET /feedback` | up/down 反馈 |

### 3.3 评测

```bash
uv run python evals/run_evaluation.py --mode compare --scripted   # 离线替身
uv run python evals/run_evaluation.py --mode compare              # 按 .env 真实模型（慢）
```

指标解读见 [`../evals/README.md`](../evals/README.md)。

### 3.4 日志

- 单行 JSON：`ts / level / logger / msg / request_id / extra`
- 请求关联：响应头 `X-Request-ID`；访问日志含 `http_request`
- 敏感键（`api_key`、`authorization` 等）自动脱敏为 `***`

---

## 4. 故障排查

| 现象 | 排查 |
|---|---|
| 起不来 / ModuleNotFoundError | `uv sync`；勿用系统 Python 直接跑 uvicorn |
| 任务一直 running | 看 SSE 与日志；查 `max_steps` / 模型超时 |
| 真实模型 401/网络错 | `.env` 的 `API_KEY`/`BASE_URL`；`APP_ENV` 勿为 `test` |
| 检索异常 | 是否切换过 embedding → **清空 `.data/` 重建索引** |
| Docker build 失败 | 确认 daemon 已启动；`.env` 未进上下文属预期 |
| compose 起来但无数据 | 宿主机 `./.data` 权限；容器内 uid 10001 |

---

## 5. 安全清单（Security Checklist）

上线/变更后逐项确认：

### 5.1 密钥与配置

- [ ] `.env` **不在 git**（`.gitignore` 已含 `.env`；勿提交真实密钥）
- [ ] `.env.example` 仅占位、无真实 `API_KEY`
- [ ] 镜像构建上下文不含 `.env`（见 `.dockerignore`）
- [ ] 生产 `LOG_LEVEL` 合理；日志无 API Key / Authorization / 完整敏感体
- [ ] 用户可见错误不含堆栈、内部路径、密钥

### 5.2 数据

- [ ] `.data/`、数据库文件不提交 git
- [ ] 测试/评测仅用**合成数据**，无真实用户文档进 fixture
- [ ] 删除业务数据时**同时**清理向量索引与元数据
- [ ] 备份策略覆盖 `checkpoints.db`（按任务重要性）

### 5.3 运行时与 Agent 行为

- [ ] `python_sandbox`：超时生效、无网络（socket 补丁）
- [ ] 循环均有计数上限（`max_steps`、`max_repair_attempts` ≤ 5）
- [ ] 检索/复盘内容按**数据**处理，不当作系统指令
- [ ] 引用 id 仅来自检索命中；无证据拒答
- [ ] LangSmith 开关默认关；开启前确认出网策略

### 5.4 部署

- [ ] 容器非 root（Dockerfile `appuser`）
- [ ] 仅暴露必要端口（默认 8000）
- [ ] 反向代理/防火墙按需限制 `/docs`、`/ui` 访问面
- [ ] `docker compose down` 后宿主机 `.data` 权限正确

### 5.5 发布前

- [ ] 四件套全绿（ruff / format / mypy / pytest）
- [ ] 无密钥、无 `.data`、无 `.env` 进入 commit
- [ ] 变更涉及评测数字时更新 README 并标注环境

---

## 6. 相关文档

| 文档 | 内容 |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | 开发契约（红线与门禁） |
| [`plan.md`](plan.md) | 阶段编排与验收 |
| [`architecture.md`](architecture.md) | 分层与机制 |
| [`../README.md`](../README.md) | 快速开始与评测结果 |
| [`../evals/README.md`](../evals/README.md) | 指标定义与限制 |
