# agent_workbench

可独立运行、可测试、可观测、可部署的**多智能体任务执行与评测平台**：
Supervisor 规划拆解 → SubAgent 分工执行 → 评审回修闭环 → 聚合交付；
双层记忆、工具体系、Agent Harness 量化评测、全链路可观测。

> 开发契约见根目录 [`CLAUDE.md`](CLAUDE.md)；阶段编排见 [`docs/plan.md`](docs/plan.md)；架构说明见 [`docs/architecture.md`](docs/architecture.md)。

## 快速开始

```bash
uv sync
uv run pytest
uv run uvicorn agent_workbench.main:app --reload
```

### Docker Compose

```bash
cp .env.example .env   # 填写 CHAT_MODEL / API_KEY 等（可选；不填则用替身）
docker compose up --build
# 面板: http://127.0.0.1:8000/ui/   API 文档: /docs
```

`.data/`（checkpoints / memory / traces）经卷持久化到宿主机；`.env` 只在运行时注入，不打进镜像。

## 常用命令

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Embedding 切换约束

默认使用确定性替身 `FakeEmbeddings`（离线可复现）。**切换真实 embedding 后必须清空 `.data/` 并重建向量索引**（否则伪向量与真实向量空间不一致，检索结果无效）。

## 评测结果（Agent Harness A/B）

数据集：`evals/datasets/tasks_v1.jsonl`（36 条合成任务，含 6 条 `should_fail`）。  
完整指标定义与限制见 [`evals/README.md`](evals/README.md)。

### 1) 替身模式（离线可复现）

- **环境：** `ScriptedChatModel` + 默认 registry；`--scripted`；`max_steps=8`，`max_repair_attempts=2`
- **运行日期：** 2026-11-17
- **限制：** 仅验证管线与相对结构，**不能**代表真实模型效果

| 指标 | single | multi | Δ (multi−single) |
|---|---:|---:|---:|
| 任务通过率 | 0.8333 | 1.0000 | +0.1667 |
| 逐步成功率 | 1.0000 | 1.0000 | +0.0000 |
| 回修率 | 0.0000 | 1.0000 | +1.0000 |
| 平均回修 | 0.0 | 2.0 | +2.0 |
| 平均 token（估算） | 71.94 | 1226.42 | +1154.47 |
| token 合计（估算） | 2590 | 44151 | +41561 |
| 延迟 p50 (ms) | 1.29 | 4.49 | +3.19 |
| 延迟 p95 (ms) | 1.71 | 4.67 | +2.96 |
| 幻觉率 | 0.0000 | 0.0000 | +0.0000 |

### 2) 真实模型（全量 36×2）

- **环境：** `APP_ENV=dev`；`CHAT_MODEL=mimo-v2.6-flash`（`BASE_URL` 见 `.env`，已连通验证）；Embedding 配置为 `BAAI/bge-m3`（本评测未走 RAG 索引）
- **参数：** `--mode compare`，`max_steps=8`，`max_repair_attempts=2`，**未截断数据集**
- **运行日期：** 2026-11-18；串行执行，全量约数小时
- **token 为启发式估算**，非供应商账单；延迟为端到端 wall-clock

| 指标 | single | multi | Δ (multi−single) |
|---|---:|---:|---:|
| 任务通过率 | 0.8056 | 0.9722 | +0.1666 |
| 逐步成功率 | 1.0000 | 0.9722 | −0.0278 |
| 回修率 | 0.0000 | 0.5000 | +0.5000 |
| 平均回修 | 0.0 | 0.9444 | +0.9444 |
| 平均 token（估算） | 1618.61 | 28557.19 | +26938.58 |
| token 合计（估算） | 58270 | 1028059 | +969789 |
| 延迟 p50 (ms) | 18619 | 300155 | +281536 |
| 延迟 p95 (ms) | 104630 | 1228399 | +1123769 |
| 引用支撑率 | 0.3333 (n=3) | 0.3000 (n=10) | −0.0333 |
| 幻觉率 | 0.6667 | 0.7000 | +0.0333 |

**解读（摘要）：**

- 两种模式下 **multi 通过率均高于 single**（替身 +0.167，真实 +0.167），Supervisor 回修对 `should_fail`/质量更有利。
- 真实模型下 multi **token 与延迟显著更高**（多 Agent 多次调用 + 回修），符合预期权衡。
- 真实幻觉率偏高、支撑率样本少：默认 judge 未强制 RAG 引用；无检索闭环时 `[n]` 无法核验。属指标口径限制，见 `evals/README.md`。
- **替身数字仅作管线回归**；面试/对外应引用上表真实数字并注明环境与日期。

## 阶段进度

- [x] Phase 0 #1 `docs: add project plan and development contract`
- [x] Phase 0 #2 `chore: bootstrap uv project with ruff, mypy, pytest config`
- [x] Phase 0 #3 `feat: add settings module and chat model factory with test doubles`
- [x] Phase 0 #4 `ci: add GitHub Actions quality gate workflow`

**Tag:** `v0.1.0-phase0` — Phase 0 已完成

- [x] Phase 1 #5 `feat: add domain schemas for task and agent messages`
- [x] Phase 1 #6 `feat: add tool registry with schema validation`
- [x] Phase 1 #7 `feat: add built-in tools finish and python sandbox`
- [x] Phase 1 #8 `feat: add single agent react loop with bounded steps`
- [x] Phase 1 #9 `feat: add task API endpoints with progress streaming`
- [x] Phase 1 #10 `test: cover agent loop and tool calling integration`

**Tag:** `v0.2.0-phase1` — Phase 1 已完成

- [x] Phase 2 #11 `feat: add supervisor state schema with repair counter`
- [x] Phase 2 #12 `feat: add plan node for task decomposition`
- [x] Phase 2 #13 `feat: add assign and execute nodes with subagent dispatch`
- [x] Phase 2 #14 `feat: add review node with bounded repair loop`
- [x] Phase 2 #15 `feat: add aggregate node and completion routing`
- [x] Phase 2 #16 `feat: wire supervisor graph with sqlite checkpointer for resume`
- [x] Phase 2 #17 `refactor: switch task API from single agent to supervisor graph`

**Tag:** `v0.3.0-phase2` — Phase 2 已完成

- [x] Phase 3 #18 `feat: add long term memory repository for task reflections`
- [x] Phase 3 #19 `feat: add memory_recall tool for supervisor`
- [x] Phase 3 #20 `feat: write task reflection to long term memory on completion`
- [x] Phase 3 #21 `feat: add context pruning and per node token accounting`
- [x] Phase 3 #22 `test: cover cross session memory and token metrics`

**Tag:** `v0.4.0-phase3` — Phase 3 已完成

- [x] Phase 4 #23 `feat: add document loaders and text splitter`
- [x] Phase 4 #24 `feat: add chroma vector store adapter and indexing pipeline`
- [x] Phase 4 #25 `feat: add retrieve tool with citation builder`
- [x] Phase 4 #26 `feat: enforce citation iron rules and abstention`
- [x] Phase 4 #27 `test: cover retrieval chat integration with synthetic documents`

**Tag:** `v0.5.0-phase4` — Phase 4 已完成

- [x] Phase 5 #28 `feat: add eval dataset schema and synthetic task fixtures`
- [x] Phase 5 #29 `feat: add eval runner with task success and stability metrics`
- [x] Phase 5 #30 `feat: add hallucination and token cost metrics`
- [x] Phase 5 #31 `feat: add single agent vs multi agent comparison mode`
- [x] Phase 5 #32 `test: cover evaluation harness with scripted model`
- [x] Phase 5 #33 `docs: embed evaluation results and ablation table in README`

**Tag:** `v0.6.0-phase5` — Phase 5 已完成

- [x] Phase 6 #34 `feat: add structured json logging with request correlation`
- [x] Phase 6 #35 `feat: add node level trace recording and replay endpoint`
- [x] Phase 6 #36 `feat: add metrics endpoint for success latency and cost`
- [x] Phase 6 #37 `feat: add feedback endpoint and static web ui`
- [x] Phase 6 #38 `feat: add dockerfile and compose deployment`
- [ ] Phase 6 #39 `docs: add runbook with security checklist`

**Tag:** `v1.0.0-phase6`（Phase 6 完成后打）
