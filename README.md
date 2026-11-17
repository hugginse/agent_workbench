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

## 常用命令

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Embedding 切换约束

默认使用确定性替身 `FakeEmbeddings`（离线可复现）。**切换真实 embedding 后必须清空 `.data/` 并重建向量索引**（否则伪向量与真实向量空间不一致，检索结果无效）。

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
- [ ] Phase 5 #30 `feat: add hallucination and token cost metrics`
- [ ] Phase 5 #31 `feat: add single agent vs multi agent comparison mode`
- [ ] Phase 5 #32 `test: cover evaluation harness with scripted model`
- [ ] Phase 5 #33 `docs: embed evaluation results and ablation table in README`

**Tag:** `v0.6.0-phase5`（Phase 5 完成后打）
