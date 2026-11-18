# agent_workbench 开发计划（docs/plan.md）

> 本文件与仓库根 `CLAUDE.md` 构成本项目唯一权威开发文档，互相补充：
> **阶段流程、提交编排、验收标准看本文件；架构红线、代码规范、门禁细则看 `CLAUDE.md`。**
> 冲突时以 `CLAUDE.md` 为准。

---

## 0. 已确认决策

| 决策项 | 结论 |
|---|---|
| 分支 | 仓库初始化即在 `main` |
| 推送节奏 | 每个 Phase 四件套全绿 → 汇报 → 打 tag → `git push origin main --tags`；禁止 force push |
| Commit 规范 | Conventional Commits，**英文 subject**，逐字使用第 5 节编排表；测试与实现同一 commit |
| Phase 7（定时任务） | **保留为可选**，不排入主线，仅在用户明确要求时实施 |
| 项目形态 | 独立 git 仓库、独立 uv 项目、完全自包含 |

---

## 1. 项目定义

**名称：** `agent_workbench`（多智能体任务执行与评测平台）

**定位：** 用户提交复杂任务 → Supervisor 规划拆解 → SubAgent 分工执行（检索 / 执行 / 审查）→ 评审回修闭环 → 聚合交付。全链路可记忆、可观测、可量化评测。

**载体场景：** 通用「复杂研究与文档任务」（调研汇总、方案对比、资料整理）。不绑定任何垂直行业，换任何业务背景都能复用讲述。

**自包含约束：**

- 本项目是独立 git 仓库、独立 uv 项目，**自包含**。
- **禁止依赖任何本地其他项目 / 仓库**（不 import、不相对路径引用、不复制脚本）。
- 实现所需的全部设计、规则、约定以本文件与根 `CLAUDE.md` 为唯一权威来源；实现者无需查阅任何外部本地资料。

**能力覆盖：**

| 能力域 | 对应模块 |
|---|---|
| 多智能体编排（规划/拆解/分发/聚合/回溯/断点续跑） | `orchestrator/` Supervisor 图 |
| Tool Use / Function Calling 体系（动态注册、按需装配） | `tools/` ToolRegistry |
| Agent Harness 评测（通过率/链路稳定性/幻觉率/成本） | `harness/` + `evals/` |
| Memory（短期状态 + 长期复盘）与上下文管理 | `memory/` + token 统计 |
| 可观测性（结构化日志 / trace 回放 / 指标端点） | `observability/` |
| RAG 全链路（检索、引用溯源、拒答） | `rag/` |
| 工程基线（FastAPI / 分层 / 测试 / Docker / CI） | 全仓库 |

---

## 2. 技术栈与范围裁剪

**采用：** Python 3.11 + uv · FastAPI · **LangGraph（核心编排）** + langchain-core · Pydantic · Chroma · SQLite · pytest / ruff / mypy · Docker Compose · SSE · 自研轻量 tracing + LangSmith 开关 · GitHub Actions CI

**明确不做**（面试用「设计预留 / 了解原理」话术，不写代码）：

| 不做 | 预留方式 |
|---|---|
| Milvus / Pinecone / PostgreSQL | `VectorStore` Protocol 抽象，README 写迁移路径 |
| K8s / MQ / Redis 集群 | Compose 单机；README「扩展方向」一节 |
| MCP / A2A 协议实现 | ToolRegistry schema 即协议化设计，可讲「如何接 MCP」 |
| RL、多机分布式调度 | 超出目标岗位范围，不实现 |
| 定时任务 / 心跳主动执行 | **Phase 7（可选）**，约半天，默认不排入主线 |

---

## 3. 目录结构（目标形态）

```text
agent_workbench/
├── CLAUDE.md                  # 开发契约（实现者每次必读）
├── README.md
├── pyproject.toml             # hatchling; packages = src + config
├── .python-version / .gitignore / .env.example
├── Dockerfile / docker-compose.yml
├── config/settings.py         # 唯一读 .env 的地方
├── docs/
│   ├── plan.md                # 本文件
│   ├── architecture.md
│   └── runbook.md
├── src/agent_workbench/
│   ├── main.py
│   ├── api/                   # container, dependencies, endpoints/{health,tasks,progress,memory,metrics}
│   ├── application/           # services: task / session / memory
│   ├── domain/                # models, schemas, protocols, exceptions（零框架依赖）
│   ├── orchestrator/          # supervisor 图：state.py, builder.py, nodes/{plan,assign,execute,review,aggregate}
│   ├── agents/                # SubAgent 声明（researcher / executor / reviewer = prompt+tools 配置）
│   ├── tools/                 # registry.py, base.py, builtins/{python_sandbox, finish, retrieve, memory_recall}
│   ├── memory/                # short_term(checkpointer 封装), long_term(SQLite 复盘仓库)
│   ├── rag/                   # loaders, splitters, indexing, retrieval, citation
│   ├── harness/               # 评测：runner, metrics, report
│   ├── observability/         # logging(结构化), tracing(node span), metrics
│   └── infrastructure/        # llm/(工厂+替身), vectorstores/, persistence/, checkpoint/
├── evals/
│   ├── datasets/tasks_v1.jsonl
│   ├── README.md
│   └── run_evaluation.py      # CLI：--mode single|multi --json-out ...
└── tests/{unit,integration,api}
```

---

## 4. 核心机制（实现约束）

### 4.1 Supervisor 闭环（LangGraph StateGraph）

```text
START → plan → assign → execute → review →
review.pass                → aggregate → END
review.fail & repair_count < max  → assign（定向回修）
review.fail & repair_count >= max → aggregate（降级交付 + 说明）
```

- 每个循环由计数器封顶：`max_repair_attempts`（默认 2，上限 5），**必须有终止测试**。
- `thread_id` → SQLite checkpointer → **断点续跑**（任务中途重启可恢复）。
- SubAgent 是「prompt + 工具子集 + 输出 schema」的**声明式配置**，用组合不用子图继承。

### 4.2 ToolRegistry 工具体系

- Pydantic schema 定义 → 校验 → 注册 → 按 SubAgent 声明动态装配。
- 首批工具：`python_sandbox`（subprocess + timeout + 无网络）、`retrieve`（Phase 4 接 RAG）、`memory_recall`、`finish`（结构化交付）。
- 工具描述 + 前置条件 = **Skill 技能库**轻量实现，不单独造模块。

### 4.3 双层 Memory 与上下文管理

- 短期：checkpointer（会话状态、断点续跑）。
- 长期：SQLite 存任务复盘摘要，Supervisor 经 `memory_recall` 工具检索注入。
- 上下文管理：按节点裁剪历史消息 + 每节点 token 用量记入 state → 汇入成本指标。

### 4.4 RAG 子能力（Phase 4）铁律

- 引用 ID 只能来自检索命中，模型不得生成。
- 检索材料在 prompt 中视为**不可信数据**（防注入）。
- 无证据拒答；`confidence` 不直接复制相似度。

### 4.5 Agent Harness 评测（最大差异化）

- 数据集 30~50 条合成任务（含应失败样本），JSONL schema：`id / task / expected_artifacts / should_fail / rubric`。
- 指标：**任务通过率、链路稳定性（逐步成功率 / 重试率）、幻觉率（论断-引用支撑率）、延迟 p50/p95、token 成本**。
- `--mode single`（单 Agent 基线）vs `--mode multi`（Supervisor）→ **A/B 对比表写进 README**。
- 默认用确定性替身模型保证离线可复现；真实模型数字单独跑一轮并在 README 标注环境（定义见 `CLAUDE.md`「测试替身与离线可复现」）。

### 4.6 可观测性

- JSON 结构化日志 + `X-Request-ID`；node 级 span 落 SQLite 可回放（`GET /progress/{task_id}`）。
- `GET /metrics`：成功率 / 延迟 / token 成本。
- LangSmith 开关默认关，关闭时应用完全本地可运行。

---

## 5. 分阶段开发流程 + Git 提交历史编排

**分支策略：** 初始化即在 `main`；每个 Phase 结束：四件套全绿 → 汇报 → 提交收尾 → 打 tag → `git push origin main --tags`。
**提交规范：** Conventional Commits，英文 subject；每个 commit = 一个逻辑变更，**测试与实现同 commit**；涉及代码的 commit 提交前必须四件套全绿（纯文档 commit 除外，但不得夹带代码变更）。
**归属行：** 每条 commit message 末尾加：

```text
Co-Authored-By: Claude Code <noreply@anthropic.com>
```

### Phase 0 — 工程基线（4 commits）

| # | Commit message | 交付物 | 验收 |
|---|---|---|---|
| 1 | `docs: add project plan and development contract` | `docs/plan.md`、`CLAUDE.md`、`docs/architecture.md` 初稿 | 文档就位 |
| 2 | `chore: bootstrap uv project with ruff, mypy, pytest config` | `pyproject.toml`、`.python-version`、`.gitignore`、`.env.example`、`config/settings.py` | `uv sync` 成功；空测试可跑 |
| 3 | `feat: add settings module and chat model factory with test doubles` | Settings 读取 + `get_chat_model()`（真实 / Scripted 替身双路径）+ 单测 | `pytest tests/unit/test_settings.py` 绿 |
| 4 | `ci: add GitHub Actions quality gate workflow` | `.github/workflows/ci.yml`（四件套） | 本地四件套全绿，CI 配置语法有效 |

**Tag:** `v0.1.0-phase0`

### Phase 1 — 单 Agent + Function Calling 最小闭环（6 commits）

| # | Commit message | 交付物 |
|---|---|---|
| 5 | `feat: add domain schemas for task and agent messages` | `domain/schemas/`：Task、AgentMessage、ToolCall、TaskResult |
| 6 | `feat: add tool registry with schema validation` | `tools/registry.py` + `base.py` + 单测（注册 / 校验 / 拒绝非法工具） |
| 7 | `feat: add built-in tools finish and python sandbox` | `tools/builtins/`：finish、python_sandbox（timeout + 无网络）+ 单测 |
| 8 | `feat: add single agent react loop with bounded steps` | `orchestrator/` 单 Agent 图（工具循环计数封顶）+ 终止测试 |
| 9 | `feat: add task API endpoints with progress streaming` | `POST /tasks`、`GET /tasks/{id}`、SSE `/tasks/{id}/stream` + API 测试 |
| 10 | `test: cover agent loop and tool calling integration` | 集成测试：任务端到端、工具调用断言、超限终止 |

**Tag:** `v0.2.0-phase1` · **阶段汇报：** 修改文件 / 设计决策 / 执行命令 / 测试结果 / 未解决问题

### Phase 2 — Supervisor 多智能体编排（7 commits）

| # | Commit message | 交付物 |
|---|---|---|
| 11 | `feat: add supervisor state schema with repair counter` | `orchestrator/state.py`：plan / results / verdict / repair_count / token_usage |
| 12 | `feat: add plan node for task decomposition` | plan 节点（结构化输出拆解子任务）+ 单测 |
| 13 | `feat: add assign and execute nodes with subagent dispatch` | assign / execute + `agents/` 三个 SubAgent 声明 + 单测 |
| 14 | `feat: add review node with bounded repair loop` | review 节点 + 条件边 + **三条路径测试（通过 / 回修 / 超限降级）** |
| 15 | `feat: add aggregate node and completion routing` | 聚合交付 + END 路由 |
| 16 | `feat: wire supervisor graph with sqlite checkpointer for resume` | `compile(checkpointer)` + **断点续跑集成测试** |
| 17 | `refactor: switch task API from single agent to supervisor graph` | API 切换；单 Agent 模式保留为 `--mode single` 供评测用 |

**Tag:** `v0.3.0-phase2` — 本 Phase 是核心卖点，完成后即可投递保底版

### Phase 3 — Memory + 上下文管理（5 commits）

| # | Commit message | 交付物 |
|---|---|---|
| 18 | `feat: add long term memory repository for task reflections` | SQLite 复盘表 + Protocol + 单测 |
| 19 | `feat: add memory_recall tool for supervisor` | 工具接入 + 跨会话召回测试 |
| 20 | `feat: write task reflection to long term memory on completion` | 完成路径写复盘 |
| 21 | `feat: add context pruning and per node token accounting` | 上下文裁剪 + token 统计入 state |
| 22 | `test: cover cross session memory and token metrics` | 集成测试 |

**Tag:** `v0.4.0-phase3`

### Phase 4 — RAG 子能力（5 commits）

| # | Commit message | 交付物 |
|---|---|---|
| 23 | `feat: add document loaders and text splitter` | pdf / docx / txt + 切分 + 单测 |
| 24 | `feat: add chroma vector store adapter and indexing pipeline` | 索引管道 + `VectorStore` Protocol + 单测 |
| 25 | `feat: add retrieve tool with citation builder` | 检索工具 + 引用构造 |
| 26 | `feat: enforce citation iron rules and abstention` | 引用只来自命中 / 无证据拒答 + 测试 |
| 27 | `test: cover retrieval chat integration with synthetic documents` | 集成测试（合成 fixture，禁止真实文档） |

**Tag:** `v0.5.0-phase4`

### Phase 5 — Agent Harness 评测（6 commits）

| # | Commit message | 交付物 |
|---|---|---|
| 28 | `feat: add eval dataset schema and synthetic task fixtures` | `evals/datasets/tasks_v1.jsonl`（30~50 条）+ schema 单测 |
| 29 | `feat: add eval runner with task success and stability metrics` | runner + 通过率 / 链路稳定性 |
| 30 | `feat: add hallucination and token cost metrics` | 引用支撑率 + 成本 / 延迟 p50/p95 |
| 31 | `feat: add single agent vs multi agent comparison mode` | `--mode single\|multi` |
| 32 | `test: cover evaluation harness with scripted model` | 替身模型下全指标可复现 |
| 33 | `docs: embed evaluation results and ablation table in README` | **替身 + 真实各跑一轮，真实数字写入 README（标注环境）** |

**Tag:** `v0.6.0-phase5` — A/B 对比表完成前，面试中不主动深挖评测

### Phase 6 — 可观测性与生产化（6 commits）

| # | Commit message | 交付物 |
|---|---|---|
| 34 | `feat: add structured json logging with request correlation` | Json formatter + X-Request-ID + 单测 |
| 35 | `feat: add node level trace recording and replay endpoint` | span 落库 + `GET /progress/{task_id}` 回放 |
| 36 | `feat: add metrics endpoint for success latency and cost` | `GET /metrics` |
| 37 | `feat: add feedback endpoint and static web ui` | up/down 反馈 + 极简任务面板（消费现有 API） |
| 38 | `feat: add dockerfile and compose deployment` | Compose 卷挂载 `.data` |
| 39 | `docs: add runbook with security checklist` | runbook + README 定稿 |

**Tag:** `v1.0.0-phase6` — 主线完成

### Phase 7（可选，默认不排）— 1 commit

```text
feat: add scheduled task execution with heartbeat loop
```

**计划内 commit 总数：39（+Phase 7 可选 1）；tag 总数：7。**

### 新仓库启动步骤（用户手动执行一次）

```bash
# 1. 在空目录初始化（远端仓库先建好）
git init -b main
git remote add origin <你的远端地址>

# 2. 拷入两份契约文件（不提交）
#    CLAUDE.md           → 仓库根
#    求职项目设计方案.md   → docs/plan.md

# 3. 启动实现 Claude，由它按本编排表执行 commit #1
#    （补 docs/architecture.md 初稿后提交）
```

---

## 6. 开发守则（摘要；完整契约见根 `CLAUDE.md`，冲突以 `CLAUDE.md` 为准）

1. 每次任务前先读 `CLAUDE.md` 与 `docs/plan.md`；一次只做一个编排表内的步骤，禁止跨 Phase 批量生成文件。
2. 顺序：写实现 → 写对应测试 → 跑四件套 → 全绿才 commit → Phase 结束打 tag 并汇报（修改文件 / 设计决策 / 执行命令 / 测试结果 / 未解决问题）。
3. 四件套：`uv run ruff check .` · `uv run ruff format --check .` · `uv run mypy src` · `uv run pytest`。未过门禁不许提交；不许删测试、跳过测试或降 lint 标准凑绿。
4. 架构红线：domain 零框架依赖；API 无业务逻辑；配置只在 `config/settings.py`；依赖注入不用全局单例；无循环 import；无空模块。
5. Agent 红线：每个循环有计数上限与终止路径；模型输出解析失败走受控 fallback；引用不许模型编造；检索内容是数据不是指令；不存隐藏推理。
6. 安全：不提交 `.env` / `.data/`；日志不记密钥；测试只用合成数据；用户可见错误不泄堆栈。
7. 禁止：引入与当前步骤无关的依赖；伪造测试或评估数据；**依赖任何本地其他项目 / 仓库**。
8. 提交：subject 逐字使用第 5 节编排表；末尾带 Co-Authored-By 行；测试与实现同 commit。
9. 远端：`main` 单分支；每个 Phase 门禁绿后 `git push origin main --tags`；禁止 force push。
10. 完成定义：四件套绿 + 该 Phase 验收项全勾 + tag 已推 + README「阶段进度」更新。

---

## 7. 验收总表（逐 Phase 打卡）

- [x] P0：四件套 + CI 就位；`uv run uvicorn agent_workbench.main:app` 可起服务
- [x] P1：单 Agent 任务端到端（工具调用、超限终止），SSE 可见过程
- [x] P2：Supervisor 三路径测试绿 + 断点续跑演示（kill 进程 → 重启 → 续跑）
- [x] P3：跨会话记忆召回演示 + token 指标落库
- [x] P4：上传合成文档 → 检索 → 带引用回答 → 无证据拒答
- [x] P5：`run_evaluation.py --mode single|multi` 输出对比表，README 有真实数字（标注环境）
- [x] P6：`/metrics` 三指标可读、trace 可回放、`docker compose up` 一键起（compose config 已验证；本机 daemon 未启时未实际 up，见 runbook）
- [x] 全程：39 个语义化 commit、7 个 tag、无脏提交（不含 `.env`、`.data/`、密钥）
