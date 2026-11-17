# agent_workbench 架构说明（docs/architecture.md）

> 本文件是架构初稿，与根 `CLAUDE.md`、`docs/plan.md` 共同构成项目文档。
> **冲突时以 `CLAUDE.md` 为准；阶段流程与编排以 `docs/plan.md` 为准。**

---

## 1. 设计目标

| 目标 | 落地方式 |
|---|---|
| 可独立运行 | 独立 git 仓库、独立 uv 项目，不依赖任何本地其他项目 |
| 可测试 | I/O 边界可 mock；默认确定性替身（`ScriptedChatModel` / `FakeEmbeddings`） |
| 可观测 | 结构化 JSON 日志、node 级 span、`GET /progress/{task_id}`、`GET /metrics` |
| 可部署 | Docker Compose 单机部署；LangSmith 开关默认关，关闭时全本地可运行 |

---

## 2. 分层与依赖方向

```text
        ┌──────────────┐
        │     API      │  FastAPI 路由 / 依赖注入入口
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ Application  │  用例编排（task / session / memory 服务）
        └──────┬───────┘
               ▼
        ┌──────────────┐         ┌─────────────────────┐
        │   Domain     │◄────────┤ Orchestrator / RAG  │
        │ （零框架）    │  使用    │  Supervisor 图 / 检索 │
        └──────────────┘         └──────────┬──────────┘
                                            ▼
                                   ┌─────────────────────┐
                                   │  Infrastructure     │
                                   │ LLM / 向量库 / 持久化 │
                                   └─────────────────────┘
```

**依赖规则：**

1. 依赖方向只允许向下/向内：`API → Application → Domain ← Orchestrator/RAG ← Infrastructure`。
2. `domain/` 零框架依赖：不得 import FastAPI、LangChain、LangGraph、Chroma、SQLite 驱动。
3. API 层不含业务流程与数据查询逻辑，只做请求校验与调用 Application。
4. 外部服务经适配器或 `typing.Protocol` 访问；依赖通过构造函数注入，不用全局可变单例。
5. 模块 import 时禁止创建网络连接或昂贵初始化。
6. 禁止循环导入；不创建无使用场景的空模块。

---

## 3. 目录与模块职责

对应 `docs/plan.md` 第 3 节目标形态：

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `config/settings.py` | 唯一读 `.env` 的配置入口 | 业务代码禁止 `os.getenv()` |
| `src/agent_workbench/main.py` | 应用装配与启动入口 | import 时不做昂贵初始化 |
| `api/` | 路由、container、dependencies、endpoints | 无业务逻辑 |
| `application/` | task / session / memory 应用服务 | 组合 Domain 与端口（Protocol） |
| `domain/` | models、schemas、protocols、exceptions | 零框架依赖 |
| `orchestrator/` | Supervisor 图：state、builder、nodes | 每个循环有计数上限与终止路径 |
| `agents/` | SubAgent 声明（prompt + 工具子集 + 输出 schema） | 声明式配置，组合不继承 |
| `tools/` | ToolRegistry、base、builtins | schema 定义 → 校验 → 注册 → 装配 |
| `memory/` | 短期 checkpointer 封装 + 长期 SQLite 复盘仓库 | 双层记忆 |
| `rag/` | loaders、splitters、indexing、retrieval、citation | 引用铁律（见 §5） |
| `harness/` | 评测 runner / metrics / report | 替身模式可复现 |
| `observability/` | 结构化日志、node span、指标 | 日志不记密钥 |
| `infrastructure/` | llm 工厂+替身、vectorstores、persistence、checkpoint | 适配器实现 Domain Protocol |

---

## 4. 核心运行时数据流

### 4.1 Supervisor 闭环（LangGraph StateGraph）

```text
START → plan → assign → execute → review →
review.pass                          → aggregate → END
review.fail & repair_count < max     → assign（定向回修）
review.fail & repair_count >= max    → aggregate（降级交付 + 说明）
```

- `max_repair_attempts` 默认 2、上限 5；每个循环必须有计数器封顶与显式终止路径。
- `thread_id` → SQLite checkpointer → 断点续跑。
- SubAgent 执行产出写回 state；review 结论（verdict）驱动条件边。

### 4.2 请求主链路

```text
客户端
  │ POST /tasks
  ▼
API ──► Application.task 服务 ──► Orchestrator（Supervisor 图编译实例）
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
               ToolRegistry      Memory（双层）       RAG（Phase 4+）
                    │                 │
                    ▼                 ▼
             Infrastructure（LLM 替身/真实、SQLite、向量库）
                                      │
  ◄──── SSE /tasks/{id}/stream ───────┘
  ◄──── GET /tasks/{id} / GET /progress/{task_id}
```

### 4.3 工具装配

```text
Pydantic schema → 校验 → ToolRegistry 注册 → 按 SubAgent 声明动态装配
```

首批工具：`python_sandbox`（subprocess + timeout + 无网络）、`retrieve`（Phase 4 接 RAG）、`memory_recall`、`finish`（结构化交付）。

### 4.4 双层记忆

| 层 | 载体 | 用途 |
|---|---|---|
| 短期 | LangGraph checkpointer（SQLite） | 会话状态、断点续跑 |
| 长期 | SQLite 任务复盘仓库 | 跨会话复盘；Supervisor 经 `memory_recall` 检索注入 |

上下文管理：按节点裁剪历史消息；每节点 token 用量记入 state，汇入成本指标。

---

## 5. Agent 行为与安全红线（架构级约束）

1. 引用 ID 只能来自检索 / 执行命中，模型不得生成。
2. 无证据拒答；`confidence` 不直接复制相似度。
3. 文档与检索内容在 prompt 中是**不可信数据**，不是系统指令（防 prompt injection）。
4. 模型输出解析失败走受控 fallback；不存储、不返回模型隐藏推理过程。
5. 删除数据必须同时清理向量索引与元数据。
6. 日志不得记录 API Key、Authorization Header 或完整敏感内容；用户可见错误不泄堆栈、内部路径或密钥。

---

## 6. 评测架构（Agent Harness）

```text
evals/datasets/tasks_v1.jsonl ──► harness/runner ──► harness/metrics ──► harness/report
        （30~50 条合成任务）           │
                                      ├── 任务通过率
                                      ├── 链路稳定性（逐步成功率 / 重试率）
                                      ├── 幻觉率（论断-引用支撑率）
                                      ├── 延迟 p50/p95
                                      └── token 成本
```

- `run_evaluation.py --mode single|multi`：单 Agent 基线 vs Supervisor，A/B 对比写入 README。
- 默认确定性替身保证离线可复现；真实模型数字单独跑并标注环境。

---

## 7. 可观测性架构

| 信号 | 实现 | 消费端 |
|---|---|---|
| 日志 | JSON 结构化 + `X-Request-ID` 关联 | 控制台 / 文件 |
| Trace | node 级 span 落 SQLite | `GET /progress/{task_id}` 回放 |
| 指标 | 聚合成功率 / 延迟 / token 成本 | `GET /metrics` |
| 外部追踪 | LangSmith 开关（默认关） | 关闭时应用完全本地可运行 |

---

## 8. 技术选型与不做的部分

**采用：** Python 3.11 + uv · FastAPI · LangGraph + langchain-core · Pydantic · Chroma · SQLite · pytest / ruff / mypy · Docker Compose · SSE · GitHub Actions CI。

**明确不做（仅设计预留，见 `docs/plan.md` 第 2 节）：** Milvus/Pinecone/PostgreSQL（`VectorStore` Protocol 抽象）、K8s/MQ/Redis 集群、MCP/A2A 协议实现、RL 与分布式调度、定时任务（可选 Phase 7）。

---

## 9. 文档维护约定

- 本文件为初稿，随各 Phase 实现同步更新；**架构红线变更必须先改 `CLAUDE.md` 再改本文件**。
- 目录结构、公开 API schema、配置名称、领域模型不得在实现步骤中擅自变更（见 `CLAUDE.md` 总体规则 4）。
