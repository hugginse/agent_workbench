# evals — Agent Harness 评测

## 数据集

- `datasets/tasks_v1.jsonl`：**36 条合成任务**（含 `should_fail=true` 应失败样本）。
- Schema：`id / task / expected_artifacts / should_fail / rubric`（见 `agent_workbench.harness.dataset`）。
- **全部为合成数据**，禁止真实用户文档进入 fixture（契约）。

## 指标（计划）

| 指标 | 含义 |
|---|---|
| 任务通过率 | 非 should_fail 且 rubric 达标的比例 |
| 链路稳定性 | 逐步成功率 / 重试率 |
| 幻觉率 | 论断-引用支撑率的补 |
| 延迟 p50/p95 | 端到端耗时分位 |
| token 成本 | 来自 `token_usage` 累计 |

## 替身模式与限制（必读）

- 默认使用 `ScriptedChatModel` + `FakeEmbeddings`（**离线可复现**，固定种子语义）。
- 替身数字**不能**代表真实模型效果；真实模型须单独跑一轮并在根 README **标注环境**。
- 切换真实 embedding 后须清空 `.data/` 并重建索引。

## 通过判定（默认 judge）

- **普通任务**（`should_fail=false`）：`completed` 且 answer 非空 → 通过。
- **应失败样本**（`should_fail=true`）：`failed`，或 `completed` 且（拒答 / 降级 / 空答）→ 通过；
  给出自信实体答案 → 不通过。

## 运行

```bash
uv run python evals/run_evaluation.py --mode multi
uv run python evals/run_evaluation.py --mode single --json-out evals/last_result.json
uv run pytest tests/unit/test_eval_runner.py
```
