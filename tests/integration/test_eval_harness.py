"""Phase 5 集成测试：替身模型下评测全指标可复现。

契约：ScriptedChatModel + FakeEmbeddings 离线确定性；
延迟分位受调度影响，比较时剔除 latency_* 字段。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_workbench.harness import (
    AggregateMetrics,
    compare_modes,
    format_comparison_markdown,
    load_dataset,
    run_dataset,
)
from agent_workbench.harness.dataset import DEFAULT_DATASET_PATH
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.main import build_default_registry


def _load_tasks() -> Any:
    root = Path(__file__).resolve().parents[2]
    return load_dataset(root / DEFAULT_DATASET_PATH)


def _factory() -> ScriptedChatModel:
    """新替身实例（空脚本 → 默认话术）。"""
    return ScriptedChatModel(scripted_responses=[])


def _stable_dump(metrics: AggregateMetrics) -> dict[str, Any]:
    """剔除延迟字段后的可比较 dict。"""
    data = metrics.model_dump()
    data.pop("latency_p50_ms", None)
    data.pop("latency_p95_ms", None)
    return data


def _stable_deltas(deltas: dict[str, float]) -> dict[str, float]:
    """剔除延迟相关 delta。"""
    return {k: v for k, v in deltas.items() if not k.startswith("latency")}


class TestFullMetricsReproducible:
    """全指标（除延迟）两次运行完全一致。"""

    def test_single_mode_all_metrics_stable(self) -> None:
        tasks = _load_tasks()
        registry = build_default_registry()

        _r1, m1 = run_dataset(tasks, _factory(), registry, mode="single", max_steps=3)
        _r2, m2 = run_dataset(tasks, _factory(), registry, mode="single", max_steps=3)

        assert _stable_dump(m1) == _stable_dump(m2)

    def test_multi_mode_all_metrics_stable(self) -> None:
        tasks = _load_tasks()
        registry = build_default_registry()

        _r1, m1 = run_dataset(tasks, _factory(), registry, mode="multi", max_steps=3)
        _r2, m2 = run_dataset(tasks, _factory(), registry, mode="multi", max_steps=3)

        assert _stable_dump(m1) == _stable_dump(m2)

    def test_all_metric_fields_present(self) -> None:
        """聚合指标覆盖契约要求的通过率/稳定性/成本/延迟/幻觉字段。"""
        tasks = _load_tasks()
        registry = build_default_registry()
        _runs, m = run_dataset(tasks, _factory(), registry, mode="single", max_steps=3)

        expected = {
            "task_count",
            "pass_count",
            "task_success_rate",
            "step_success_rate",
            "retry_rate",
            "mean_repairs",
            "token_input_sum",
            "token_output_sum",
            "token_total_sum",
            "mean_token_total",
            "latency_p50_ms",
            "latency_p95_ms",
            "citation_support_rate",
            "hallucination_rate",
            "citation_sample_count",
            "normal_count",
            "should_fail_count",
        }
        data = m.model_dump()
        assert expected <= set(data)

    def test_run_results_have_cost_and_support_fields(self) -> None:
        tasks = _load_tasks()[:5]
        registry = build_default_registry()
        runs, m = run_dataset(tasks, _factory(), registry, mode="single", max_steps=3)
        assert len(runs) == 5
        for r in runs:
            assert r.token_total == r.token_input + r.token_output
            assert r.duration_ms >= 0
            assert r.citation_support_rate is None or 0.0 <= r.citation_support_rate <= 1.0
        assert m.token_total_sum == sum(r.token_total for r in runs)
        assert m.latency_p50_ms > 0
        assert m.latency_p95_ms >= m.latency_p50_ms

    def test_json_roundtrip_of_full_metrics(self) -> None:
        tasks = _load_tasks()[:8]
        registry = build_default_registry()
        _runs, m = run_dataset(tasks, _factory(), registry, mode="single", max_steps=3)
        raw = json.dumps(m.model_dump(), ensure_ascii=False)
        back = json.loads(raw)
        assert back["task_count"] == m.task_count
        assert back["hallucination_rate"] == m.hallucination_rate
        assert back["token_total_sum"] == m.token_total_sum


class TestCompareHarness:
    """compare 模式在替身下稳定。"""

    def test_compare_deterministic_without_latency(self) -> None:
        tasks = _load_tasks()
        registry = build_default_registry()

        r1 = compare_modes(tasks, _factory, registry, max_steps=3)
        r2 = compare_modes(tasks, _factory, registry, max_steps=3)

        assert _stable_dump(r1.single) == _stable_dump(r2.single)
        assert _stable_dump(r1.multi) == _stable_dump(r2.multi)
        # deltas 剔除 latency 后应完全一致
        assert _stable_deltas(r1.deltas) == _stable_deltas(r2.deltas)

    def test_compare_covers_ab_fields(self) -> None:
        tasks = _load_tasks()
        report = compare_modes(tasks, _factory, build_default_registry(), max_steps=2)
        for key in (
            "task_success_rate",
            "step_success_rate",
            "token_total_sum",
            "hallucination_rate",
        ):
            assert key in report.deltas
        assert report.deltas["task_success_rate"] == round(
            report.multi.task_success_rate - report.single.task_success_rate, 4
        )

    def test_markdown_ab_table_renderable(self) -> None:
        tasks = _load_tasks()
        report = compare_modes(tasks, _factory, build_default_registry(), max_steps=2)
        md = format_comparison_markdown(report)
        lines = md.splitlines()
        assert lines[0].startswith("| 指标 |")
        assert "single" in lines[0] and "multi" in lines[0]
        assert any("任务通过率" in ln for ln in lines)
        assert any("token 合计" in ln for ln in lines)


class TestCliCompareIntegration:
    """CLI compare 端到端（替身、离线）。"""

    def test_cli_compare_json_stable(self, tmp_path: Path) -> None:
        from evals.run_evaluation import main

        out1 = tmp_path / "a.json"
        out2 = tmp_path / "b.json"
        for out in (out1, out2):
            code = main(["--mode", "compare", "--json-out", str(out), "--max-steps", "2"])
            assert code == 0

        p1 = json.loads(out1.read_text(encoding="utf-8"))
        p2 = json.loads(out2.read_text(encoding="utf-8"))

        def strip_latency(d: dict[str, Any]) -> dict[str, Any]:
            single = {k: v for k, v in d["single"].items() if not k.startswith("latency")}
            multi = {k: v for k, v in d["multi"].items() if not k.startswith("latency")}
            deltas = {k: v for k, v in d["deltas"].items() if not k.startswith("latency")}
            return {"single": single, "multi": multi, "deltas": deltas}

        assert strip_latency(p1) == strip_latency(p2)
        assert p1["mode"] == "compare"
        assert p1["single"]["task_count"] >= 30
