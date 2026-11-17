"""评测 runner 与指标单测（替身、确定性）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_workbench.harness import (
    EvalTask,
    RunResult,
    citation_support_rate,
    default_judge,
    load_dataset,
    percentile,
    run_dataset,
    run_one,
    summarize_runs,
)
from agent_workbench.harness.dataset import DEFAULT_DATASET_PATH
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.main import build_default_registry


def _task(**overrides: object) -> EvalTask:
    base: dict[str, object] = {
        "id": "tX",
        "task": "普通任务",
        "rubric": "完成",
        "should_fail": False,
    }
    base.update(overrides)
    return EvalTask.model_validate(base)


class TestDefaultJudge:
    """默认通过判定。"""

    def test_normal_completed_pass(self) -> None:
        r = RunResult(
            task_id="t",
            should_fail=False,
            status="completed",
            answer="有内容",
        )
        assert default_judge(_task(), r) is True

    def test_normal_failed_no_pass(self) -> None:
        r = RunResult(task_id="t", should_fail=False, status="failed", error="X")
        assert default_judge(_task(), r) is False

    def test_normal_empty_answer_no_pass(self) -> None:
        r = RunResult(task_id="t", should_fail=False, status="completed", answer="  ")
        assert default_judge(_task(), r) is False

    def test_should_fail_on_failed_pass(self) -> None:
        r = RunResult(task_id="t", should_fail=True, status="failed", error="boom")
        assert default_judge(_task(should_fail=True), r) is True

    def test_should_fail_on_abstain_pass(self) -> None:
        r = RunResult(
            task_id="t",
            should_fail=True,
            status="completed",
            answer="知识库中没有证据，无法回答（拒答）。",
        )
        assert default_judge(_task(should_fail=True), r) is True

    def test_should_fail_on_degraded_pass(self) -> None:
        r = RunResult(
            task_id="t",
            should_fail=True,
            status="completed",
            answer="随便",
            degraded=True,
        )
        assert default_judge(_task(should_fail=True), r) is True

    def test_should_fail_with_confident_answer_no_pass(self) -> None:
        r = RunResult(
            task_id="t",
            should_fail=True,
            status="completed",
            answer="我认为答案就是 42，确定无疑。",
        )
        assert default_judge(_task(should_fail=True), r) is False


class TestSummarizeRuns:
    """聚合指标。"""

    def test_rates(self) -> None:
        runs = [
            RunResult(
                task_id="a",
                should_fail=False,
                status="completed",
                answer="x",
                passed=True,
                repair_count=0,
            ),
            RunResult(
                task_id="b",
                should_fail=False,
                status="failed",
                passed=False,
                repair_count=2,
            ),
            RunResult(
                task_id="c",
                should_fail=True,
                status="failed",
                passed=True,
                repair_count=1,
            ),
        ]
        m = summarize_runs(runs)
        assert m.task_count == 3
        assert m.pass_count == 2
        # summarize 四舍五入到 4 位小数
        assert m.task_success_rate == pytest.approx(2 / 3, abs=1e-4)
        assert m.step_success_rate == pytest.approx(1 / 3, abs=1e-4)
        assert m.retry_rate == pytest.approx(2 / 3, abs=1e-4)
        assert m.mean_repairs == pytest.approx(1.0, abs=1e-6)
        assert m.normal_count == 2
        assert m.normal_pass_count == 1
        assert m.should_fail_count == 1
        assert m.should_fail_pass_count == 1

    def test_empty_runs(self) -> None:
        m = summarize_runs([])
        assert m.task_count == 0
        assert m.task_success_rate == 0.0


class TestRunOneAndDataset:
    """替身模型下 runner 可复现。"""

    def test_run_one_single_mode_completes(self) -> None:
        llm = ScriptedChatModel(scripted_responses=["答案ABC"])
        registry = build_default_registry()
        r = run_one(_task(id="t001"), llm, registry, mode="single", max_steps=4)
        assert r.status == "completed"
        assert r.passed is True
        assert r.duration_ms >= 0

    def test_run_one_multi_mode_completes(self) -> None:
        llm = ScriptedChatModel(scripted_responses=[])
        registry = build_default_registry()
        r = run_one(_task(id="t002"), llm, registry, mode="multi", max_steps=4)
        assert r.status == "completed"
        assert r.error is None

    def test_dataset_deterministic(self) -> None:
        """同数据集同替身两次运行：除延迟分位外指标一致。"""
        root = Path(__file__).resolve().parents[2]
        tasks = load_dataset(root / DEFAULT_DATASET_PATH)

        def once() -> Any:
            llm = ScriptedChatModel(scripted_responses=[])
            registry = build_default_registry()
            _runs, metrics = run_dataset(tasks, llm, registry, mode="single", max_steps=4)
            data = metrics.model_dump()
            # 延迟受调度影响，不纳入确定性比较
            data.pop("latency_p50_ms", None)
            data.pop("latency_p95_ms", None)
            return data

        a = once()
        b = once()
        assert a == b

    def test_pass_rate_in_unit_interval(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tasks = load_dataset(root / DEFAULT_DATASET_PATH)
        llm = ScriptedChatModel(scripted_responses=[])
        registry = build_default_registry()
        runs, m = run_dataset(tasks, llm, registry, mode="single", max_steps=4)
        assert len(runs) == len(tasks)
        assert 0.0 <= m.task_success_rate <= 1.0
        assert m.task_count == len(tasks)

    def test_json_roundtrip_metrics(self) -> None:
        runs = [
            RunResult(
                task_id="a",
                should_fail=False,
                status="completed",
                answer="x",
                passed=True,
            )
        ]
        m = summarize_runs(runs)
        data = json.loads(m.model_dump_json())
        assert data["task_success_rate"] == 1.0


class TestCostLatencyHallucination:
    """成本 / 延迟分位 / 引用支撑率。"""

    def test_percentile_median(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
        assert percentile([10.0], 95) == 10.0
        assert percentile([1.0, 2.0], 0) == 1.0
        assert percentile([1.0, 2.0], 100) == 2.0

    def test_percentile_invalid(self) -> None:
        with pytest.raises(ValueError):
            percentile([], 50)
        with pytest.raises(ValueError):
            percentile([1.0], 101)

    def test_support_rate_supported(self) -> None:
        rate = citation_support_rate(
            "根据 [1] 与 [2] 可知。",
            allowed_refs=["[1]", "[2]"],
        )
        assert rate == 1.0

    def test_support_rate_partial(self) -> None:
        rate = citation_support_rate(
            "根据 [1] 与 [9] 可知。",
            allowed_refs=["[1]", "[2]"],
        )
        assert rate == pytest.approx(0.5, abs=1e-4)

    def test_support_rate_none_when_no_refs(self) -> None:
        assert citation_support_rate("普通陈述无引用。", allowed_refs=["[1]"]) is None

    def test_support_rate_abstain_is_one(self) -> None:
        rate = citation_support_rate(
            "知识库中没有足以支持该问题的证据，无法给出可靠回答（拒答）。",
            allowed_refs=None,
        )
        assert rate == 1.0

    def test_support_rate_unverified_refs_zero(self) -> None:
        """有引用但无合法集合 → 0（未支撑）。"""
        rate = citation_support_rate("看 [1] 与 [2]。", allowed_refs=None)
        assert rate == 0.0

    def test_summarize_token_and_latency(self) -> None:
        runs = [
            RunResult(
                task_id="a",
                should_fail=False,
                status="completed",
                answer="x",
                passed=True,
                duration_ms=10.0,
                token_input=100,
                token_output=50,
                token_total=150,
                citation_support_rate=1.0,
            ),
            RunResult(
                task_id="b",
                should_fail=False,
                status="completed",
                answer="y",
                passed=True,
                duration_ms=30.0,
                token_input=200,
                token_output=100,
                token_total=300,
                citation_support_rate=0.5,
            ),
        ]
        m = summarize_runs(runs)
        assert m.token_input_sum == 300
        assert m.token_output_sum == 150
        assert m.token_total_sum == 450
        assert m.mean_token_total == 225.0
        assert m.latency_p50_ms == pytest.approx(20.0, abs=0.1)
        assert m.latency_p95_ms >= m.latency_p50_ms
        assert m.citation_sample_count == 2
        assert m.citation_support_rate == pytest.approx(0.75, abs=1e-4)
        assert m.hallucination_rate == pytest.approx(0.25, abs=1e-4)

    def test_hallucination_zero_without_samples(self) -> None:
        m = summarize_runs(
            [
                RunResult(
                    task_id="a",
                    should_fail=False,
                    status="completed",
                    answer="x",
                    passed=True,
                    citation_support_rate=None,
                )
            ]
        )
        assert m.citation_support_rate is None
        assert m.hallucination_rate == 0.0
        assert m.citation_sample_count == 0

    def test_runner_records_tokens_and_latency(self) -> None:
        llm = ScriptedChatModel(scripted_responses=["带 token 的答案"])
        registry = build_default_registry()
        r = run_one(_task(id="t030"), llm, registry, mode="single", max_steps=4)
        assert r.duration_ms > 0
        assert r.token_input > 0
        assert r.token_total == r.token_input + r.token_output
        # 无引用非拒答 → 支撑率 None
        assert r.citation_support_rate is None


class TestCliSmoke:
    """run_evaluation.py 冒烟。"""

    def test_cli_json_out(self, tmp_path: Path) -> None:
        from evals.run_evaluation import main

        out = tmp_path / "result.json"
        code = main(
            [
                "--mode",
                "single",
                "--json-out",
                str(out),
                "--max-steps",
                "3",
            ]
        )
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["mode"] == "single"
        assert "metrics" in payload
        assert payload["metrics"]["task_count"] >= 30
