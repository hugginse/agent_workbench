"""评测 runner 与指标单测（替身、确定性）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_workbench.harness import (
    EvalTask,
    RunResult,
    default_judge,
    load_dataset,
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
        """同数据集同替身两次运行 metrics 一致。"""
        root = Path(__file__).resolve().parents[2]
        tasks = load_dataset(root / DEFAULT_DATASET_PATH)

        def once() -> Any:
            llm = ScriptedChatModel(scripted_responses=[])
            registry = build_default_registry()
            _runs, metrics = run_dataset(tasks, llm, registry, mode="single", max_steps=4)
            return metrics

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
