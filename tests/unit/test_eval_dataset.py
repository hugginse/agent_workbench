"""评测数据集 schema 与合成 fixtures 单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_workbench.harness import (
    EvalTask,
    load_dataset,
    load_dataset_from_path,
    validate_dataset,
)
from agent_workbench.harness.dataset import (
    DEFAULT_DATASET_PATH,
    MAX_TASKS,
    MIN_TASKS,
    EvalDatasetError,
)


class TestEvalTaskSchema:
    """单条任务 schema。"""

    def test_minimal_valid(self) -> None:
        t = EvalTask(id="x1", task="做一件事", rubric="完成")
        assert t.should_fail is False
        assert t.expected_artifacts == []

    def test_full_fields(self) -> None:
        t = EvalTask(
            id="x2",
            task="写报告",
            expected_artifacts=["报告"],
            should_fail=True,
            rubric="应拒答",
        )
        assert t.should_fail is True

    def test_empty_task_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvalTask(id="x", task="", rubric="r")


class TestValidateDataset:
    """规模与配比校验。"""

    def _ok_tasks(self, n: int = MIN_TASKS, *, fails: int = 3) -> list[EvalTask]:
        tasks = []
        for i in range(n):
            tasks.append(
                EvalTask(
                    id=f"t{i:03d}",
                    task=f"任务{i}",
                    rubric="标准",
                    should_fail=i < fails,
                )
            )
        return tasks

    def test_accepts_valid_size(self) -> None:
        validate_dataset(self._ok_tasks())  # 30 条，3 条 fail

    def test_too_few_rejected(self) -> None:
        with pytest.raises(EvalDatasetError, match="30"):
            validate_dataset(self._ok_tasks(n=MIN_TASKS - 1, fails=1))

    def test_too_many_rejected(self) -> None:
        with pytest.raises(EvalDatasetError, match="50"):
            validate_dataset(self._ok_tasks(n=MAX_TASKS + 1))

    def test_duplicate_ids_rejected(self) -> None:
        tasks = self._ok_tasks()
        tasks[1] = tasks[0].model_copy()
        with pytest.raises(EvalDatasetError, match="重复"):
            validate_dataset(tasks)

    def test_requires_some_should_fail(self) -> None:
        tasks = self._ok_tasks(fails=0)
        with pytest.raises(EvalDatasetError, match="should_fail"):
            validate_dataset(tasks)

    def test_all_fail_rejected(self) -> None:
        tasks = [t.model_copy(update={"should_fail": True}) for t in self._ok_tasks()]
        with pytest.raises(EvalDatasetError, match="全部"):
            validate_dataset(tasks)


class TestDefaultDataset:
    """tasks_v1.jsonl 仓库内置数据集。"""

    def test_path_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        assert (root / DEFAULT_DATASET_PATH).exists()

    def test_loads_and_validates(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tasks = load_dataset(root / DEFAULT_DATASET_PATH)
        assert MIN_TASKS <= len(tasks) <= MAX_TASKS
        assert any(t.should_fail for t in tasks)
        assert any(not t.should_fail for t in tasks)

    def test_ids_unique_and_prefixed(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tasks = load_dataset(root / DEFAULT_DATASET_PATH)
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("t") for i in ids)

    def test_every_task_has_rubric_and_task(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for t in load_dataset(root / DEFAULT_DATASET_PATH):
            assert t.task.strip()
            assert t.rubric.strip()

    def test_should_fail_ratio_in_range(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tasks = load_dataset(root / DEFAULT_DATASET_PATH)
        fails = sum(1 for t in tasks if t.should_fail)
        # 30~50 中至少 1 条应失败，且不超过一半（避免全是坏样本）
        assert 1 <= fails <= len(tasks) // 2

    def test_jsonl_one_object_per_line(self) -> None:
        root = Path(__file__).resolve().parents[2]
        lines = [
            ln
            for ln in (root / DEFAULT_DATASET_PATH).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for ln in lines:
            obj = json.loads(ln)
            assert isinstance(obj, dict)
            assert {"id", "task", "rubric"} <= set(obj)

    def test_synthetic_no_pii_heuristic(self) -> None:
        """合成数据启发式：不含常见邮箱/手机号。"""
        import re

        root = Path(__file__).resolve().parents[2]
        text = (root / DEFAULT_DATASET_PATH).read_text(encoding="utf-8")
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        assert not re.search(r"1[3-9]\d{9}", text)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset_from_path(tmp_path / "nope.jsonl")

    def test_bad_json_line(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(EvalDatasetError, match="JSON"):
            load_dataset_from_path(p)
