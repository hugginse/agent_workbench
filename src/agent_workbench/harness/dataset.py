"""评测数据集 schema 与加载（JSONL）。

Schema（plan §4.5）：`id / task / expected_artifacts / should_fail / rubric`。
数据必须为合成样本，禁止真实用户文档（契约）。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

__all__ = [
    "DEFAULT_DATASET_PATH",
    "EvalDatasetError",
    "EvalTask",
    "load_dataset",
    "load_dataset_from_path",
    "validate_dataset",
]

# 相对仓库根的默认数据集（evals/run_evaluation.py 等共用）
DEFAULT_DATASET_PATH = Path("evals") / "datasets" / "tasks_v1.jsonl"

# 数据集规模契约：30~50 条
MIN_TASKS = 30
MAX_TASKS = 50


class EvalDatasetError(ValueError):
    """数据集校验失败。"""


class EvalTask(BaseModel):
    """单条评测任务。

    Attributes:
        id: 唯一标识（如 t001）。
        task: 任务自然语言描述。
        expected_artifacts: 期望产出物关键词/结构（供 rubric 机检扩展）。
        should_fail: True 表示「应失败」样本（系统应拒答/优雅失败而非硬崩）。
        rubric: 评分标准。
    """

    id: str = Field(min_length=1, description="任务唯一 id")
    task: str = Field(min_length=1, description="任务描述")
    expected_artifacts: list[str] = Field(
        default_factory=list, description="期望产出物（关键词或结构名）"
    )
    should_fail: bool = Field(default=False, description="是否应失败样本")
    rubric: str = Field(min_length=1, description="评分标准")

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, v: str) -> str:
        """id 去除首尾空白后非空。"""
        if not v.strip():
            msg = "id 不得为空白"
            raise ValueError(msg)
        return v.strip()


def validate_dataset(tasks: list[EvalTask]) -> None:
    """校验数据集规模、id 唯一与基本配比。

    Args:
        tasks: 已解析的任务列表。

    Raises:
        EvalDatasetError: 数量越界、id 重复或全为应失败/全为正常。
    """
    if not MIN_TASKS <= len(tasks) <= MAX_TASKS:
        msg = f"数据集必须有 {MIN_TASKS}~{MAX_TASKS} 条，当前 {len(tasks)}"
        raise EvalDatasetError(msg)
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        msg = f"任务 id 重复: {dup}"
        raise EvalDatasetError(msg)
    fails = sum(1 for t in tasks if t.should_fail)
    if fails == 0:
        msg = "数据集必须包含 should_fail=true 的应失败样本"
        raise EvalDatasetError(msg)
    if fails == len(tasks):
        msg = "数据集不能全部是 should_fail 样本"
        raise EvalDatasetError(msg)


def load_dataset_from_path(path: str | Path) -> list[EvalTask]:
    """从 JSONL 文件加载并校验数据集。

    Args:
        path: JSONL 路径（每行一个 JSON 对象）。

    Returns:
        EvalTask 列表（保持文件顺序）。

    Raises:
        FileNotFoundError: 文件不存在。
        EvalDatasetError: 行 JSON 非法或 schema/规模校验失败。
    """
    p = Path(path)
    if not p.exists():
        msg = f"数据集文件不存在: {p}"
        raise FileNotFoundError(msg)

    tasks: list[EvalTask] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"第 {line_no} 行不是合法 JSON: {exc}"
            raise EvalDatasetError(msg) from exc
        if not isinstance(data, dict):
            msg = f"第 {line_no} 行必须是 JSON 对象"
            raise EvalDatasetError(msg)
        try:
            tasks.append(EvalTask.model_validate(data))
        except ValidationError as exc:
            msg = f"第 {line_no} 行 schema 校验失败: {exc}"
            raise EvalDatasetError(msg) from exc

    validate_dataset(tasks)
    return tasks


def load_dataset(path: str | Path | None = None) -> list[EvalTask]:
    """加载默认或指定数据集。

    Args:
        path: None 时使用 DEFAULT_DATASET_PATH（相对 CWD）。

    Returns:
        EvalTask 列表。
    """
    return load_dataset_from_path(path or DEFAULT_DATASET_PATH)
