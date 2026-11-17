"""长期记忆（SQLite 复盘仓库）单测。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workbench.memory import (
    LongTermMemoryRepository,
    SqliteLongTermMemory,
    TaskReflection,
)


@pytest.fixture
def repo(tmp_path: Path) -> SqliteLongTermMemory:
    """临时文件库。"""
    mem = SqliteLongTermMemory(tmp_path / "memory.db")
    yield mem
    mem.close()


def _reflection(**overrides: object) -> TaskReflection:
    base: dict = {
        "task_id": "t-1",
        "task_description": "写向量库对比报告",
        "summary": "Chroma 适合原型；生产需评估过滤性能",
        "outcome": "completed",
    }
    base.update(overrides)
    return TaskReflection(**base)  # type: ignore[arg-type]


class TestTaskReflection:
    """复盘模型。"""

    def test_defaults(self) -> None:
        r = _reflection()
        assert r.id
        assert r.outcome == "completed"
        assert r.created_at.tzinfo is not None

    def test_invalid_outcome_rejected(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            _reflection(outcome="unknown")

    def test_outcomes_accepted(self) -> None:
        for outcome in ("completed", "failed", "degraded"):
            assert _reflection(outcome=outcome).outcome == outcome


class TestSqliteRepository:
    """SQLite 实现。"""

    def test_save_and_get_roundtrip(self, repo: SqliteLongTermMemory) -> None:
        r = _reflection()
        repo.save(r)
        loaded = repo.get_by_task_id("t-1")
        assert loaded is not None
        assert loaded.task_id == "t-1"
        assert loaded.summary == r.summary
        assert loaded.created_at == r.created_at

    def test_get_missing_returns_none(self, repo: SqliteLongTermMemory) -> None:
        assert repo.get_by_task_id("nope") is None

    def test_save_same_task_id_overwrites(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection(summary="第一版"))
        repo.save(_reflection(summary="修订版", task_id="t-1"))
        loaded = repo.get_by_task_id("t-1")
        assert loaded is not None
        assert loaded.summary == "修订版"

    def test_search_by_keyword(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection())
        repo.save(
            _reflection(
                task_id="t-2",
                task_description="整理部署清单",
                summary="Docker 卷挂载 .data 目录",
            )
        )
        hits = repo.search("Docker")
        assert len(hits) == 1
        assert hits[0].task_id == "t-2"

    def test_search_matches_description(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection())
        hits = repo.search("向量库")
        assert len(hits) == 1

    def test_search_empty_query_lists_recent(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection())
        assert len(repo.search("")) == 1

    def test_search_limit(self, repo: SqliteLongTermMemory) -> None:
        for i in range(10):
            repo.save(_reflection(task_id=f"t-{i}", summary=f"要点{i}"))
        assert len(repo.search("要点", limit=3)) == 3

    def test_search_no_match(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection())
        assert repo.search("不存在的词xyz") == []

    def test_list_recent_order(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection(task_id="t-old", summary="旧"))
        repo.save(_reflection(task_id="t-new", summary="新"))
        recent = repo.list_recent(limit=5)
        ids = [r.task_id for r in recent]
        assert ids.index("t-new") < ids.index("t-old")

    def test_list_recent_limit(self, repo: SqliteLongTermMemory) -> None:
        for i in range(5):
            repo.save(_reflection(task_id=f"x-{i}"))
        assert len(repo.list_recent(limit=2)) == 2

    def test_protocol_runtime_check(self, repo: SqliteLongTermMemory) -> None:
        """实现满足 Protocol 结构。"""
        assert isinstance(repo, LongTermMemoryRepository)

    def test_created_at_is_utc(self, repo: SqliteLongTermMemory) -> None:
        repo.save(_reflection())
        loaded = repo.get_by_task_id("t-1")
        assert loaded is not None
        assert loaded.created_at.tzinfo is not None
        assert loaded.created_at.utcoffset().total_seconds() == 0

    def test_context_manager(self, tmp_path: Path) -> None:
        with SqliteLongTermMemory(tmp_path / "cm.db") as mem:
            mem.save(_reflection())
            assert mem.get_by_task_id("t-1") is not None
