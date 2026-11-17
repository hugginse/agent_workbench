"""引用铁律与无证据拒答单测。"""

from __future__ import annotations

import json
from pathlib import Path

from agent_workbench.infrastructure.embeddings import FakeEmbeddings
from agent_workbench.infrastructure.vectorstores import InMemoryVectorStore, VectorStoreHit
from agent_workbench.rag import build_citations
from agent_workbench.rag.guard import (
    ABSTAIN_ANSWER,
    check_answer,
    confidence_from_support,
    enforce_answer,
)
from agent_workbench.tools.builtins import RetrieveTool


def _citations(*hit_ids: str) -> object:
    hits = [
        VectorStoreHit(id=i, text=f"内容-{i}", score=0.9, metadata={"source": "s.txt"})
        for i in hit_ids
    ]
    return build_citations(hits)


class TestCheckAnswer:
    """答案合规检查。"""

    def test_fabricated_ref_rejected(self) -> None:
        """编号 [99] 不在命中集合 → 不通过。"""
        cs = _citations("c1", "c2")
        check = check_answer("结论正确 [99]。", cs)  # type: ignore[arg-type]
        assert check.ok is False
        assert "[99]" in check.unsupported_refs

    def test_valid_refs_pass(self) -> None:
        cs = _citations("c1", "c2")
        check = check_answer("根据材料 [1]，结论是 A。", cs)  # type: ignore[arg-type]
        assert check.ok is True
        assert "[1]" in check.used_refs
        assert check.confidence > 0

    def test_no_evidence_must_abstain(self) -> None:
        """无命中 + 实体答案 → 拒答违规。"""
        from agent_workbench.rag.citation import CitationSet

        empty = CitationSet(citations=[], notes="无命中")
        check = check_answer("答案是 42。", empty)
        assert check.ok is False
        assert check.must_abstain is True

    def test_no_evidence_already_abstained_ok(self) -> None:
        from agent_workbench.rag.citation import CitationSet

        empty = CitationSet(citations=[], notes="")
        check = check_answer(ABSTAIN_ANSWER, empty)
        assert check.ok is True
        assert check.must_abstain is True
        assert check.confidence == 0.0

    def test_hits_but_no_citation_rejected(self) -> None:
        """有命中但完全不引用 → 拒绝（要求引用）。"""
        cs = _citations("c1")
        check = check_answer("这就是结论，不给引用。", cs)  # type: ignore[arg-type]
        assert check.ok is False
        assert "未引用" in check.reason

    def test_suspicious_chunk_id_rejected(self) -> None:
        """正文出现非白名单 chunk_id → 编造嫌疑。"""
        cs = _citations("real-chunk-01")
        check = check_answer(
            "见 id=fabricated-xyz99 的材料 [1]",
            cs,  # type: ignore[arg-type]
        )
        assert check.ok is False
        assert check.fabricated_ids or check.unsupported_refs


class TestConfidenceNotCopyScore:
    """confidence 不得直接复制相似度。"""

    def test_confidence_differs_from_max_score(self) -> None:
        cs = _citations("c1", "c2", "c3")
        max_score = 0.97
        conf = confidence_from_support(cs, used_ref_count=1, max_score=max_score)
        assert conf != max_score
        assert conf < 0.95
        assert conf > 0

    def test_zero_usage_zero_conf(self) -> None:
        cs = _citations("c1")
        assert confidence_from_support(cs, used_ref_count=0) == 0.0

    def test_full_coverage_high_but_capped(self) -> None:
        cs = _citations("c1", "c2")
        conf = confidence_from_support(cs, used_ref_count=2, max_score=1.0)
        assert conf <= 0.95
        assert conf >= 0.9

    def test_check_confidence_not_equal_raw_score(self) -> None:
        cs = _citations("c1")
        check = check_answer("材料 [1] 表明如此。", cs, max_score=0.99)  # type: ignore[arg-type]
        assert check.ok
        assert check.confidence != 0.99
        assert check.confidence < 0.99


class TestEnforceAnswer:
    """强制改写：违规 → 拒答。"""

    def test_violation_becomes_abstain(self) -> None:
        cs = _citations("c1")
        final, check = enforce_answer("乱编 [77] 引用", cs)  # type: ignore[arg-type]
        assert check.ok is False
        assert final == ABSTAIN_ANSWER

    def test_valid_answer_kept(self) -> None:
        cs = _citations("c1")
        raw = "根据 [1] 可知结论。"
        final, check = enforce_answer(raw, cs)  # type: ignore[arg-type]
        assert check.ok is True
        assert final == raw

    def test_empty_evidence_abstain(self) -> None:
        from agent_workbench.rag.citation import CitationSet

        final, check = enforce_answer("直接给答案", CitationSet(citations=[]))
        assert final == ABSTAIN_ANSWER
        assert check.must_abstain


class TestRetrieveAbstainFlag:
    """retrieve 空证据时给出 abstain 信号。"""

    def test_empty_store_abstain_true(self) -> None:
        tool = RetrieveTool(InMemoryVectorStore(), FakeEmbeddings())
        payload = json.loads(tool.run({"query": "任意"}))
        assert payload["abstain"] is True
        assert "abstain_answer" in payload

    def test_seeded_store_abstain_false(self, tmp_path: Path) -> None:
        p = tmp_path / "k.txt"
        p.write_text("Chroma 向量数据库说明。" * 40, encoding="utf-8")
        store = InMemoryVectorStore()
        embedder = FakeEmbeddings()
        from agent_workbench.rag import index_paths

        index_paths([p], store=store, embedder=embedder)
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "Chroma 向量数据库", "k": 3}))
        if payload["hits"]:
            assert payload["abstain"] is False
        else:
            assert payload["abstain"] is True

    def test_end_to_end_guard_on_retrieve(self, tmp_path: Path) -> None:
        """检索 → 引用 → 强制校验 闭环。"""
        p = tmp_path / "k.txt"
        p.write_text("嵌入式数据库 Chroma 支持本地持久化。" * 30, encoding="utf-8")
        store = InMemoryVectorStore()
        embedder = FakeEmbeddings()
        from agent_workbench.rag import index_paths

        index_paths([p], store=store, embedder=embedder)
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "Chroma 本地持久化", "k": 3}))
        cs = build_citations(
            [
                VectorStoreHit(
                    id=h["chunk_id"],
                    text=h["text"],
                    score=h["score"],
                    metadata={"source": h["source"]},
                )
                for h in payload["hits"]
            ]
        )
        if not cs.citations:
            final, check = enforce_answer("没有证据的答案", cs)
            assert final == ABSTAIN_ANSWER
            assert check.must_abstain
        else:
            ref = cs.citations[0].ref
            final, check = enforce_answer(f"根据 {ref} 可知。", cs)
            assert check.ok
            assert final == f"根据 {ref} 可知。"
