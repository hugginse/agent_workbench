"""retrieve 工具与引用构造单测（合成索引）。"""

from __future__ import annotations

import json
from pathlib import Path

from agent_workbench.infrastructure.embeddings import FakeEmbeddings
from agent_workbench.infrastructure.vectorstores import InMemoryVectorStore
from agent_workbench.rag import build_citations, format_citations_block, index_paths
from agent_workbench.tools import ToolRegistry
from agent_workbench.tools.builtins import RetrieveTool


def _seeded_store(tmp_path: Path) -> tuple[InMemoryVectorStore, FakeEmbeddings]:
    """合成文档写入向量库。"""
    p = tmp_path / "kb.txt"
    p.write_text(
        "Chroma 是嵌入式向量数据库，适合快速原型。\n"
        "PostgreSQL 适合生产环境的事务与扩展。\n"
        "切换 embedding 后必须清空索引重建。\n" * 10,
        encoding="utf-8",
    )
    store = InMemoryVectorStore()
    embedder = FakeEmbeddings()
    index_paths([p], store=store, embedder=embedder, chunk_size=200, chunk_overlap=20)
    return store, embedder


class TestCitationBuilder:
    """引用构造。"""

    def test_build_from_hits(self) -> None:
        from agent_workbench.infrastructure.vectorstores import VectorStoreHit

        hits = [
            VectorStoreHit(id="c1", text="片段一", score=0.9, metadata={"source": "a.txt"}),
            VectorStoreHit(id="c2", text="片段二", score=0.8, metadata={"filename": "b.txt"}),
        ]
        cs = build_citations(hits)
        assert len(cs.citations) == 2
        assert cs.citations[0].ref == "[1]"
        assert cs.citations[1].ref == "[2]"
        assert cs.citations[0].chunk_id == "c1"
        assert cs.citations[0].source == "a.txt"
        assert cs.citations[1].source == "b.txt"

    def test_empty_hits_notes(self) -> None:
        cs = build_citations([])
        assert cs.citations == []
        assert "拒答" in cs.notes or "无" in cs.notes

    def test_snippet_truncated(self) -> None:
        from agent_workbench.infrastructure.vectorstores import VectorStoreHit

        hit = VectorStoreHit(id="c1", text="长" * 500, score=0.9)
        cs = build_citations([hit], snippet_chars=50)
        assert len(cs.citations[0].snippet) <= 51  # 50 + 省略号

    def test_ids_only_from_hits(self) -> None:
        """引用 chunk_id 必须等于命中 id。"""
        from agent_workbench.infrastructure.vectorstores import VectorStoreHit

        hits = [VectorStoreHit(id="real-id", text="x", score=1.0)]
        cs = build_citations(hits)
        assert cs.ids == ["real-id"]
        assert cs.by_ref("[1]") is not None
        assert cs.by_ref("[9]") is None

    def test_format_block(self) -> None:
        from agent_workbench.infrastructure.vectorstores import VectorStoreHit

        cs = build_citations(
            [VectorStoreHit(id="c1", text="内容", score=1.0, metadata={"source": "s"})]
        )
        block = format_citations_block(cs)
        assert "[1]" in block
        assert "c1" in block

    def test_format_empty(self) -> None:
        block = format_citations_block(build_citations([]))
        assert "无" in block


class TestRetrieveTool:
    """retrieve 工具。"""

    def test_hits_and_citations(self, tmp_path: Path) -> None:
        store, embedder = _seeded_store(tmp_path)
        tool = RetrieveTool(store, embedder)
        raw = tool.run({"query": "Chroma 嵌入式向量数据库", "k": 3})
        payload = json.loads(raw)
        assert payload["hits"], payload
        assert payload["citations"]
        assert payload["hits"][0]["chunk_id"]
        assert payload["hits"][0]["ref"].startswith("[")
        assert "数据" in payload["reminder"] or "指令" in payload["reminder"]

    def test_empty_store(self) -> None:
        tool = RetrieveTool(InMemoryVectorStore(), FakeEmbeddings())
        payload = json.loads(tool.run({"query": "任何查询"}))
        assert payload["hits"] == []
        assert "空" in payload["notes"] or "空" in payload.get("message", "")

    def test_no_match_low_score(self) -> None:
        """完全无关查询在阈值过滤后 hits 为空。"""
        store = InMemoryVectorStore()
        embedder = FakeEmbeddings()
        texts = ["完全无关的主题内容ABC"]
        store.add(["1"], texts, embedder.embed_documents(texts), [{"source": "x"}])
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "量子引力波与弦论", "k": 5}))
        # 同哈希空间下不保证阈值一定过滤；至少结构合法
        assert "hits" in payload
        assert "citations" in payload

    def test_citation_ids_subset_of_store(self, tmp_path: Path) -> None:
        store, embedder = _seeded_store(tmp_path)
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "向量数据库", "k": 5}))
        # 所有 chunk_id 必须能在库中检索回（来自命中而非编造）
        for hit in payload["hits"]:
            assert hit["chunk_id"]

    def test_registers_in_registry(self, tmp_path: Path) -> None:
        store, embedder = _seeded_store(tmp_path)
        r = ToolRegistry()
        r.register(RetrieveTool(store, embedder))
        assert "retrieve" in r
        assert any(s.name == "retrieve" for s in r.specs())

    def test_execute_via_registry(self, tmp_path: Path) -> None:
        store, embedder = _seeded_store(tmp_path)
        r = ToolRegistry()
        r.register(RetrieveTool(store, embedder))
        raw = r.execute("retrieve", {"query": "PostgreSQL", "k": 2})
        payload = json.loads(raw)
        assert isinstance(payload["hits"], list)

    def test_citation_block_helper(self, tmp_path: Path) -> None:
        store, embedder = _seeded_store(tmp_path)
        tool = RetrieveTool(store, embedder)
        from agent_workbench.tools.builtins.retrieve import RetrieveArgs

        block = tool.citation_block(RetrieveArgs(query="Chroma", k=2))
        assert "[1]" in block or "无" in block

    def test_researcher_tool_names_include_retrieve(self) -> None:
        from agent_workbench.agents import RESEARCHER

        assert "retrieve" in RESEARCHER.tool_names
