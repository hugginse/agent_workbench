"""向量库 Protocol、FakeEmbeddings 与索引管道单测（合成数据）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workbench.infrastructure.embeddings import FakeEmbeddings
from agent_workbench.infrastructure.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreHit,
)
from agent_workbench.rag import Document, index_document, index_paths


class TestFakeEmbeddings:
    """确定性替身向量。"""

    def test_same_input_same_vector(self) -> None:
        e = FakeEmbeddings()
        a = e.embed_query("同一文本")
        b = e.embed_query("同一文本")
        assert a == b

    def test_different_input_different_vector(self) -> None:
        e = FakeEmbeddings()
        a = e.embed_query("文本A")
        b = e.embed_query("文本B")
        assert a != b

    def test_dim(self) -> None:
        e = FakeEmbeddings(dim=32)
        v = e.embed_query("x")
        assert len(v) == 32

    def test_normalized(self) -> None:
        v = FakeEmbeddings().embed_query("归一化检查")
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_invalid_dim(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            FakeEmbeddings(dim=2)

    def test_embed_documents_batch(self) -> None:
        e = FakeEmbeddings()
        vecs = e.embed_documents(["a", "b", "a"])
        assert len(vecs) == 3
        assert vecs[0] == vecs[2]
        assert vecs[0] != vecs[1]


class TestInMemoryVectorStore:
    """内存向量库实现 Protocol。"""

    def test_satisfies_protocol(self) -> None:
        store = InMemoryVectorStore()
        assert isinstance(store, VectorStore)

    def test_add_and_count(self) -> None:
        store = InMemoryVectorStore()
        e = FakeEmbeddings()
        ids = ["1", "2"]
        texts = ["甲", "乙"]
        embs = e.embed_documents(texts)
        store.add(ids, texts, embs, [{"source": "a"}, {"source": "b"}])
        assert store.count() == 2

    def test_similarity_search_finds_match(self) -> None:
        store = InMemoryVectorStore()
        e = FakeEmbeddings()
        texts = ["向量数据库选型", "今天天气很好"]
        store.add(
            ["1", "2"],
            texts,
            e.embed_documents(texts),
            [{"source": "s1"}, {"source": "s2"}],
        )
        hits = store.similarity_search(e.embed_query("向量数据库选型"), k=1)
        assert len(hits) == 1
        assert isinstance(hits[0], VectorStoreHit)
        assert hits[0].text == "向量数据库选型"
        assert hits[0].score > 0.5

    def test_k_limits_results(self) -> None:
        store = InMemoryVectorStore()
        e = FakeEmbeddings()
        texts = [f"文档{i}" for i in range(10)]
        store.add([str(i) for i in range(10)], texts, e.embed_documents(texts))
        assert len(store.similarity_search(e.embed_query("文档0"), k=3)) == 3

    def test_delete_by_source(self) -> None:
        store = InMemoryVectorStore()
        e = FakeEmbeddings()
        store.add(
            ["1", "2"],
            ["甲", "乙"],
            e.embed_documents(["甲", "乙"]),
            [{"source": "keep"}, {"source": "drop"}],
        )
        removed = store.delete_by_source("drop")
        assert removed == 1
        assert store.count() == 1

    def test_empty_store_search(self) -> None:
        store = InMemoryVectorStore()
        assert store.similarity_search([0.1, 0.2], k=5) == []

    def test_length_mismatch_rejected(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="长度"):
            store.add(["1"], ["a", "b"], [[0.1, 0.2]])


class TestIndexPipeline:
    """索引管道。"""

    def test_index_document_into_store(self) -> None:
        store = InMemoryVectorStore()
        embedder = FakeEmbeddings()
        doc = Document(
            content="合成内容。" * 80,
            metadata={"filename": "合成.txt", "type": "txt"},
        )
        report = index_document(
            doc,
            store=store,
            embedder=embedder,
            chunk_size=100,
            chunk_overlap=10,
        )
        assert report.files == 1
        assert report.chunks >= 2
        assert report.sources == ["合成.txt"]
        assert store.count() == report.chunks

    def test_index_empty_document(self) -> None:
        store = InMemoryVectorStore()
        report = index_document(
            Document(content="", metadata={}),
            store=store,
            embedder=FakeEmbeddings(),
        )
        assert report.chunks == 0
        assert store.count() == 0

    def test_index_paths_txt(self, tmp_path: Path) -> None:
        p = tmp_path / "doc1.txt"
        p.write_text("合成文档正文。" * 50, encoding="utf-8")
        store = InMemoryVectorStore()
        report = index_paths([p], store=store, embedder=FakeEmbeddings())
        assert report.files == 1
        assert report.chunks > 0
        assert "doc1.txt" in report.sources

    def test_index_then_search_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "knowledge.txt"
        p.write_text(
            "Chroma 是一种嵌入式向量数据库，适合原型验证。\n" * 20,
            encoding="utf-8",
        )
        store = InMemoryVectorStore()
        embedder = FakeEmbeddings()
        index_paths([p], store=store, embedder=embedder)
        hits = store.similarity_search(embedder.embed_query("Chroma 嵌入式向量数据库"), k=2)
        assert hits
        assert any("Chroma" in h.text for h in hits)
        assert all(h.metadata.get("source") == "knowledge.txt" for h in hits)

    def test_delete_source_after_index(self, tmp_path: Path) -> None:
        p = tmp_path / "todel.txt"
        p.write_text("将被删除的合成内容。" * 30, encoding="utf-8")
        store = InMemoryVectorStore()
        index_paths([p], store=store, embedder=FakeEmbeddings())
        assert store.count() > 0
        store.delete_by_source("todel.txt")
        assert store.count() == 0

    def test_deterministic_indexing(self, tmp_path: Path) -> None:
        p = tmp_path / "det.txt"
        p.write_text("确定性索引测试。" * 40, encoding="utf-8")

        def run() -> list[str]:
            store = InMemoryVectorStore()
            embedder = FakeEmbeddings()
            # 固定 id 工厂便于对比
            counter = {"n": 0}

            def factory() -> str:
                counter["n"] += 1
                return f"id-{counter['n']}"

            index_paths([p], store=store, embedder=embedder, id_factory=factory)
            hits = store.similarity_search(embedder.embed_query("确定性"), k=3)
            return [h.id for h in hits]

        assert run() == run()


class TestChromaVectorStore:
    """Chroma 适配器（临时目录，合成向量）。"""

    def test_satisfies_protocol_and_search(self, tmp_path: Path) -> None:
        from agent_workbench.infrastructure.vectorstores import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=tmp_path / "chroma",
            collection_name="aw_test",
        )
        assert isinstance(store, VectorStore)
        embedder = FakeEmbeddings()
        texts = ["Chroma 向量库说明", "无关文本内容"]
        store.add(
            ["c1", "c2"],
            texts,
            embedder.embed_documents(texts),
            [{"source": "s1"}, {"source": "s2"}],
        )
        assert store.count() == 2
        hits = store.similarity_search(embedder.embed_query("Chroma 向量库说明"), k=1)
        assert hits
        assert "Chroma" in hits[0].text

    def test_index_pipeline_with_chroma(self, tmp_path: Path) -> None:
        from agent_workbench.infrastructure.vectorstores import ChromaVectorStore

        doc_path = tmp_path / "kb.txt"
        doc_path.write_text("嵌入式向量数据库 Chroma 入门。" * 30, encoding="utf-8")
        store = ChromaVectorStore(
            persist_dir=tmp_path / "chroma2",
            collection_name="aw_pipe",
        )
        embedder = FakeEmbeddings()
        report = index_paths([doc_path], store=store, embedder=embedder)
        assert report.chunks > 0
        hits = store.similarity_search(embedder.embed_query("Chroma 入门"), k=2)
        assert hits
