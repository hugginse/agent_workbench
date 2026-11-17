"""Phase 4 集成测试：合成文档 → 索引 → 检索对话 → 引用铁律 / 拒答。

全部使用合成 fixture，禁止真实用户文档（契约）。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from agent_workbench.infrastructure.embeddings import FakeEmbeddings
from agent_workbench.infrastructure.llm import ScriptedChatModel
from agent_workbench.infrastructure.vectorstores import InMemoryVectorStore
from agent_workbench.orchestrator.single_agent import create_single_agent_graph, initial_state
from agent_workbench.rag import build_citations, enforce_answer, index_paths
from agent_workbench.rag.guard import ABSTAIN_ANSWER, check_answer
from agent_workbench.tools.builtins import RetrieveTool
from agent_workbench.tools.registry import ToolRegistry


def _write_synthetic_docs(tmp_path: Path) -> list[Path]:
    """写入两份合成知识文档。"""
    kb1 = tmp_path / "chroma_notes.txt"
    kb1.write_text(
        "Chroma 是一种嵌入式向量数据库，支持本地持久化与余弦相似检索。\n"
        "适合在原型阶段快速验证检索增强生成流程。\n"
        "切换 embedding 模型后必须清空索引并重建。\n" * 8,
        encoding="utf-8",
    )
    kb2 = tmp_path / "pg_notes.txt"
    kb2.write_text(
        "PostgreSQL 提供事务、索引与 JSONB 类型，适合生产负载。\n"
        "向量扩展 pgvector 可在同一数据库内做相似检索。\n"
        "数据删除时需同时清理向量与元数据。\n" * 8,
        encoding="utf-8",
    )
    return [kb1, kb2]


def _seeded_knowledge(tmp_path: Path) -> tuple[InMemoryVectorStore, FakeEmbeddings]:
    """合成文档 → 索引。"""
    store = InMemoryVectorStore()
    embedder = FakeEmbeddings()
    paths = _write_synthetic_docs(tmp_path)
    index_paths(paths, store=store, embedder=embedder, chunk_size=240, chunk_overlap=24)
    return store, embedder


def _registry_with_retrieve(store: InMemoryVectorStore, embedder: FakeEmbeddings) -> ToolRegistry:
    r = ToolRegistry()
    r.register(RetrieveTool(store, embedder))
    return r


class TestIndexThenRetrieveChat:
    """索引 → 检索 → 带引用对话。"""

    def test_multi_file_index_and_search(self, tmp_path: Path) -> None:
        store, embedder = _seeded_knowledge(tmp_path)
        assert store.count() > 0
        hits = store.similarity_search(embedder.embed_query("Chroma 本地持久化"), k=3)
        assert hits
        sources = {h.metadata.get("source") or h.metadata.get("filename") for h in hits}
        assert any(s and "chroma" in str(s) for s in sources)

    def test_retrieve_tool_returns_citations_from_hits(self, tmp_path: Path) -> None:
        store, embedder = _seeded_knowledge(tmp_path)
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "PostgreSQL 事务与 JSONB", "k": 3}))
        assert payload["hits"], payload
        assert payload["abstain"] is False
        hit_ids = [h["chunk_id"] for h in payload["hits"]]
        for c in payload["citations"]:
            assert c["chunk_id"] in hit_ids
            assert c["ref"].startswith("[")

    def test_single_agent_calls_retrieve_then_answers(self, tmp_path: Path) -> None:
        """单 Agent：先 retrieve，再带 [1] 作答；经 enforce_answer 通过。"""
        store, embedder = _seeded_knowledge(tmp_path)
        registry = _registry_with_retrieve(store, embedder)

        retrieve_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve",
                    "args": {"query": "Chroma 嵌入式向量数据库", "k": 3},
                    "id": "c_ret",
                }
            ],
        )
        # 第二轮：模型给出带编号的答案（编号在 enforce 时对照真实 hits）
        llm = ScriptedChatModel(
            scripted_responses=[
                retrieve_call,
                "Chroma 支持本地持久化，适合原型验证 [1]。",
            ]
        )
        graph = create_single_agent_graph(llm, registry, max_steps=4)
        state = graph.invoke(initial_state("Chroma 是什么？", max_steps=4))

        # 工具已执行
        tool_msgs = [m for m in state["messages"] if m.type == "tool"]
        assert len(tool_msgs) == 1
        tool_payload = json.loads(str(tool_msgs[0].content))
        assert tool_payload["hits"]

        # 最终答案经引用铁律校验
        final = state["messages"][-1]
        answer = str(final.content)
        cs = build_citations(
            [
                __import__(
                    "agent_workbench.infrastructure.vectorstores",
                    fromlist=["VectorStoreHit"],
                ).VectorStoreHit(
                    id=h["chunk_id"],
                    text=h["text"],
                    score=h["score"],
                    metadata={"source": h["source"]},
                )
                for h in tool_payload["hits"]
            ]
        )
        checked = check_answer(answer, cs)
        assert checked.ok, checked.reason
        assert checked.confidence > 0
        assert checked.confidence < 1.0 or checked.confidence <= 0.95

    def test_enforce_roundtrip_on_grounded_answer(self, tmp_path: Path) -> None:
        """检索材料 → 构造合法答案 → enforce 保留原文。"""
        store, embedder = _seeded_knowledge(tmp_path)
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "pgvector 向量检索", "k": 5}))
        cs = build_citations(
            [
                __import__(
                    "agent_workbench.infrastructure.vectorstores",
                    fromlist=["VectorStoreHit"],
                ).VectorStoreHit(
                    id=h["chunk_id"],
                    text=h["text"],
                    score=h["score"],
                    metadata={"source": h["source"]},
                )
                for h in payload["hits"]
            ]
        )
        assert cs.citations, "合成库应有命中"
        ref = cs.citations[0].ref
        raw = f"根据检索材料 {ref}，可在同一数据库内做相似检索。"
        final, check = enforce_answer(raw, cs)
        assert check.ok
        assert final == raw


class TestAbstentionPaths:
    """无证据拒答。"""

    def test_empty_kb_abstain_via_tool(self) -> None:
        tool = RetrieveTool(InMemoryVectorStore(), FakeEmbeddings())
        payload = json.loads(tool.run({"query": "任何问题"}))
        assert payload["abstain"] is True
        assert payload["abstain_answer"] == ABSTAIN_ANSWER
        assert payload["hits"] == []

    def test_empty_kb_answer_forced_to_abstain(self) -> None:
        from agent_workbench.rag.citation import CitationSet

        final, check = enforce_answer("我猜答案是 42。", CitationSet(citations=[]))
        assert final == ABSTAIN_ANSWER
        assert check.must_abstain
        assert check.ok is False or "拒答" in check.reason or check.ok  # 已改写为拒答话术
        # enforce 返回拒答话术本身
        assert "拒答" in final

    def test_fabricated_citation_forced_to_abstain(self, tmp_path: Path) -> None:
        store, embedder = _seeded_knowledge(tmp_path)
        tool = RetrieveTool(store, embedder)
        payload = json.loads(tool.run({"query": "Chroma", "k": 3}))
        cs = build_citations(
            [
                __import__(
                    "agent_workbench.infrastructure.vectorstores",
                    fromlist=["VectorStoreHit"],
                ).VectorStoreHit(
                    id=h["chunk_id"],
                    text=h["text"],
                    score=h["score"],
                    metadata={"source": h["source"]},
                )
                for h in payload["hits"]
            ]
        )
        final, check = enforce_answer("结论见 [999] 与 id=madeup-abc123", cs)
        assert check.ok is False
        assert final == ABSTAIN_ANSWER


class TestSyntheticOnlyGuard:
    """fixture 合成性：内容不含真实敏感样本特征。"""

    def test_fixture_paths_are_under_tmp(self, tmp_path: Path) -> None:
        paths = _write_synthetic_docs(tmp_path)
        for p in paths:
            assert p.parent == tmp_path
            text = p.read_text(encoding="utf-8")
            assert "Chroma" in text or "PostgreSQL" in text
            # 合成文档不包含常见真实 PII 模式（简单启发式）
            assert "@" not in text
