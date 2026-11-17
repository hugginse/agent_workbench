"""索引管道：加载 → 切分 → 向量化 → 写入 VectorStore。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol as TypingProtocol

from pydantic import BaseModel, Field

from agent_workbench.rag.loaders import Document, load_document
from agent_workbench.rag.splitters import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_document,
)

__all__ = ["Embedder", "IndexReport", "index_document", "index_paths"]


class Embedder(TypingProtocol):
    """最小 Embedding 端口（Fake / OpenAI 等均可注入）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。"""
        ...


class IndexReport(BaseModel):
    """一次索引的结果报告。"""

    files: int = Field(default=0, description="处理文件数")
    chunks: int = Field(default=0, description="写入块数")
    sources: list[str] = Field(default_factory=list, description="来源文件名列表")


def index_document(
    document: Document,
    *,
    store: Any,
    embedder: Embedder,
    id_factory: Any = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IndexReport:
    """将单个文档切分、向量化并写入向量库。

    Args:
        document: 已加载文档。
        store: 实现 VectorStore Protocol 的存储。
        embedder: Embedding 提供方。
        id_factory: 生成块 id 的可调用；默认 uuid4 字符串。
        chunk_size: 切分块大小。
        chunk_overlap: 重叠字符数。

    Returns:
        IndexReport。
    """
    from uuid import uuid4

    factory = id_factory or (lambda: str(uuid4()))
    chunks = split_document(
        document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return IndexReport(files=1, chunks=0, sources=[])

    texts = [c.content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    ids = [str(factory()) for _ in chunks]
    metadatas = [dict(c.metadata) for c in chunks]
    store.add(ids, texts, embeddings, metadatas)
    source = str(chunks[0].metadata.get("filename") or chunks[0].metadata.get("source") or "")
    return IndexReport(files=1, chunks=len(chunks), sources=[source] if source else [])


def index_paths(
    paths: list[str | Path],
    *,
    store: Any,
    embedder: Embedder,
    id_factory: Any = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IndexReport:
    """批量索引文件路径（txt/md/pdf/docx）。

    Args:
        paths: 文件路径列表。
        store: VectorStore 实现。
        embedder: Embedding 提供方。
        id_factory: 块 id 工厂。
        chunk_size: 块大小。
        chunk_overlap: 重叠。

    Returns:
        汇总 IndexReport。

    Raises:
        FileNotFoundError: 任一文件不存在（fail-fast）。
    """
    total_files = 0
    total_chunks = 0
    sources: list[str] = []
    for path in paths:
        doc = load_document(path)
        report = index_document(
            doc,
            store=store,
            embedder=embedder,
            id_factory=id_factory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        total_files += report.files
        total_chunks += report.chunks
        sources.extend(report.sources)
    return IndexReport(files=total_files, chunks=total_chunks, sources=sources)
