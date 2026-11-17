"""Chroma 向量库适配器（实现 VectorStore Protocol）。

chromadb 依赖延迟导入：仅在构造/调用时需要，避免 import 期昂贵初始化。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agent_workbench.infrastructure.vectorstores.base import VectorStoreHit

__all__ = ["ChromaVectorStore"]


class ChromaVectorStore:
    """基于 Chroma 的持久化向量存储。

    实现 `VectorStore` Protocol；持久化目录可配置（默认 `.data/chroma`）。
    """

    def __init__(
        self,
        *,
        persist_dir: str | Path = ".data/chroma",
        collection_name: str = "aw_docs",
    ) -> None:
        """打开或创建 Chroma 集合。

        Args:
            persist_dir: 持久化目录。
            collection_name: 集合名（3~512 字符，[a-zA-Z0-9._-]）。

        Raises:
            ImportError: chromadb 未安装。
        """
        try:
            import chromadb
        except ImportError as exc:
            msg = "未安装 chromadb，请执行: uv add chromadb"
            raise ImportError(msg) from exc

        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Chroma 仅接受标量/标量列表元数据，过滤不兼容类型。"""
        clean: dict[str, Any] = {}
        for key, value in meta.items():
            if (
                isinstance(value, (str, int, float, bool))
                or value is None
                or (
                    isinstance(value, list)
                    and all(isinstance(v, (str, int, float, bool)) for v in value)
                )
            ):
                clean[str(key)] = value
            else:
                clean[str(key)] = str(value)
        return clean

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """写入 Chroma 集合。"""
        if not (len(ids) == len(texts) == len(embeddings)):
            msg = "ids/texts/embeddings 长度必须一致"
            raise ValueError(msg)
        meta = metadatas or [{} for _ in ids]
        clean = [self._sanitize_metadata(m) for m in meta]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=clean,  # type: ignore[arg-type]
        )
        return list(ids)

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
    ) -> list[VectorStoreHit]:
        """Chroma 余弦相似 top-k。"""
        if k < 1:
            return []
        res = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[VectorStoreHit] = []
        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        for row_ids, row_docs, row_metas, row_dists in zip(ids, docs, metas, dists, strict=False):
            for i, doc_id in enumerate(row_ids):
                dist = float(row_dists[i]) if i < len(row_dists) else 1.0
                # cosine distance → 相似度
                score = 1.0 - dist
                hits.append(
                    VectorStoreHit(
                        id=str(doc_id),
                        text=str(row_docs[i]) if i < len(row_docs) else "",
                        score=score,
                        metadata=dict(row_metas[i] or {}) if i < len(row_metas) else {},
                    )
                )
        return hits

    def delete_by_source(self, source: str) -> int:
        """按 metadata.source 删除；返回 -1 表示条数未知（Chroma 不返回）。"""
        self._collection.delete(where={"source": source})
        return -1

    def count(self) -> int:
        """集合内块数量。"""
        return int(self._collection.count())

    @staticmethod
    def new_id() -> str:
        """生成块 id。"""
        return str(uuid.uuid4())

    @property
    def collection(self) -> Any:
        """底层 collection（适配器细节）。"""
        return self._collection
