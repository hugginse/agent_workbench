"""VectorStore 端口：便于替换 Chroma / Milvus / Pinecone（设计预留）。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = ["VectorStore", "VectorStoreHit"]


class VectorStoreHit(BaseModel):
    """一次相似检索命中。"""

    id: str = Field(description="块 id")
    text: str = Field(description="原文")
    score: float = Field(default=0.0, description="相似度分数（越大越相关）")
    metadata: dict[str, Any] = Field(default_factory=dict, description="来源元数据")


@runtime_checkable
class VectorStore(Protocol):
    """向量存储端口（结构化 Protocol，可注入替换实现）。"""

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """写入向量与原文，返回入库 id 列表。"""
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
    ) -> list[VectorStoreHit]:
        """按查询向量返回 top-k 命中。"""
        ...

    def delete_by_source(self, source: str) -> int:
        """删除 metadata.source 匹配的全部块，返回删除条数。"""
        ...

    def count(self) -> int:
        """当前存储的块数量。"""
        ...
