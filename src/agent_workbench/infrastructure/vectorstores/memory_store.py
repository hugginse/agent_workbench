"""内存向量库：测试与离线演示用（同一 Protocol）。"""

from __future__ import annotations

import math
import uuid
from typing import Any

from agent_workbench.infrastructure.vectorstores.base import VectorStoreHit

__all__ = ["InMemoryVectorStore"]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0


class InMemoryVectorStore:
    """进程内向量存储（可注入；测试默认）。"""

    def __init__(self) -> None:
        """初始化空库。"""
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._metadatas: list[dict[str, Any]] = []

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """追加写入。"""
        if not (len(ids) == len(texts) == len(embeddings)):
            msg = "ids/texts/embeddings 长度必须一致"
            raise ValueError(msg)
        meta = metadatas or [{} for _ in ids]
        if len(meta) != len(ids):
            msg = "metadatas 长度必须与 ids 一致"
            raise ValueError(msg)
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(dict(m) for m in meta)
        return list(ids)

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
    ) -> list[VectorStoreHit]:
        """余弦相似度 top-k。"""
        if k < 1 or not self._ids:
            return []
        scored: list[tuple[float, int]] = []
        qn = _norm(query_embedding)
        for i, emb in enumerate(self._embeddings):
            sim = _dot(query_embedding, emb) / (qn * _norm(emb))
            scored.append((sim, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        hits: list[VectorStoreHit] = []
        for sim, i in scored[:k]:
            hits.append(
                VectorStoreHit(
                    id=self._ids[i],
                    text=self._texts[i],
                    score=float(sim),
                    metadata=dict(self._metadatas[i]),
                )
            )
        return hits

    def delete_by_source(self, source: str) -> int:
        """按 source 删除，返回条数。"""
        keep = [i for i, m in enumerate(self._metadatas) if m.get("source") != source]
        removed = len(self._ids) - len(keep)
        self._ids = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._embeddings = [self._embeddings[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        return removed

    def count(self) -> int:
        """块数量。"""
        return len(self._ids)

    @staticmethod
    def new_id() -> str:
        """生成块 id。"""
        return str(uuid.uuid4())
