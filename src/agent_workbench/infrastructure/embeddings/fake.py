"""确定性替身 Embeddings：同输入同向量（哈希伪向量）。"""

from __future__ import annotations

import hashlib
import math
from typing import Any

__all__ = ["DEFAULT_DIM", "FakeEmbeddings"]

DEFAULT_DIM = 64


class FakeEmbeddings:
    """离线确定性 Embeddings 替身。

    - 同一文本 → 同一向量；
    - 不同文本 → 不同向量（哈希分布）；
    - 符合 langchain Embeddings 的 embed_documents / embed_query 接口。
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        """初始化维度。

        Args:
            dim: 向量维度，必须 >= 4。
        """
        if dim < 4:
            msg = f"dim 必须 >= 4，当前 {dim}"
            raise ValueError(msg)
        self._dim = dim

    @property
    def dim(self) -> int:
        """向量维度。"""
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        """单条文本 → 归一化伪向量。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        counter = 0
        while len(values) < self._dim:
            block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(block), 4):
                if len(values) >= self._dim:
                    break
                raw = int.from_bytes(block[i : i + 4], "big")
                values.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        # L2 归一化（避免全零）
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文档向量化。"""
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """查询向量化（与文档同一空间）。"""
        return self._embed_one(text)

    # 兼容 langchain 风格别名
    def embed_documents_texts(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        """别名，兼容部分调用方。"""
        return self.embed_documents(texts)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 不应命中
        """未知属性显式失败，避免静默吞掉拼写错误。"""
        msg = f"FakeEmbeddings 没有属性 {name!r}"
        raise AttributeError(msg)
