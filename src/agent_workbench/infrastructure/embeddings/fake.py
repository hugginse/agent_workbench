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
        """单条文本 → 归一化伪向量（字符 1/2-gram 哈希累加）。

        同输入必得同向量；共享 n-gram 的相似文本余弦更高，
        便于离线 RAG 检索测试（比整串哈希更接近「语义」行为）。
        """
        values = [0.0] * self._dim
        normalized = text.strip().lower()
        if not normalized:
            normalized = text.lower() or " "

        grams: list[str] = list(normalized)
        grams.extend(normalized[i : i + 2] for i in range(max(0, len(normalized) - 1)))
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            # 用摘要前 8 字节选维度，后 4 字节给符号/幅度
            slot = int.from_bytes(digest[:4], "big") % self._dim
            sign_raw = int.from_bytes(digest[4:8], "big")
            sign = 1.0 if sign_raw % 2 == 0 else -1.0
            weight = 1.0 + (sign_raw % 1000) / 1000.0
            values[slot] += sign * weight

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
