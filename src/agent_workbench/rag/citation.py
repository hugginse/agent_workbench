"""引用构造：把检索命中转为可溯源的引用条目。

铁律：引用 id 只能来自检索命中，模型不得生成（本层保证 id 来源）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_workbench.infrastructure.vectorstores.base import VectorStoreHit

__all__ = ["Citation", "CitationSet", "build_citations", "format_citations_block"]


class Citation(BaseModel):
    """一条可溯源引用。

    Attributes:
        ref: 引用标记（如 [1]），与正文标注对应。
        chunk_id: 检索命中的块 id（来自存储，非模型生成）。
        source: 来源文件名或路径。
        snippet: 摘录正文。
        score: 相似度。
    """

    ref: str = Field(description="引用标记")
    chunk_id: str = Field(min_length=1, description="块 id（必须来自检索命中）")
    source: str = Field(default="unknown", description="来源")
    snippet: str = Field(default="", description="摘录")
    score: float = Field(default=0.0, description="相似度")


class CitationSet(BaseModel):
    """一次检索产生的引用集合。

    Attributes:
        citations: 有序引用列表（ref 为 [1]..[n]）。
        notes: 给模型/用户的提示（如无命中说明）。
    """

    citations: list[Citation] = Field(default_factory=list)
    notes: str = Field(default="", description="附加说明")

    @property
    def ids(self) -> list[str]:
        """全部 chunk_id（供校验模型引用是否越界）。"""
        return [c.chunk_id for c in self.citations]

    def by_ref(self, ref: str) -> Citation | None:
        """按引用标记查找。"""
        for c in self.citations:
            if c.ref == ref:
                return c
        return None


def build_citations(
    hits: list[VectorStoreHit],
    *,
    snippet_chars: int = 200,
) -> CitationSet:
    """将检索命中转为带序号的引用集合。

    Args:
        hits: VectorStore 相似检索结果（有序）。
        snippet_chars: 摘录最大字符数。

    Returns:
        CitationSet；无命中时 citations 为空并带 notes。
    """
    if not hits:
        return CitationSet(
            citations=[],
            notes="无检索命中，回答时不得编造引用；无证据应拒答。",
        )

    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        source = str(
            hit.metadata.get("filename")
            or hit.metadata.get("source")
            or hit.metadata.get("path")
            or "unknown"
        )
        snippet = hit.text.strip()
        if snippet_chars > 0 and len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + "…"
        citations.append(
            Citation(
                ref=f"[{index}]",
                chunk_id=hit.id,
                source=source,
                snippet=snippet,
                score=hit.score,
            )
        )
    return CitationSet(citations=citations, notes="引用 id 仅来自本次检索命中。")


def format_citations_block(citation_set: CitationSet) -> str:
    """渲染引用列表为可嵌入 prompt 的文本块。

    Args:
        citation_set: 引用集合。

    Returns:
        Markdown 风格引用列表；无命中时返回拒答提示。
    """
    if not citation_set.citations:
        return citation_set.notes or "无检索命中，无可引用材料。"
    lines: list[str] = []
    for c in citation_set.citations:
        lines.append(f"{c.ref} (source={c.source}, id={c.chunk_id}): {c.snippet}")
    header = citation_set.notes or "以下引用仅来自检索命中："
    return header + "\n" + "\n".join(lines)


def extract_ref_ids_from_text(text: str, *, allowed: list[str]) -> list[str]:
    """抽取文本中出现的合法 chunk_id（仅 allowed 子集）。

    用于 #26 强制校验的前置工具；数字引用 `[n]` 需经 CitationSet 映射后再调本函数。

    Args:
        text: 模型输出或待检文本。
        allowed: 允许的 chunk_id 列表（来自检索命中）。

    Returns:
        出现过的合法 chunk_id（按 allowed 顺序去重）。
    """
    allowed_set = set(allowed)
    found: list[str] = []
    for chunk_id in allowed:
        if chunk_id in allowed_set and chunk_id in text and chunk_id not in found:
            found.append(chunk_id)
    return found
