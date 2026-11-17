"""文本切分：按字符窗口 + 重叠，优先在段落/句边界断开。

中文友好（按字符计数）；确定性、无网络。
"""

from __future__ import annotations

from agent_workbench.rag.loaders import Document

__all__ = ["split_document", "split_text"]

# 默认块大小与重叠（字符数）
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# 优先在此类字符后断行/断句
_BOUNDARIES = "。！？；\n.!?;"


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """将长文本切成带重叠的块。

    Args:
        text: 原文。
        chunk_size: 单块最大字符数，必须 >= 2。
        chunk_overlap: 相邻块重叠字符数，须满足 0 <= overlap < chunk_size。

    Returns:
        非空块列表；空文本返回 []。

    Raises:
        ValueError: 参数不合法。
    """
    if chunk_size < 2:
        msg = f"chunk_size 必须 >= 2，当前 {chunk_size}"
        raise ValueError(msg)
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        msg = f"chunk_overlap 必须在 [0, {chunk_size})，当前 {chunk_overlap}"
        raise ValueError(msg)

    compact = text.replace("\r\n", "\n").strip()
    if not compact:
        return []
    if len(compact) <= chunk_size:
        return [compact]

    chunks: list[str] = []
    start = 0
    length = len(compact)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # 在窗口末尾向前找最近的边界字符，避免拦腰截断句子
            window = compact[start:end]
            cut_rel = -1
            for ch in reversed(window):
                if ch in _BOUNDARIES:
                    cut_rel = window.rfind(ch)
                    break
            # 仅当边界不在太靠前时采用，否则硬切
            if cut_rel > chunk_size // 4:
                end = start + cut_rel + 1
        piece = compact[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def split_document(
    document: Document,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """切分文档，保留来源元数据并附加 chunk 序号。

    Args:
        document: 源文档。
        chunk_size: 块大小。
        chunk_overlap: 重叠字符数。

    Returns:
        子 Document 列表；metadata 含 source/chunk_index/chunk_total。
    """
    pieces = split_text(
        document.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    total = len(pieces)
    out: list[Document] = []
    for index, piece in enumerate(pieces):
        meta = dict(document.metadata)
        meta.update(
            {
                "chunk_index": index,
                "chunk_total": total,
                "source": meta.get("filename") or meta.get("path") or "unknown",
            }
        )
        out.append(Document(content=piece, metadata=meta))
    return out
