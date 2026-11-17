"""文档加载：txt / pdf / docx → 统一 Document。

仅读取本地文件；测试必须使用合成 fixture（契约）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["Document", "UnsupportedFormatError", "load_document", "load_documents"]

# 后缀 → 加载器名
_SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


class UnsupportedFormatError(ValueError):
    """不支持的文件格式。"""


class Document(BaseModel):
    """统一文档对象。

    Attributes:
        content: 全文文本。
        metadata: 来源信息（path / filename / pages 等）。
    """

    content: str = Field(min_length=0, description="文档全文")
    metadata: dict[str, Any] = Field(default_factory=dict, description="来源元数据")


def load_document(path: str | Path) -> Document:
    """按后缀加载单个文档。

    Args:
        path: 文件路径（.txt/.md/.pdf/.docx）。

    Returns:
        Document 实例。

    Raises:
        FileNotFoundError: 文件不存在。
        UnsupportedFormatError: 后缀不支持。
    """
    p = Path(path)
    if not p.exists():
        msg = f"文件不存在: {p.name}"
        raise FileNotFoundError(msg)
    suffix = p.suffix.lower()
    if suffix not in _SUPPORTED:
        msg = f"不支持的格式: {suffix or '(无后缀)'}（支持 {sorted(_SUPPORTED)}）"
        raise UnsupportedFormatError(msg)

    if suffix in {".txt", ".md"}:
        content = p.read_text(encoding="utf-8", errors="replace")
        return Document(
            content=content,
            metadata={"path": str(p), "filename": p.name, "type": suffix.lstrip(".")},
        )
    if suffix == ".pdf":
        return _load_pdf(p)
    return _load_docx(p)


def load_documents(paths: list[str | Path]) -> list[Document]:
    """批量加载；单个失败不吞掉错误（调用方可先过滤）。"""
    return [load_document(p) for p in paths]


def _load_pdf(path: Path) -> Document:
    """用 pypdf 抽取 PDF 文本（合成 fixture 可离线生成）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return Document(
        content="\n\n".join(parts),
        metadata={
            "path": str(path),
            "filename": path.name,
            "type": "pdf",
            "pages": len(reader.pages),
        },
    )


def _load_docx(path: Path) -> Document:
    """用 python-docx 抽取段落文本。"""
    import docx

    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text is not None]
    content = "\n".join(paragraphs)
    return Document(
        content=content,
        metadata={"path": str(path), "filename": path.name, "type": "docx"},
    )
