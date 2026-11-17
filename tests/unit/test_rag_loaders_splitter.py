"""RAG 加载与切分单测（全部使用合成 fixture）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workbench.rag import (
    Document,
    load_document,
    split_document,
    split_text,
)
from agent_workbench.rag.loaders import UnsupportedFormatError


def _make_txt(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_docx(path: Path, paragraphs: list[str]) -> Path:
    import docx

    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))
    return path


def _make_pdf(path: Path, lines: list[str]) -> Path:
    """合成单页 PDF（reportlab 不在依赖内 → 用 pypdf 可读的最小手写 PDF）。"""
    # 构造最小合法 PDF（单页，Helvetica 文本）
    content_stream = "BT /F1 12 Tf 72 720 Td\n"
    for line in lines:
        # PDF 字符串转义
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_stream += f"({safe}) Tj T*\n"
    content_stream += "ET"
    stream_bytes = content_stream.encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(out))
    return path


class TestLoadTxt:
    """txt / md 加载。"""

    def test_load_txt(self, tmp_path: Path) -> None:
        p = _make_txt(tmp_path / "a.txt", "合成正文内容ABC")
        doc = load_document(p)
        assert doc.content == "合成正文内容ABC"
        assert doc.metadata["filename"] == "a.txt"
        assert doc.metadata["type"] == "txt"

    def test_load_md_as_text(self, tmp_path: Path) -> None:
        p = _make_txt(tmp_path / "a.md", "# 标题\n正文")
        doc = load_document(p)
        assert "# 标题" in doc.content

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nope.txt")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        p = _make_txt(tmp_path / "a.xyz", "x")
        with pytest.raises(UnsupportedFormatError, match="不支持"):
            load_document(p)


class TestLoadDocx:
    """docx 加载（合成文档）。"""

    def test_load_docx_paragraphs(self, tmp_path: Path) -> None:
        p = _make_docx(tmp_path / "syn.docx", ["第一段合成内容", "第二段"])
        doc = load_document(p)
        assert "第一段合成内容" in doc.content
        assert "第二段" in doc.content
        assert doc.metadata["type"] == "docx"


class TestLoadPdf:
    """pdf 加载（合成 PDF）。"""

    def test_load_pdf_text(self, tmp_path: Path) -> None:
        p = _make_pdf(tmp_path / "syn.pdf", ["Hello Synthetic PDF", "Line Two"])
        doc = load_document(p)
        assert "Hello Synthetic PDF" in doc.content
        assert "Line Two" in doc.content
        assert doc.metadata["type"] == "pdf"
        assert doc.metadata["pages"] == 1


class TestSplitText:
    """切分核心。"""

    def test_empty(self) -> None:
        assert split_text("") == []
        assert split_text("   \n  ") == []

    def test_short_text_single_chunk(self) -> None:
        assert split_text("短文本", chunk_size=100) == ["短文本"]

    def test_long_text_multiple_chunks(self) -> None:
        text = "内容" * 500  # 1000 字
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2
        assert all(len(c) <= 100 for c in chunks)

    def test_overlap_between_chunks(self) -> None:
        text = "abcdefghij" * 50
        chunks = split_text(text, chunk_size=40, chunk_overlap=8)
        # 相邻块应有重叠内容
        if len(chunks) >= 2:
            assert chunks[0][-8:] in chunks[1] or chunks[1][:8] in chunks[0]

    def test_prefers_sentence_boundary(self) -> None:
        text = ("这是第一句话。" * 20) + ("这是第二句话。" * 20)
        chunks = split_text(text, chunk_size=50, chunk_overlap=0)
        assert chunks
        # 尽量不在句中硬切（允许个别边界情况，多数块应以句读结尾或完整句）
        assert any(c.endswith(("。", "！", "？")) or len(c) < 50 for c in chunks)

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            split_text("x", chunk_size=1)
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_text("x", chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_text("x", chunk_size=10, chunk_overlap=-1)

    def test_deterministic(self) -> None:
        text = "确定性切分测试。" * 100
        a = split_text(text, chunk_size=80, chunk_overlap=8)
        b = split_text(text, chunk_size=80, chunk_overlap=8)
        assert a == b


class TestSplitDocument:
    """文档级切分与元数据。"""

    def test_chunk_metadata(self) -> None:
        doc = Document(
            content="段落内容。" * 200,
            metadata={"filename": "合成.txt", "type": "txt"},
        )
        parts = split_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(parts) >= 2
        assert parts[0].metadata["chunk_index"] == 0
        assert parts[0].metadata["chunk_total"] == len(parts)
        assert parts[-1].metadata["chunk_index"] == len(parts) - 1
        assert parts[0].metadata["source"] == "合成.txt"

    def test_short_doc_one_chunk(self) -> None:
        doc = Document(content="短", metadata={"filename": "s.txt"})
        parts = split_document(doc)
        assert len(parts) == 1
        assert parts[0].content == "短"

    def test_load_then_split_roundtrip(self, tmp_path: Path) -> None:
        p = _make_txt(
            tmp_path / "long.txt",
            "合成段落一。" * 100,
        )
        doc = load_document(p)
        parts = split_document(doc, chunk_size=120, chunk_overlap=12)
        assert parts
        assert all(p.metadata["source"] == "long.txt" for p in parts)
