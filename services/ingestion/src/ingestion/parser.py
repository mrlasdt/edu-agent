from __future__ import annotations

from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def is_allowed(filename: str, content_type: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS or content_type in ALLOWED_CONTENT_TYPES


def parse_bytes(content: bytes, filename: str) -> str:
    """Extract plain text from file bytes based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return content.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _parse_pdf(content)
    if ext == ".docx":
        return _parse_docx(content)
    raise ValueError(f"Unsupported file type: {ext!r}")


def _parse_pdf(content: bytes) -> str:
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def _parse_docx(content: bytes) -> str:
    import io
    from docx import Document
    doc = Document(io.BytesIO(content))
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
