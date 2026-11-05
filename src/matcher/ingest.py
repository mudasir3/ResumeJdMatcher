"""Turn resume and JD files into plain text."""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix!r}; use one of {sorted(SUPPORTED)}")
    if suffix == ".pdf":
        text = _read_pdf(p)
    elif suffix == ".docx":
        text = _read_docx(p)
    else:
        text = p.read_text(encoding="utf-8")
    return normalize(text)


def _read_pdf(p: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(p: Path) -> str:
    import docx

    d = docx.Document(str(p))
    lines = []
    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "2"
            lines.append("#" * min(int(level), 3) + " " + text)
        elif "list" in style:
            lines.append("- " + text)
        else:
            lines.append(text)
    return "\n".join(lines)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
