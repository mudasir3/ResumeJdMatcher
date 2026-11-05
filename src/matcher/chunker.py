"""Split a resume into evidence units with stable, readable IDs.

One chunk = one bullet, one skills line, one role header or one short paragraph.
IDs look like exp2_b3 (role 2, bullet 3), exp2_h (role 2 header), skills_1, sum_1.
"""

from __future__ import annotations

import re

from .models import Chunk

BULLET_RE = re.compile(r"^\s*(?:[-*•▪●‣–]|\d+[.)])\s+")
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE = rf"(?:{MONTHS}\s+)?(?:19|20)\d{{2}}"
END = rf"(?:{DATE}|present|current|now|today)"
DATE_RANGE_RE = re.compile(rf"({DATE})\s*(?:-|–|—|to)\s*({END})", re.IGNORECASE)

KNOWN_SECTIONS = {
    "summary": "sum", "profile": "sum", "about": "sum", "objective": "sum",
    "experience": "exp", "work experience": "exp", "professional experience": "exp",
    "employment": "exp", "work history": "exp", "career": "exp",
    "skills": "skills", "technical skills": "skills", "core skills": "skills",
    "technologies": "skills", "tech stack": "skills", "tools": "skills",
    "education": "edu", "certifications": "cert", "certificates": "cert",
    "projects": "proj", "personal projects": "proj", "open source": "oss",
    "achievements": "ach", "awards": "ach", "publications": "pub",
    "languages": "lang", "speaking": "talk", "talks": "talk",
}
MAX_PARAGRAPH = 350


def _section_prefix(name: str) -> str:
    key = re.sub(r"[^a-z ]", "", name.lower()).strip()
    if key in KNOWN_SECTIONS:
        return KNOWN_SECTIONS[key]
    for known, prefix in KNOWN_SECTIONS.items():
        if known in key:
            return prefix
    slug = re.sub(r"[^a-z]", "", key)
    return slug[:5] or "sec"


def _plain_section_heading(line: str) -> str | None:
    """Detect a section heading in non-Markdown text (PDF/DOCX exports)."""
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > 40:
        return None
    key = re.sub(r"[^a-z ]", "", stripped.lower()).strip()
    if key in KNOWN_SECTIONS:
        return stripped
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    if len(letters) >= 4 and letters.isupper() and len(stripped.split()) <= 4:
        return stripped.title()
    return None


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_PARAGRAPH:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > MAX_PARAGRAPH:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_resume(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    section, prefix = "Header", "head"
    role, role_idx = "", 0
    counters: dict[str, int] = {}
    paragraph: list[str] = []

    def next_id(base: str) -> str:
        counters[base] = counters.get(base, 0) + 1
        return f"{base}_{counters[base]}"

    def is_skills() -> bool:
        return prefix == "skills"

    def is_experience() -> bool:
        return prefix in {"exp", "proj"}

    def add(text_: str, kind: str) -> None:
        text_ = text_.strip()
        if not text_:
            return
        if is_skills():
            kind = "skills"
        if is_experience() and role_idx > 0:
            base = f"{prefix}{role_idx}"
            if kind == "role_header":
                cid = f"{base}_h"
            else:
                counters[base] = counters.get(base, 0) + 1
                cid = f"{base}_b{counters[base]}"
        else:
            cid = next_id(prefix)
        chunks.append(Chunk(id=cid, text=text_, section=section, role=role, kind=kind))  # type: ignore[arg-type]

    def flush_paragraph() -> None:
        if paragraph:
            for part in _split_long(" ".join(paragraph)):
                add(part, "paragraph")
            paragraph.clear()

    def start_role(header: str) -> None:
        nonlocal role, role_idx
        flush_paragraph()
        role_idx += 1
        role = header
        add(header, "role_header")

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_paragraph()
            continue

        md = re.match(r"^(#{1,6})\s+(.*)$", line)
        if md:
            flush_paragraph()
            level, title = len(md.group(1)), md.group(2).strip()
            if level == 1:
                # The candidate's name: not evidence for anything.
                section, prefix, role, role_idx = "Header", "head", "", 0
            elif level == 2 or not is_experience():
                section, prefix, role, role_idx = title, _section_prefix(title), "", 0
            else:
                start_role(title)
            continue

        plain_heading = _plain_section_heading(line)
        if plain_heading and not BULLET_RE.match(line):
            flush_paragraph()
            section, prefix, role, role_idx = plain_heading, _section_prefix(plain_heading), "", 0
            continue

        bullet = BULLET_RE.match(line)
        if bullet:
            flush_paragraph()
            add(line[bullet.end():], "bullet")
            continue

        # A non-bullet line with a date range inside an experience section starts a role,
        # unless it is the date line directly under a role heading that has no dates yet.
        if is_experience() and DATE_RANGE_RE.search(line) and len(line) < 160:
            last = chunks[-1] if chunks else None
            if (
                last is not None
                and last.kind == "role_header"
                and last.role == role
                and not DATE_RANGE_RE.search(last.text)
                and not paragraph
            ):
                last.text = f"{last.text} | {line.strip()}"
                last.role = last.text
                role = last.text
            else:
                start_role(line.strip())
            continue

        if is_skills():
            add(line, "skills")
        else:
            paragraph.append(line.strip())

    flush_paragraph()
    return chunks
