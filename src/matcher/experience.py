"""Years of experience computed from dated role headers, not from adjectives."""

from __future__ import annotations

import re
from datetime import date

from .chunker import DATE_RANGE_RE
from .models import Chunk

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _parse(token: str, *, is_end: bool, today: date) -> tuple[int, int] | None:
    token = token.strip().lower().rstrip(".")
    if token in {"present", "current", "now", "today"}:
        return today.year, today.month
    year_match = re.search(r"(19|20)\d{2}", token)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month_match = re.match(r"([a-z]{3})", token)
    if month_match and month_match.group(1) in _MONTHS:
        return year, _MONTHS[month_match.group(1)]
    # Year only: assume the whole year for an end date, January for a start date.
    return year, (12 if is_end else 1)


def role_intervals(chunks: list[Chunk], today: date | None = None) -> list[tuple[int, int]]:
    """Return (start_month_index, end_month_index) per dated role header."""
    today = today or date.today()
    intervals = []
    for c in chunks:
        if c.kind != "role_header":
            continue
        m = DATE_RANGE_RE.search(c.text)
        if not m:
            continue
        start = _parse(m.group(1), is_end=False, today=today)
        end = _parse(m.group(2), is_end=True, today=today)
        if not start or not end:
            continue
        s = start[0] * 12 + start[1] - 1
        e = end[0] * 12 + end[1]  # inclusive of the end month
        if e > s:
            intervals.append((s, e))
    return intervals


def total_years(chunks: list[Chunk], today: date | None = None) -> float | None:
    """Total years across dated roles, with overlapping roles merged (not double counted)."""
    intervals = sorted(role_intervals(chunks, today))
    if not intervals:
        return None
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    months = sum(e - s for s, e in merged)
    return round(months / 12, 1)
