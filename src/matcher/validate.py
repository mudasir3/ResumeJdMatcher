"""Code-side checks on every verdict. The prompt asks for honesty; this enforces it.

Rules, in order:
1. Cited IDs must be among the chunks the judge was shown. Unknown IDs are dropped.
2. met/partial needs at least one valid citation and a quote found verbatim in a cited chunk.
   Otherwise the verdict is downgraded to missing.
3. met backed only by skills-list lines is capped at partial (listed is not demonstrated).
4. An experience requirement with a stated number of years is capped at partial when the
   years computed from role dates fall short.
5. gap_type is made consistent with the final verdict.
"""

from __future__ import annotations

import re
import unicodedata

from .models import Chunk, Requirement, Verdict

_ORDER = {"missing": 0, "partial": 1, "met": 2}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = (text.replace("‘", "'").replace("’", "'").replace("“", '"')
            .replace("”", '"').replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_chunks(quote: str | None, chunks: list[Chunk]) -> bool:
    if not quote:
        return False
    q = _norm(quote).strip(" .,;:\"'")
    if len(q) < 3:
        return False
    return any(q in _norm(c.text) for c in chunks)


def validate_verdict(verdict: Verdict, requirement: Requirement, shown: list[Chunk],
                     experience_years: float | None) -> tuple[Verdict, bool, str | None, list[str]]:
    """Return (final verdict, passed_evidence_check, downgraded_from, notes)."""
    notes: list[str] = []
    original = verdict.verdict
    shown_by_id = {c.id: c for c in shown}

    valid_ids = [i for i in verdict.evidence_ids if i in shown_by_id]
    dropped = [i for i in verdict.evidence_ids if i not in shown_by_id]
    if dropped:
        notes.append(f"dropped unknown evidence IDs: {', '.join(dropped)}")
    v = verdict.model_copy(update={"evidence_ids": valid_ids})
    passed = True

    if v.verdict in ("met", "partial"):
        cited = [shown_by_id[i] for i in valid_ids]
        if not cited:
            passed = False
            notes.append("no valid citation")
        elif not quote_in_chunks(v.evidence_quote, cited):
            passed = False
            notes.append("quote not found verbatim in cited chunks")
        if not passed:
            v = v.model_copy(update={"verdict": "missing", "evidence_ids": []})

    if v.verdict == "met":
        cited = [shown_by_id[i] for i in v.evidence_ids]
        if cited and all(c.kind == "skills" for c in cited):
            v = v.model_copy(update={"verdict": "partial"})
            notes.append("only listed in a skills line; capped at partial")

    if (v.verdict == "met" and requirement.category == "experience" and requirement.years
            and experience_years is not None and experience_years < requirement.years):
        v = v.model_copy(update={"verdict": "partial"})
        notes.append(f"dated roles total {experience_years:g} years < {requirement.years:g} required")

    gap = {"met": None, "partial": v.gap_type or "reword", "missing": "real_gap"}[v.verdict]
    if v.verdict == "partial" and gap not in ("reword", "real_gap"):
        gap = "reword"
    v = v.model_copy(update={"gap_type": gap})

    downgraded = original if _ORDER[v.verdict] < _ORDER[original] else None
    return v, passed, downgraded, notes
