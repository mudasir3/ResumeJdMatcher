"""score = 100 * sum(w_i * v_i) / sum(w_i); must-have w=2, nice-to-have w=1;
met=1, partial=0.5, missing=0. Missing must-haves are flagged separately."""

from __future__ import annotations

from .models import RequirementResult

VALUE = {"met": 1.0, "partial": 0.5, "missing": 0.0}


def weighted_score(results: list[RequirementResult], must_weight: float = 2.0,
                   nice_weight: float = 1.0) -> float:
    total = earned = 0.0
    for r in results:
        w = must_weight if r.requirement.must_have else nice_weight
        total += w
        earned += w * VALUE[r.verdict.verdict]
    return round(100 * earned / total, 1) if total else 0.0


def missing_must_haves(results: list[RequirementResult]) -> list[str]:
    return [r.requirement.id for r in results
            if r.requirement.must_have and r.verdict.verdict == "missing"]
