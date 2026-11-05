"""One requirement + its retrieved chunks -> verdict (LLM, structured output)."""

from __future__ import annotations

from .extract import _prompt
from .llm_client import BaseLLM
from .models import Chunk, Requirement, Verdict


def build_judge_input(requirement: Requirement, chunks: list[Chunk],
                      experience_years: float | None) -> str:
    lines = [
        f"Requirement {requirement.id}: {requirement.text}",
        f"Must-have: {'yes' if requirement.must_have else 'no'}",
    ]
    if requirement.years is not None:
        lines.append(f"Years required: {requirement.years:g}")
        if experience_years is not None:
            lines.append(f"Candidate's total dated experience (computed from role dates): "
                         f"{experience_years:g} years")
    lines.append("")
    lines.append("Resume chunks:")
    for c in chunks:
        where = f" [{c.section}{' / ' + c.role if c.role else ''}]" if c.section else ""
        kind = " (skills list)" if c.kind == "skills" else ""
        lines.append(f"- {c.id}{where}{kind}: {c.text}")
    return "\n".join(lines)


def judge_requirement(requirement: Requirement, chunks: list[Chunk], llm: BaseLLM,
                      experience_years: float | None = None) -> Verdict:
    system = _prompt("judge_requirement.md")
    user = build_judge_input(requirement, chunks, experience_years)
    verdict = llm.complete_json(system, user, Verdict, "verdict",
                                context={"requirement": requirement, "chunks": chunks,
                                         "experience_years": experience_years})
    assert isinstance(verdict, Verdict)
    return verdict.model_copy(update={"requirement_id": requirement.id})
