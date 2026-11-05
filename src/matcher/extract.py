"""JD text -> list of atomic requirements (LLM, structured output)."""

from __future__ import annotations

from importlib import resources

from .llm_client import BaseLLM
from .models import RequirementList


def _prompt(name: str) -> str:
    return resources.files("matcher").joinpath("prompts", name).read_text(encoding="utf-8")


def extract_requirements(jd_text: str, llm: BaseLLM) -> RequirementList:
    system = _prompt("extract_requirements.md")
    user = f"Job description:\n<<<\n{jd_text}\n>>>"
    result = llm.complete_json(system, user, RequirementList, "requirement_list",
                               context={"jd_text": jd_text})
    assert isinstance(result, RequirementList)
    # Normalise IDs so downstream references are stable even if the model skips numbers.
    seen = set()
    cleaned = []
    for i, r in enumerate(result.requirements, start=1):
        text = r.text.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(r.model_copy(update={"id": f"r{len(cleaned) + 1}", "text": text}))
    return RequirementList(requirements=cleaned)
