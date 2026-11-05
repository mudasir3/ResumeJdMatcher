"""Data contracts. The LLM must return exactly these shapes; the validator enforces them."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["experience", "skill", "domain", "education", "other"]
VerdictLabel = Literal["met", "partial", "missing"]
GapType = Literal["reword", "real_gap"]


# ---------- LLM outputs ----------

class Requirement(BaseModel):
    id: str
    text: str
    must_have: bool
    category: Category = "skill"
    years: Optional[float] = None


class RequirementList(BaseModel):
    requirements: list[Requirement]


class Verdict(BaseModel):
    requirement_id: str
    verdict: VerdictLabel
    evidence_ids: list[str] = Field(default_factory=list)
    # A short span copied word for word from the cited chunk. Checked in code.
    evidence_quote: Optional[str] = None
    reason: str = ""
    gap_type: Optional[GapType] = None


# ---------- Pipeline data ----------

class Chunk(BaseModel):
    id: str
    text: str
    section: str = ""
    role: str = ""
    kind: Literal["bullet", "skills", "role_header", "paragraph"] = "bullet"


class Candidate(BaseModel):
    chunk_id: str
    similarity: float


class RequirementResult(BaseModel):
    requirement: Requirement
    verdict: Verdict
    candidates: list[Candidate]
    top_similarity: float
    gated: bool = False          # skipped the LLM because similarity < threshold
    validated: bool = True       # passed the verbatim-evidence check
    downgraded_from: Optional[VerdictLabel] = None
    notes: list[str] = Field(default_factory=list)


class RunStats(BaseModel):
    model: str = ""
    embed_backend: str = ""
    threshold: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0
    cached_calls: int = 0
    gated_calls: int = 0


class Report(BaseModel):
    resume: str
    jd: str
    score: float
    missing_must_haves: list[str]
    experience_years: Optional[float] = None
    results: list[RequirementResult]
    run: RunStats
