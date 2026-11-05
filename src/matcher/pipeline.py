"""End-to-end: chunk -> extract -> embed -> retrieve -> gate -> judge -> validate -> score."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .chunker import chunk_resume
from .config import Settings
from .embed import Embedder
from .experience import total_years
from .extract import extract_requirements
from .judge import judge_requirement
from .llm_client import BaseLLM
from .models import (Candidate, Chunk, Report, Requirement, RequirementResult, RunStats, Verdict)
from .retrieve import passes_gate, similarity_matrix, top_k
from .score import missing_must_haves, weighted_score
from .validate import validate_verdict


@dataclass
class PreparedResume:
    name: str
    text: str
    chunks: list[Chunk]
    vectors: np.ndarray
    experience_years: float | None


def prepare_resume(name: str, text: str, embedder: Embedder) -> PreparedResume:
    chunks = chunk_resume(text)
    if not chunks:
        raise ValueError(f"No content found in resume {name!r}")
    vectors = embedder.embed([c.text for c in chunks])
    return PreparedResume(name, text, chunks, vectors, total_years(chunks))


def match(resume: PreparedResume, jd_name: str, jd_text: str, llm: BaseLLM, embedder: Embedder,
          settings: Settings, requirements: list[Requirement] | None = None) -> Report:
    """Score one resume against one JD. Pass `requirements` to skip extraction (used by eval)."""
    start = time.perf_counter()
    before = llm.totals.snapshot()
    threshold = settings.gate_threshold()

    if requirements is None:
        requirements = extract_requirements(jd_text, llm).requirements

    results: list[RequirementResult] = []
    if requirements:
        req_vecs = embedder.embed([r.text for r in requirements])
        sims = similarity_matrix(req_vecs, resume.vectors)
        by_id = {c.id: c for c in resume.chunks}
        for i, req in enumerate(requirements):
            cands = top_k(sims[i], resume.chunks, settings.top_k)
            top_sim = cands[0].similarity if cands else 0.0
            # Years requirements are judged from role dates, which similarity can't see:
            # they bypass the gate and the judge is always shown the most recent role header.
            years_req = req.years is not None and resume.experience_years is not None
            if years_req:
                header = next((c for c in resume.chunks if c.kind == "role_header"), None)
                if header and all(c.chunk_id != header.id for c in cands):
                    cands = cands + [Candidate(chunk_id=header.id, similarity=0.0)]
            if not years_req and not passes_gate(cands, threshold):
                verdict = Verdict(requirement_id=req.id, verdict="missing", evidence_ids=[],
                                  reason=f"Best resume chunk similarity {top_sim:.3f} is below "
                                         f"the gate ({threshold:.3f}).", gap_type="real_gap")
                results.append(RequirementResult(requirement=req, verdict=verdict, candidates=cands,
                                                 top_similarity=top_sim, gated=True))
                continue
            shown = [by_id[c.chunk_id] for c in cands]
            raw = judge_requirement(req, shown, llm, resume.experience_years)
            final, passed, downgraded, notes = validate_verdict(raw, req, shown,
                                                                resume.experience_years)
            results.append(RequirementResult(requirement=req, verdict=final, candidates=cands,
                                             top_similarity=top_sim, validated=passed,
                                             downgraded_from=downgraded, notes=notes))

    used = llm.totals.since(before)
    run = RunStats(
        model=llm.model_name, embed_backend=getattr(embedder, "name", ""), threshold=threshold,
        tokens_in=used.tokens_in, tokens_out=used.tokens_out, cost_usd=round(used.cost_usd, 6),
        latency_s=round(time.perf_counter() - start, 2), llm_calls=used.calls,
        cached_calls=used.cached_calls, gated_calls=sum(r.gated for r in results),
    )
    return Report(
        resume=resume.name, jd=jd_name,
        score=weighted_score(results, settings.must_have_weight, settings.nice_to_have_weight),
        missing_must_haves=missing_must_haves(results),
        experience_years=resume.experience_years, results=results, run=run,
    )
