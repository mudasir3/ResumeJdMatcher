"""Cheaper approaches the full pipeline must beat to justify its LLM cost.

A. keyword overlap      : share of a requirement's terms found anywhere in the resume
B. whole-doc similarity : one cosine for the whole resume vs the whole JD (ranking only)
C. retrieval + threshold: per-requirement best chunk similarity, no LLM
"""

from __future__ import annotations

import numpy as np

from .embed import Embedder
from .llm_client import content_tokens


def keyword_verdict(requirement_text: str, resume_text: str) -> str:
    req = set(content_tokens(requirement_text))
    if not req:
        return "missing"
    resume = set(content_tokens(resume_text))
    coverage = len(req & resume) / len(req)
    if coverage >= 0.75:
        return "met"
    if coverage >= 0.4:
        return "partial"
    return "missing"


def retrieval_verdict(top_similarity: float, threshold: float, margin: float = 0.15) -> str:
    if top_similarity >= threshold + margin:
        return "met"
    if top_similarity >= threshold:
        return "partial"
    return "missing"


def whole_doc_similarity(resume_text: str, jd_text: str, embedder: Embedder) -> float:
    """Mean of line embeddings on each side, then cosine. (Embedding each document as one
    string would silently truncate it at the model's input limit.)"""
    def doc_vec(text: str) -> np.ndarray:
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 2]
        v = embedder.embed(lines).mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    return round(float(doc_vec(resume_text) @ doc_vec(jd_text)), 4)
