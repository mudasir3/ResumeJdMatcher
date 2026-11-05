"""Per-requirement retrieval: cosine similarity over normalised vectors, top-k chunks.

NumPy is enough here: one resume is ~40-80 chunks, so a full matrix product is instant.
A vector DB only becomes useful when searching across many stored documents."""

from __future__ import annotations

import numpy as np

from .models import Candidate, Chunk


def similarity_matrix(req_vecs: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """Rows = requirements, columns = chunks. Vectors are already L2-normalised."""
    if req_vecs.size == 0 or chunk_vecs.size == 0:
        return np.zeros((len(req_vecs), len(chunk_vecs)), dtype=np.float32)
    return req_vecs @ chunk_vecs.T


def top_k(sims_row: np.ndarray, chunks: list[Chunk], k: int) -> list[Candidate]:
    order = np.argsort(-sims_row)[:k]
    return [Candidate(chunk_id=chunks[i].id, similarity=round(float(sims_row[i]), 4)) for i in order]


def passes_gate(candidates: list[Candidate], threshold: float) -> bool:
    return bool(candidates) and candidates[0].similarity >= threshold
