"""Embeddings. Default: a small local sentence-transformers model (free, offline after the
first download, resume never leaves the machine). Fallback: a hashing embedder with no
dependencies, used for tests and when sentence-transformers is not installed."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

from .cache import content_key
from .config import Settings


class Embedder:
    name = "base"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) / "emb" / re.sub(r"[^a-zA-Z0-9_.-]", "_", self.name) \
            if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors, one row per text, using the on-disk cache."""
        out: list[np.ndarray | None] = [None] * len(texts)
        todo_idx, todo_text = [], []
        for i, t in enumerate(texts):
            path = self._path(t)
            if path is not None and path.exists():
                out[i] = np.load(path)
            else:
                todo_idx.append(i)
                todo_text.append(t)
        if todo_text:
            vecs = self._encode(todo_text)
            for i, t, v in zip(todo_idx, todo_text, vecs):
                v = _normalize(np.asarray(v, dtype=np.float32))
                out[i] = v
                path = self._path(t)
                if path is not None:
                    np.save(path, v)
        if not out:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(out)  # type: ignore[arg-type]

    def _path(self, text: str) -> Path | None:
        return self.cache_dir / f"{content_key(self.name, text)}.npy" if self.cache_dir else None


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, cache_dir: Path | None = None):
        self.name = model_name
        super().__init__(cache_dir)
        self.model_name = model_name
        self._model = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class HashingEmbedder(Embedder):
    """Bag of words + character n-grams hashed into a fixed vector. Lexical, not semantic:
    it will miss synonyms ("declarative UI" vs "Jetpack Compose"). Useful as a baseline."""

    name = "hash-2048"

    def __init__(self, dim: int = 2048, cache_dir: Path | None = None):
        self.dim = dim
        super().__init__(cache_dir)

    def _encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9+#]+", text.lower())
            feats = list(words) + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for w in words:
                padded = f" {w} "
                feats += [padded[i:i + 4] for i in range(max(1, len(padded) - 3))]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                mat[row, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            mat[row] = np.sign(mat[row]) * np.log1p(np.abs(mat[row]))
        return mat


def make_embedder(settings: Settings, use_cache: bool = True) -> Embedder:
    cache = settings.cache_dir if use_cache else None
    if settings.embed_backend == "st":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            print("sentence-transformers is not installed; falling back to the hashing embedder. "
                  "Install with: pip install -e \".[st]\"", file=sys.stderr)
            settings.embed_backend = "hash"
            return HashingEmbedder(cache_dir=cache)
        return SentenceTransformerEmbedder(settings.embed_model, cache_dir=cache)
    if settings.embed_backend == "hash":
        return HashingEmbedder(cache_dir=cache)
    raise ValueError(f"Unknown EMBED_BACKEND {settings.embed_backend!r} (use 'st' or 'hash')")
