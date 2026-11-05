"""Settings, read from environment variables (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Default gate thresholds per embedding backend. These are starting points only:
# run `matcher tune` on your labeled set and set THRESHOLD to the tuned value.
DEFAULT_THRESHOLDS = {"st": 0.30, "hash": 0.12}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


@dataclass
class Settings:
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    llm_max_rpm: int = 0
    llm_max_retries: int = 5

    embed_backend: str = "st"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k: int = 3
    threshold: float | None = None

    must_have_weight: float = 2.0
    nice_to_have_weight: float = 1.0

    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    runs_dir: Path = field(default_factory=lambda: Path("runs"))

    def gate_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        return DEFAULT_THRESHOLDS.get(self.embed_backend, 0.3)

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        if env_file and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        threshold_raw = os.getenv("THRESHOLD", "").strip()
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            price_in_per_m=_float("PRICE_IN_PER_M", 0.0),
            price_out_per_m=_float("PRICE_OUT_PER_M", 0.0),
            llm_max_rpm=_int("LLM_MAX_RPM", 0),
            llm_max_retries=_int("LLM_MAX_RETRIES", 5),
            embed_backend=os.getenv("EMBED_BACKEND", "st").strip() or "st",
            embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
            top_k=_int("TOP_K", 3),
            threshold=float(threshold_raw) if threshold_raw else None,
            must_have_weight=_float("MUST_HAVE_WEIGHT", 2.0),
            nice_to_have_weight=_float("NICE_TO_HAVE_WEIGHT", 1.0),
            cache_dir=Path(os.getenv("CACHE_DIR", ".cache")),
            runs_dir=Path(os.getenv("RUNS_DIR", "runs")),
        )
