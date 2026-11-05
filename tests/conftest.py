from pathlib import Path

import pytest

from matcher.config import Settings

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"


@pytest.fixture
def settings(tmp_path):
    return Settings(embed_backend="hash", cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs",
                    llm_base_url="http://example.invalid/v1", llm_api_key="test", llm_model="test-model",
                    price_in_per_m=1.0, price_out_per_m=2.0)


@pytest.fixture
def sample_resume_text():
    return (EVAL_DIR / "resumes" / "sample_resume.md").read_text(encoding="utf-8")
