from matcher.baselines import keyword_verdict, retrieval_verdict
from matcher.embed import HashingEmbedder
from matcher.evaluate import Item, run_eval, spearman, tune_threshold, write_outputs
from matcher.llm_client import HeuristicLLM
from matcher.models import Requirement, Verdict
from matcher.pipeline import match, prepare_resume
from matcher.report import report_markdown

from conftest import EVAL_DIR


def test_pipeline_end_to_end_offline(settings, sample_resume_text):
    emb = HashingEmbedder(cache_dir=settings.cache_dir)
    llm = HeuristicLLM(settings)
    resume = prepare_resume("cv.md", sample_resume_text, emb)
    jd = (EVAL_DIR / "jds" / "android_lead.txt").read_text()
    report = match(resume, "android_lead.txt", jd, llm, emb, settings)
    texts = [r.requirement.text for r in report.results]
    assert "Remote-first, 30 days holiday" not in texts  # benefits ignored
    assert 0 <= report.score <= 100
    years = next(r for r in report.results if r.requirement.years == 8)
    assert not years.gated and years.verdict.verdict == "met"  # judged from dates
    assert "exp1_h" in [c.chunk_id for c in years.candidates]
    md = report_markdown(report, resume.chunks)
    assert "## Gaps" in md and "Score:" in md


class LyingLLM(HeuristicLLM):
    """Claims every requirement is met with an invented quote."""

    def _complete(self, system, user, model, schema_name, context):
        req = context["requirement"]
        return Verdict(requirement_id=req.id, verdict="met", evidence_ids=[context["chunks"][0].id],
                       evidence_quote="ten years of Rust in production", reason="made up"), 0, 0


def test_validator_blocks_padding_end_to_end(settings, sample_resume_text):
    emb = HashingEmbedder()
    resume = prepare_resume("cv.md", sample_resume_text, emb)
    reqs = [Requirement(id="r1", text="Experience with Rust", must_have=True),
            Requirement(id="r2", text="Kotlin and Jetpack Compose", must_have=True)]
    settings.threshold = 0.0  # nothing gated: every verdict goes through the validator
    report = match(resume, "jd", "", LyingLLM(settings), emb, settings, requirements=reqs)
    assert all(r.verdict.verdict == "missing" for r in report.results)
    assert all(r.downgraded_from == "met" and not r.validated for r in report.results)
    assert report.score == 0


def test_baselines():
    assert keyword_verdict("Kotlin Multiplatform", "kotlin multiplatform and swift") == "met"
    assert keyword_verdict("Kubernetes in production", "android kotlin") == "missing"
    assert retrieval_verdict(0.5, 0.3) == "met"
    assert retrieval_verdict(0.35, 0.3) == "partial"
    assert retrieval_verdict(0.1, 0.3) == "missing"


def test_tune_threshold_keeps_most_matches():
    items = [Item("j", "r", f"r{i}", "t", True, "met", top_similarity=s)
             for i, s in enumerate([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9])]
    items.append(Item("j", "r", "rx", "t", True, "missing", top_similarity=0.05))
    t = tune_threshold(items, keep=0.9)
    assert sum(i.top_similarity >= t for i in items if i.label == "met") >= 9


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2], [1, 2]) is None


def test_eval_runs_offline(settings, tmp_path):
    result = run_eval(EVAL_DIR, settings, HeuristicLLM(settings), HashingEmbedder())
    assert result["n_labeled"] == 37
    for system in ("A_keyword", "C_retrieval", "D_pipeline"):
        assert 0 <= result["metrics"][system]["accuracy"] <= 1
    assert result["extraction"]["recall"] is not None
    assert result["ranking"]["D_pipeline_spearman"] is not None
    write_outputs(result, tmp_path)
    assert (tmp_path / "eval_report.md").exists() and (tmp_path / "predictions.csv").exists()
