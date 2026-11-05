"""Command line: score, batch, diff, eval, tune, chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import Settings
from .embed import make_embedder
from .evaluate import run_eval, tune_threshold, write_outputs
from .extract import extract_requirements
from .ingest import read_text
from .llm_client import make_llm
from .pipeline import match, prepare_resume
from .report import batch_markdown, diff_markdown, report_markdown

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Score a resume against job descriptions, one requirement at a time.")

OfflineOpt = typer.Option(False, "--offline",
                          help="Use rule-based stand-ins instead of an LLM (demo/testing only).")
EmbedOpt = typer.Option(None, "--embed", help="Embedding backend: st or hash (overrides .env).")


def _setup(offline: bool, embed: Optional[str], threshold: Optional[float] = None):
    settings = Settings.from_env()
    if embed:
        settings.embed_backend = embed
    if threshold is not None:
        settings.threshold = threshold
    embedder = make_embedder(settings)
    llm = make_llm(settings, offline=offline)
    return settings, llm, embedder


def _out_dir(settings: Settings, out: Optional[Path]) -> Path:
    d = out or settings.runs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.command()
def score(resume: Path = typer.Option(..., "--resume", "-r", exists=True),
          jd: Path = typer.Option(..., "--jd", "-j", exists=True),
          out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output folder (default runs/)."),
          threshold: Optional[float] = typer.Option(None, "--threshold"),
          offline: bool = OfflineOpt, embed: Optional[str] = EmbedOpt):
    """Score one resume against one JD; writes <jd>.json and <jd>.md."""
    settings, llm, embedder = _setup(offline, embed, threshold)
    prepared = prepare_resume(resume.name, read_text(resume), embedder)
    report = match(prepared, jd.name, read_text(jd), llm, embedder, settings)
    d = _out_dir(settings, out)
    (d / f"{jd.stem}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md = report_markdown(report, prepared.chunks)
    (d / f"{jd.stem}.md").write_text(md, encoding="utf-8")
    flags = f"; missing must-haves: {len(report.missing_must_haves)}" if report.missing_must_haves else ""
    typer.echo(f"{jd.name}: score {report.score:g}/100{flags}")
    typer.echo(f"LLM calls {report.run.llm_calls} (gated {report.run.gated_calls}, "
               f"cached {report.run.cached_calls}); cost ${report.run.cost_usd:.6f}; "
               f"{report.run.latency_s:g} s")
    typer.echo(f"Report: {d / (jd.stem + '.md')}")
    if offline:
        typer.echo("NOTE: --offline uses rule-based stand-ins, not an LLM. Use it to try the "
                   "pipeline, not to judge fit.")


@app.command()
def batch(resume: Path = typer.Option(..., "--resume", "-r", exists=True),
          jds: Path = typer.Option(..., "--jds", exists=True, file_okay=False),
          out: Optional[Path] = typer.Option(None, "--out", "-o"),
          threshold: Optional[float] = typer.Option(None, "--threshold"),
          offline: bool = OfflineOpt, embed: Optional[str] = EmbedOpt):
    """Rank every JD in a folder against one resume; writes ranking.md plus a report per JD."""
    settings, llm, embedder = _setup(offline, embed, threshold)
    prepared = prepare_resume(resume.name, read_text(resume), embedder)
    d = _out_dir(settings, out)
    files = sorted(p for p in jds.iterdir() if p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"})
    if not files:
        raise typer.BadParameter(f"No JD files in {jds}")
    reports = []
    for f in files:
        rep = match(prepared, f.name, read_text(f), llm, embedder, settings)
        (d / f"{f.stem}.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")
        (d / f"{f.stem}.md").write_text(report_markdown(rep, prepared.chunks), encoding="utf-8")
        typer.echo(f"{f.name}: {rep.score:g}")
        reports.append(rep)
    (d / "ranking.md").write_text(batch_markdown(reports), encoding="utf-8")
    typer.echo(f"Ranking: {d / 'ranking.md'}")


@app.command()
def diff(resume: Path = typer.Option(..., "--resume", "-r", exists=True),
         resume_b: Path = typer.Option(..., "--resume-b", exists=True),
         jd: Path = typer.Option(..., "--jd", "-j", exists=True),
         out: Optional[Path] = typer.Option(None, "--out", "-o"),
         offline: bool = OfflineOpt, embed: Optional[str] = EmbedOpt):
    """Compare two resume versions against the same JD."""
    settings, llm, embedder = _setup(offline, embed)
    jd_text = read_text(jd)
    reqs = extract_requirements(jd_text, llm).requirements  # same requirements for both
    reports = []
    for path in (resume, resume_b):
        prepared = prepare_resume(path.name, read_text(path), embedder)
        reports.append(match(prepared, jd.name, jd_text, llm, embedder, settings, requirements=reqs))
    d = _out_dir(settings, out)
    md = diff_markdown(*reports)
    (d / f"diff_{jd.stem}.md").write_text(md, encoding="utf-8")
    typer.echo(md)


@app.command("eval")
def eval_cmd(eval_dir: Path = typer.Option(Path("eval"), "--dir", exists=True, file_okay=False),
             tune_jds: Optional[int] = typer.Option(None, "--tune-jds",
                                                    help="How many JDs (alphabetical) to tune the gate on."),
             threshold: Optional[float] = typer.Option(None, "--threshold",
                                                       help="Use a fixed gate instead of tuning."),
             skip_extraction: bool = typer.Option(False, "--skip-extraction"),
             offline: bool = OfflineOpt, embed: Optional[str] = EmbedOpt):
    """Run the golden-set eval; writes eval_report.md, eval_result.json, predictions.csv."""
    settings, llm, embedder = _setup(offline, embed, threshold)
    result = run_eval(eval_dir, settings, llm, embedder, tune_jds=tune_jds,
                      skip_extraction=skip_extraction)
    write_outputs(result, eval_dir)
    d = result["metrics"].get("D_pipeline", {})
    typer.echo(f"Pipeline: accuracy {d.get('accuracy')}, false-match rate {d.get('false_match_rate')}, "
               f"gap recall {d.get('gap_recall')} (n={d.get('n')})")
    if offline:
        typer.echo("NOTE: --offline uses rule-based stand-ins; these are not the tool's real numbers.")
    typer.echo(f"Report: {eval_dir / 'eval_report.md'}")


@app.command()
def tune(eval_dir: Path = typer.Option(Path("eval"), "--dir", exists=True, file_okay=False),
         keep: float = typer.Option(0.95, help="Share of labeled met/partial items that must pass the gate."),
         embed: Optional[str] = EmbedOpt):
    """Suggest a gate THRESHOLD from labeled data (no LLM calls)."""
    from .evaluate import Item, load_labels
    from .retrieve import similarity_matrix

    settings = Settings.from_env()
    if embed:
        settings.embed_backend = embed
    embedder = make_embedder(settings)
    items, cache = [], {}
    for l in load_labels(eval_dir):
        if l.resume not in cache:
            cache[l.resume] = prepare_resume(l.resume, read_text(eval_dir / "resumes" / l.resume), embedder)
        sim = float(similarity_matrix(embedder.embed([l.req.text]), cache[l.resume].vectors).max())
        items.append(Item(l.jd, l.resume, l.req.id, l.req.text, l.req.must_have, l.label, sim))
    t = tune_threshold(items, keep)
    missing = [i for i in items if i.label == "missing"]
    skipped = sum(i.top_similarity < t for i in missing)
    typer.echo(f"Suggested THRESHOLD={t} ({getattr(embedder, 'name', '')}). At this value "
               f"{skipped}/{len(missing)} labeled-missing requirements skip the LLM.")


@app.command()
def chunks(resume: Path = typer.Option(..., "--resume", "-r", exists=True)):
    """Print how the resume is chunked (check this on your real resume first)."""
    from .chunker import chunk_resume
    from .experience import total_years

    cs = chunk_resume(read_text(resume))
    for c in cs:
        typer.echo(f"{c.id:<12} {c.kind:<11} {c.text[:100]}")
    typer.echo(f"\n{len(cs)} chunks; dated experience: {total_years(cs)} years")


if __name__ == "__main__":
    app()
