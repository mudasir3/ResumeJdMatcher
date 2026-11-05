"""Readable Markdown reports (the JSON report is Report.model_dump_json())."""

from __future__ import annotations

from .models import Chunk, Report

ICON = {"met": "met", "partial": "partial", "missing": "MISSING"}


def _cell(text: str, limit: int = 140) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def report_markdown(report: Report, chunks: list[Chunk]) -> str:
    by_id = {c.id: c for c in chunks}
    out = [f"# Match report: {report.resume} vs {report.jd}", ""]
    out.append(f"**Score: {report.score:g} / 100**")
    if report.missing_must_haves:
        names = [r.requirement.text for r in report.results
                 if r.requirement.id in report.missing_must_haves]
        out.append("")
        out.append(f"**Missing must-haves ({len(names)}):** " + "; ".join(_cell(n, 80) for n in names))
    if report.experience_years is not None:
        out.append("")
        out.append(f"Dated experience on resume: {report.experience_years:g} years")
    out += ["", "## Requirements", "",
            "| ID | Requirement | Must | Verdict | Evidence | Top sim |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in report.results:
        ev = ", ".join(r.verdict.evidence_ids) or ("gated" if r.gated else "-")
        verdict = ICON[r.verdict.verdict]
        if r.downgraded_from:
            verdict += f" (was {r.downgraded_from})"
        out.append(f"| {r.requirement.id} | {_cell(r.requirement.text, 90)} | "
                   f"{'yes' if r.requirement.must_have else 'no'} | {verdict} | {ev} | "
                   f"{r.top_similarity:.2f} |")

    reword = [r for r in report.results if r.verdict.gap_type == "reword"]
    real = [r for r in report.results if r.verdict.gap_type == "real_gap"]
    out += ["", "## Gaps", ""]
    out += ["### Reword: evidence exists but is weak or indirect", ""]
    if not reword:
        out.append("None.")
    for r in reword:
        cited = "; ".join(f"`{i}`: {_cell(by_id[i].text, 120)}" for i in r.verdict.evidence_ids
                          if i in by_id)
        out.append(f"- **{_cell(r.requirement.text, 90)}**: {_cell(r.verdict.reason)}"
                   + (f" Evidence: {cited}" if cited else ""))
    out += ["", "### Real gaps: no evidence; learn it or address it honestly", ""]
    if not real:
        out.append("None.")
    for r in real:
        tag = " (must-have)" if r.requirement.must_have else ""
        out.append(f"- **{_cell(r.requirement.text, 90)}**{tag}: {_cell(r.verdict.reason)}")

    notes = [(r.requirement.id, n) for r in report.results for n in r.notes]
    if notes:
        out += ["", "## Validator notes", ""]
        out += [f"- {rid}: {n}" for rid, n in notes]

    run = report.run
    out += ["", "## Run", "",
            f"- Model: {run.model}; embeddings: {run.embed_backend}; gate threshold: {run.threshold:g}",
            f"- LLM calls: {run.llm_calls} (cached {run.cached_calls}, gated out {run.gated_calls})",
            f"- Tokens: {run.tokens_in} in / {run.tokens_out} out; cost ${run.cost_usd:.6f}; "
            f"latency {run.latency_s:g} s", ""]
    return "\n".join(out)


def batch_markdown(reports: list[Report]) -> str:
    ranked = sorted(reports, key=lambda r: (-r.score, len(r.missing_must_haves)))
    out = ["# JD ranking", "", "| Rank | JD | Score | Missing must-haves | Top gaps |",
           "| --- | --- | --- | --- | --- |"]
    for i, rep in enumerate(ranked, start=1):
        gaps = [r for r in rep.results if r.verdict.verdict != "met"]
        gaps.sort(key=lambda r: (not r.requirement.must_have, r.verdict.verdict != "missing"))
        top = "; ".join(_cell(r.requirement.text, 50) for r in gaps[:3]) or "-"
        out.append(f"| {i} | {rep.jd} | {rep.score:g} | {len(rep.missing_must_haves)} | {top} |")
    cost = sum(r.run.cost_usd for r in reports)
    calls = sum(r.run.llm_calls for r in reports)
    out += ["", f"{len(reports)} JDs, {calls} LLM calls, total cost ${cost:.6f}.", ""]
    return "\n".join(out)


def diff_markdown(a: Report, b: Report) -> str:
    out = [f"# Resume diff on {a.jd}", "",
           f"- {a.resume}: **{a.score:g}**", f"- {b.resume}: **{b.score:g}**", "",
           "| Requirement | " + _cell(a.resume, 30) + " | " + _cell(b.resume, 30) + " |",
           "| --- | --- | --- |"]
    b_by_text = {r.requirement.text.lower(): r for r in b.results}
    changed = 0
    for ra in a.results:
        rb = b_by_text.get(ra.requirement.text.lower())
        if rb is None:
            continue
        if ra.verdict.verdict != rb.verdict.verdict:
            changed += 1
            out.append(f"| {_cell(ra.requirement.text, 80)} | {ra.verdict.verdict} | "
                       f"{rb.verdict.verdict} |")
    if not changed:
        out.append("| (no verdict changes) | | |")
    out.append("")
    return "\n".join(out)
