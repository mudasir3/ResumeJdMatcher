"""Golden-set evaluation: full pipeline vs baselines, against hand labels.

eval/
  jds/<name>.txt            job descriptions
  resumes/<name>.md         resumes
  labels.csv                jd,resume,req_id,requirement_text,must_have,category,years,label,evidence_id
  fit_ranking.csv (optional) jd,my_fit   (your own 1-5 fit judgment, for ranking agreement)

Verdict metrics feed the LABELED requirements to the pipeline (extraction skipped), so
judge errors and extraction errors are measured separately.
"""

from __future__ import annotations

import csv
import difflib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .baselines import keyword_verdict, retrieval_verdict, whole_doc_similarity
from .config import Settings
from .embed import Embedder
from .extract import extract_requirements
from .ingest import read_text
from .llm_client import BaseLLM, content_tokens
from .models import Requirement
from .pipeline import PreparedResume, match, prepare_resume
from .retrieve import similarity_matrix

LABELS = ("met", "partial", "missing")
_ORDER = {"missing": 0, "partial": 1, "met": 2}


@dataclass
class LabeledReq:
    jd: str
    resume: str
    req: Requirement
    label: str
    evidence_id: str = ""


@dataclass
class Item:
    jd: str
    resume: str
    req_id: str
    text: str
    must_have: bool
    label: str
    top_similarity: float = 0.0
    preds: dict[str, str] = field(default_factory=dict)
    gated: bool = False
    validated: bool = True
    reason: str = ""
    evidence: str = ""


def load_labels(eval_dir: Path) -> list[LabeledReq]:
    path = eval_dir / "labels.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().lower()
            if label not in LABELS:
                raise ValueError(f"labels.csv: bad label {label!r} in row {row}")
            years = (row.get("years") or "").strip()
            req = Requirement(
                id=row["req_id"].strip(), text=row["requirement_text"].strip(),
                must_have=row["must_have"].strip().lower() in {"1", "true", "yes", "y"},
                category=(row.get("category") or "skill").strip() or "skill",
                years=float(years) if years else None,
            )
            rows.append(LabeledReq(row["jd"].strip(), row["resume"].strip(), req, label,
                                   (row.get("evidence_id") or "").strip()))
    return rows


def tune_threshold(items: list[Item], keep: float = 0.95) -> float:
    """Highest threshold that still lets `keep` of the labeled met/partial items reach the
    judge. The gate should only skip requirements that truly have no evidence."""
    sims = sorted(i.top_similarity for i in items if i.label != "missing")
    if not sims:
        return 0.0
    idx = int((1 - keep) * len(sims))
    return round(max(0.0, sims[min(idx, len(sims) - 1)] - 0.005), 3)


def verdict_metrics(items: list[Item], system: str) -> dict:
    preds = [(i.label, i.preds[system]) for i in items if system in i.preds]
    if not preds:
        return {}
    n = len(preds)
    missing = [p for lab, p in preds if lab == "missing"]
    return {
        "n": n,
        "accuracy": round(sum(lab == p for lab, p in preds) / n, 3),
        "false_match_rate": round(sum(p == "met" for p in missing) / len(missing), 3) if missing else None,
        "gap_recall": round(sum(p == "missing" for p in missing) / len(missing), 3) if missing else None,
        "overclaim_rate": round(sum(_ORDER[p] > _ORDER[lab] for lab, p in preds) / n, 3),
        "confusion": {lab: dict(Counter(p for l2, p in preds if l2 == lab)) for lab in LABELS},
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(cov / (va * vb), 3) if va and vb else None


def _extraction_match(labeled: str, extracted: list[str]) -> bool:
    lt = set(content_tokens(labeled))
    for e in extracted:
        if difflib.SequenceMatcher(None, labeled.lower(), e.lower()).ratio() >= 0.6:
            return True
        et = set(content_tokens(e))
        if lt and et and len(lt & et) / len(lt | et) >= 0.5:
            return True
    return False


def run_eval(eval_dir: Path, settings: Settings, llm: BaseLLM, embedder: Embedder,
             tune_jds: int | None = None, skip_extraction: bool = False) -> dict:
    eval_dir = Path(eval_dir)
    labels = load_labels(eval_dir)
    jd_names = sorted({l.jd for l in labels})
    jd_text = {j: read_text(eval_dir / "jds" / j) for j in jd_names}
    resumes: dict[str, PreparedResume] = {}
    for name in sorted({l.resume for l in labels}):
        resumes[name] = prepare_resume(name, read_text(eval_dir / "resumes" / name), embedder)

    groups: dict[tuple[str, str], list[LabeledReq]] = defaultdict(list)
    for l in labels:
        groups[(l.resume, l.jd)].append(l)

    # 1. Similarities for every labeled requirement (no LLM) -> baselines A and C, tuning.
    items: list[Item] = []
    for (res, jd), group in groups.items():
        prepared = resumes[res]
        sims = similarity_matrix(embedder.embed([g.req.text for g in group]), prepared.vectors)
        for g, row in zip(group, sims):
            items.append(Item(jd=jd, resume=res, req_id=g.req.id, text=g.req.text,
                              must_have=g.req.must_have, label=g.label,
                              top_similarity=round(float(row.max()), 4),
                              preds={"A_keyword": keyword_verdict(g.req.text, prepared.text)}))

    # 2. Tune the gate on the first `tune_jds` JDs, report on the rest.
    if tune_jds is None:
        tune_jds = max(1, round(len(jd_names) * 0.3)) if len(jd_names) >= 4 else 0
    tune_set = set(jd_names[:tune_jds])
    tune_items = [i for i in items if i.jd in tune_set] or items
    threshold = settings.threshold if settings.threshold is not None else tune_threshold(tune_items)
    split_note = (f"threshold tuned on {sorted(tune_set)}; metrics on the other JDs"
                  if tune_set else "too few JDs to split: tuned and reported on the same data")
    for i in items:
        i.preds["C_retrieval"] = retrieval_verdict(i.top_similarity, threshold)

    # 3. Full pipeline on labeled requirements.
    run_settings = Settings(**{**settings.__dict__, "threshold": threshold})
    by_key = {(i.resume, i.jd, i.req_id): i for i in items}
    cost = 0.0
    calls = cached = 0
    for (res, jd), group in groups.items():
        report = match(resumes[res], jd, jd_text[jd], llm, embedder, run_settings,
                       requirements=[g.req for g in group])
        cost += report.run.cost_usd
        calls += report.run.llm_calls
        cached += report.run.cached_calls
        for r in report.results:
            it = by_key[(res, jd, r.requirement.id)]
            it.preds["D_pipeline"] = r.verdict.verdict
            it.gated, it.validated = r.gated, r.validated
            it.reason = r.verdict.reason
            it.evidence = ",".join(r.verdict.evidence_ids)

    test_items = [i for i in items if i.jd not in tune_set] or items
    systems = ["A_keyword", "C_retrieval", "D_pipeline"]
    metrics = {s: verdict_metrics(test_items, s) for s in systems}

    # 4. Extraction recall (one LLM call per JD).
    extraction = None
    if not skip_extraction:
        found = total = 0
        per_jd = {}
        for jd in jd_names:
            extracted = [r.text for r in extract_requirements(jd_text[jd], llm).requirements]
            labeled_texts = sorted({l.req.text for l in labels if l.jd == jd})
            hits = sum(_extraction_match(t, extracted) for t in labeled_texts)
            per_jd[jd] = {"labeled": len(labeled_texts), "found": hits, "extracted": len(extracted)}
            found += hits
            total += len(labeled_texts)
        extraction = {"recall": round(found / total, 3) if total else None, "per_jd": per_jd}

    # 5. Ranking agreement with your own fit judgment (optional file).
    ranking = None
    fit_path = eval_dir / "fit_ranking.csv"
    if fit_path.exists():
        with fit_path.open(newline="", encoding="utf-8") as f:
            fit = {r["jd"].strip(): float(r["my_fit"]) for r in csv.DictReader(f)}
        main_resume = sorted(resumes)[0]
        jds = [j for j in jd_names if j in fit]
        b_scores = [whole_doc_similarity(resumes[main_resume].text, jd_text[j], embedder) for j in jds]
        d_scores = []
        for j in jds:
            group = groups.get((main_resume, j))
            rep = match(resumes[main_resume], j, jd_text[j], llm, embedder, run_settings,
                        requirements=[g.req for g in group] if group else None)
            d_scores.append(rep.score)
        mine = [fit[j] for j in jds]
        ranking = {"resume": main_resume, "jds": jds,
                   "B_whole_doc_spearman": spearman(b_scores, mine),
                   "D_pipeline_spearman": spearman(d_scores, mine)}

    gated = sum(i.gated for i in test_items)
    return {
        "llm_model": llm.model_name, "embed_backend": getattr(embedder, "name", ""),
        "threshold": threshold, "split": split_note,
        "n_labeled": len(items), "n_reported": len(test_items),
        "metrics": metrics, "extraction": extraction, "ranking": ranking,
        "pipeline": {"llm_calls": calls, "cached_calls": cached, "gated": gated, "cost_usd": round(cost, 6),
                     "cost_per_jd": round(cost / max(1, len(groups)), 6)},
        "items": [i.__dict__ for i in items],
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def eval_markdown(result: dict) -> str:
    m = result["metrics"]
    out = ["# Evaluation report", "",
           f"- LLM: {result['llm_model']}; embeddings: {result['embed_backend']}; "
           f"gate threshold: {result['threshold']}",
           f"- {result['n_labeled']} labeled requirements; {result['n_reported']} reported "
           f"({result['split']})", "",
           "| System | Accuracy | False-match rate | Gap recall | Overclaim rate |",
           "| --- | --- | --- | --- | --- |"]
    for name, label in [("A_keyword", "A. Keyword overlap"), ("C_retrieval", "C. Retrieval + threshold"),
                        ("D_pipeline", "D. Full pipeline")]:
        s = m.get(name) or {}
        out.append(f"| {label} | {_pct(s.get('accuracy'))} | {_pct(s.get('false_match_rate'))} | "
                   f"{_pct(s.get('gap_recall'))} | {_pct(s.get('overclaim_rate'))} |")
    d = m.get("D_pipeline") or {}
    if d.get("confusion"):
        out += ["", "**Pipeline confusion (rows = your label, columns = prediction)**", "",
                "| label \\ pred | met | partial | missing |", "| --- | --- | --- | --- |"]
        for lab in LABELS:
            row = d["confusion"].get(lab, {})
            out.append(f"| {lab} | {row.get('met', 0)} | {row.get('partial', 0)} | {row.get('missing', 0)} |")
    if result.get("extraction"):
        e = result["extraction"]
        out += ["", f"**Requirement extraction recall:** {_pct(e['recall'])}"]
    if result.get("ranking"):
        r = result["ranking"]
        out += ["", f"**Ranking agreement with your fit scores** ({r['resume']}, {len(r['jds'])} JDs): "
                    f"whole-doc cosine Spearman {r['B_whole_doc_spearman']}, "
                    f"pipeline Spearman {r['D_pipeline_spearman']}"]
    p = result["pipeline"]
    out += ["", f"**Pipeline cost:** {p['llm_calls']} LLM calls ({p['cached_calls']} more served "
                f"from cache), {p['gated']} requirements gated, "
                f"${p['cost_usd']:.6f} total, ${p['cost_per_jd']:.6f} per resume-JD pair"]
    false_matches = [i for i in result["items"]
                     if i["label"] == "missing" and i["preds"].get("D_pipeline") == "met"]
    out += ["", "## False matches to read by hand", ""]
    if not false_matches:
        out.append("None.")
    for i in false_matches:
        out.append(f"- {i['jd']} / {i['req_id']}: {i['text']} (cited {i['evidence'] or '-'}): {i['reason']}")
    out += ["", "Per-item predictions: `predictions.csv`.", ""]
    return "\n".join(out)


def write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.md").write_text(eval_markdown(result), encoding="utf-8")
    (out_dir / "eval_result.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jd", "resume", "req_id", "requirement", "must_have", "label", "top_sim",
                    "A_keyword", "C_retrieval", "D_pipeline", "gated", "validated", "evidence", "reason"])
        for i in result["items"]:
            p = i["preds"]
            w.writerow([i["jd"], i["resume"], i["req_id"], i["text"], i["must_have"], i["label"],
                        i["top_similarity"], p.get("A_keyword"), p.get("C_retrieval"),
                        p.get("D_pipeline"), i["gated"], i["validated"], i["evidence"], i["reason"]])
