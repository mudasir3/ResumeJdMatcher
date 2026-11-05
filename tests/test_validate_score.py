from matcher.models import (Candidate, Chunk, Requirement, RequirementResult, Verdict)
from matcher.score import missing_must_haves, weighted_score
from matcher.validate import validate_verdict

BULLET = Chunk(id="exp1_b1", text="Led a team of 4 Android engineers in Kotlin", kind="bullet")
SKILLS = Chunk(id="skills_1", text="Kotlin, GraphQL, Python", kind="skills")
HEADER = Chunk(id="exp1_h", text="Lead Engineer | Jan 2022 – Present", kind="role_header")
REQ = Requirement(id="r1", text="Team leadership", must_have=True)


def v(verdict, ids, quote, gap=None):
    return Verdict(requirement_id="r1", verdict=verdict, evidence_ids=ids, evidence_quote=quote,
                   reason="x", gap_type=gap)


def test_valid_met_passes():
    final, passed, down, notes = validate_verdict(v("met", ["exp1_b1"], "Led a team of 4"), REQ,
                                                  [BULLET], None)
    assert final.verdict == "met" and passed and down is None and final.gap_type is None


def test_quote_matching_ignores_case_and_dashes():
    final, passed, _, _ = validate_verdict(v("met", ["exp1_h"], "jan 2022 - present"), REQ,
                                           [HEADER], None)
    assert passed and final.verdict == "met"


def test_invented_quote_is_downgraded():
    final, passed, down, notes = validate_verdict(
        v("met", ["exp1_b1"], "Led a team of 12 engineers"), REQ, [BULLET], None)
    assert final.verdict == "missing" and not passed and down == "met"
    assert final.gap_type == "real_gap" and final.evidence_ids == []


def test_unknown_id_is_dropped_then_downgraded():
    final, passed, _, notes = validate_verdict(v("met", ["exp9_b9"], "Led a team"), REQ, [BULLET], None)
    assert final.verdict == "missing" and not passed
    assert any("exp9_b9" in n for n in notes)


def test_missing_quote_is_downgraded():
    final, passed, _, _ = validate_verdict(v("partial", ["exp1_b1"], None), REQ, [BULLET], None)
    assert final.verdict == "missing"


def test_skills_line_only_caps_at_partial():
    final, passed, down, _ = validate_verdict(v("met", ["skills_1"], "GraphQL"), REQ, [SKILLS], None)
    assert passed and final.verdict == "partial" and down == "met" and final.gap_type == "reword"


def test_years_short_caps_at_partial():
    req = Requirement(id="r1", text="10+ years mobile", must_have=True, category="experience", years=10)
    final, _, down, notes = validate_verdict(v("met", ["exp1_h"], "Lead Engineer"), req, [HEADER], 8.5)
    assert final.verdict == "partial" and down == "met"
    assert "8.5 years < 10 required" in notes[-1]


def test_missing_verdict_untouched():
    final, passed, down, _ = validate_verdict(v("missing", [], None), REQ, [BULLET], None)
    assert final.verdict == "missing" and passed and down is None and final.gap_type == "real_gap"


def _result(must, verdict):
    req = Requirement(id=f"r{must}{verdict}", text="t", must_have=must)
    return RequirementResult(requirement=req, verdict=Verdict(requirement_id=req.id, verdict=verdict),
                             candidates=[Candidate(chunk_id="c", similarity=0.5)], top_similarity=0.5)


def test_weighted_score_formula():
    results = [_result(True, "met"), _result(True, "missing"), _result(False, "partial")]
    # (2*1 + 2*0 + 1*0.5) / (2 + 2 + 1) = 0.5
    assert weighted_score(results) == 50.0
    assert missing_must_haves(results) == ["rTruemissing"]


def test_empty_score_is_zero():
    assert weighted_score([]) == 0.0
