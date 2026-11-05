from datetime import date

from matcher.chunker import chunk_resume
from matcher.experience import total_years


def test_markdown_resume_ids_and_kinds(sample_resume_text):
    chunks = chunk_resume(sample_resume_text)
    by_id = {c.id: c for c in chunks}
    assert "Alex Rivera" not in [c.text for c in chunks]  # the name is not evidence
    assert by_id["exp1_h"].kind == "role_header"
    assert "Jan 2022" in by_id["exp1_h"].text  # date line merged into the heading
    assert by_id["exp1_b1"].text.startswith("Led a team of 4")
    assert by_id["exp2_b3"].text.endswith("Kotlin Multiplatform")
    assert by_id["exp3_b2"].role.startswith("Android Developer, Pixel Forge")
    assert by_id["skills_1"].kind == "skills"
    assert len({c.id for c in chunks}) == len(chunks)


def test_plain_text_resume_from_pdf_export():
    text = """Jane Doe
EXPERIENCE
Senior Engineer, Acme  2019 - Present
• Built payment APIs in Kotlin
• Ran on-call for 3 services
Engineer, Beta Corp  2016 - 2019
• Wrote Android features
SKILLS
Kotlin, Go
"""
    chunks = chunk_resume(text)
    ids = [c.id for c in chunks]
    assert "exp1_h" in ids and "exp1_b2" in ids and "exp2_h" in ids and "exp2_b1" in ids
    assert any(c.kind == "skills" and "Go" in c.text for c in chunks)


def test_long_paragraph_is_split():
    para = " ".join(["This sentence talks about mobile work in some detail."] * 20)
    chunks = chunk_resume(f"## Summary\n\n{para}\n")
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)


def test_total_years_merges_overlaps():
    text = """## Experience
### A
Jan 2020 – Dec 2021
- x
### B
Jun 2021 – Dec 2022
- y
"""
    assert total_years(chunk_resume(text)) == 3.0


def test_total_years_present_and_year_only():
    text = "## Experience\n### A\n2024 - Present\n- x\n"
    assert total_years(chunk_resume(text), today=date(2026, 6, 15)) == 2.5


def test_no_dates_gives_none():
    assert total_years(chunk_resume("## Experience\n- did things\n")) is None
