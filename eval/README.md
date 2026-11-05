# Golden set

The files here are a **fictional sample** (resume of "Alex Rivera", 4 short JDs, 37 labels) so the eval runs out of the box. Replace them with your own before reporting any numbers.

## Build your real set

1. Save about 10 JDs you would actually apply to in `jds/` as `.txt`: e.g. 4 senior/lead Android, 2 KMP, 2 mobile + AI, 2 deliberate mismatches.
2. Put your resume in `resumes/` as Markdown, plus one variant with a role's bullets removed (you then know exactly which verdicts should change).
3. Run `matcher chunks -r resumes/<file>` to see the chunk IDs.
4. Fill `labels.csv` **before running the tool**, so its output can't bias you:

| column | meaning |
| --- | --- |
| jd, resume | file names in `jds/` and `resumes/` |
| req_id | r1, r2 … per JD |
| requirement_text | one atomic requirement, close to the JD wording |
| must_have | true / false |
| category | experience, skill, domain, education, other |
| years | number if the JD states one, else empty |
| label | met, partial, missing |
| evidence_id | the chunk ID that proves it (empty for missing) |

5. Optional `fit_ranking.csv` (`jd,my_fit` on a 1–5 scale): your own judgment, used to check whether scores rank JDs the way you would.

Include adversarial cases: a skill you don't have, a synonym with no shared keyword, a skill only in your skills line, a years requirement above your total, adjacent tech (Flutter vs React Native/KMP), a vague soft skill.

## Run

```bash
matcher tune                 # suggested THRESHOLD (no LLM calls)
matcher eval --tune-jds 3    # tune the gate on the first 3 JDs, report on the rest
```

Outputs: `eval_report.md` (summary + false matches to read by hand), `eval_result.json`, `predictions.csv` (one row per requirement, all systems).
