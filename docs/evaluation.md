# Evaluation: PS 99 Environmental Report Citation Auditor

- Date: 2026-11-25 11:46
- Command: `python scripts/evaluate.py` (run from the repo root with `PYTHONPATH` set to the repo root)
- Mode: deterministic, `use_llm=False` (regex claim parser, template explanations); reproducible.
- Official data: official data used (live API when reachable, otherwise the curated snapshots; the pipeline does not record which one served each fact). Only the Country demo uses it; the PS sample and the company dataset run with `use_official=False` (their claims name no country).
- Overall: 34/34 claims given the expected status.

Datasets: **PS sample** (`sample_report_2025.txt` + `evidence_2024_2025.csv`, 6 claims), **Country demo** (`demo_country_report.txt`, no evidence files, 13 claims), **Company (Greenfield)** (fictional PDF report p.2-4 + ruled-table PDF + XLSX; built by `scripts/make_company_dataset.py`, 15 claims including a PDF/XLSX conflict and two greenwash sentences).

Matching: claims are paired with the answer key by sentence (company) or claim id (others). A claim the parser did not extract counts as wrong. "Cited" = every evidence fact of a Supported/Contradicted verdict has source, locator and quote. Mean confidence is over checked verdicts only.

## Results (deterministic)

### Summary

| Dataset | Claims in key | Extracted (recall) | Status accuracy | Flag accuracy | Cited (checked verdicts) | Guard failures | Mean conf. Supported / Contradicted | Time (s) | s/claim |
|---|---|---|---|---|---|---|---|---|---|
| PS sample | 6 | 6/6 (100.0%) | 6/6 (100.0%) | 1/1 | 3/3 | 0 | 0.80 / 0.80 | 0.02 | 0.003 |
| Country demo | 13 | 13/13 (100.0%) | 13/13 (100.0%) | 1/1 | 9/9 | 0 | 1.00 / 1.00 | 0.11 | 0.008 |
| Company (Greenfield) | 15 | 15/15 (100.0%) | 15/15 (100.0%) | 3/3 | 8/8 | 0 | 0.90 / 0.90 | 0.42 | 0.028 |

**PS sample** per-status precision / recall (support = expected count)

| Status | Precision | Recall | Support |
|---|---|---|---|
| Supported | 100.0% | 100.0% | 2 |
| Contradicted | 100.0% | 100.0% | 1 |
| Unsupported | 100.0% | 100.0% | 3 |

Confusion matrix (rows = expected, columns = got):

| expected \ got | Supported | Contradicted | Unsupported | (not extracted) |
|---|---|---|---|---|
| Supported | 2 | 0 | 0 | 0 |
| Contradicted | 0 | 1 | 0 | 0 |
| Unsupported | 0 | 0 | 3 | 0 |

**Country demo** per-status precision / recall (support = expected count)

| Status | Precision | Recall | Support |
|---|---|---|---|
| Supported | 100.0% | 100.0% | 5 |
| Contradicted | 100.0% | 100.0% | 4 |
| Unsupported | 100.0% | 100.0% | 4 |

Confusion matrix (rows = expected, columns = got):

| expected \ got | Supported | Contradicted | Unsupported | (not extracted) |
|---|---|---|---|---|
| Supported | 5 | 0 | 0 | 0 |
| Contradicted | 0 | 4 | 0 | 0 |
| Unsupported | 0 | 0 | 4 | 0 |

**Company (Greenfield)** per-status precision / recall (support = expected count)

| Status | Precision | Recall | Support |
|---|---|---|---|
| Supported | 100.0% | 100.0% | 5 |
| Contradicted | 100.0% | 100.0% | 3 |
| Unsupported | 100.0% | 100.0% | 7 |

Confusion matrix (rows = expected, columns = got):

| expected \ got | Supported | Contradicted | Unsupported | (not extracted) |
|---|---|---|---|---|
| Supported | 5 | 0 | 0 | 0 |
| Contradicted | 0 | 3 | 0 | 0 |
| Unsupported | 0 | 0 | 7 | 0 |

## Errors and analysis

Deterministic run:

No mismatches: every expected status, flag and computed value was reproduced.

## Limitations

- Tiny datasets (6 / 13 / 15 claims): a single claim moves accuracy by 7-17 points; treat as a smoke benchmark, not a statistical estimate.
- The company report was written by us to match the documented parser phrasings, so extraction recall is optimistic; real reports are more varied.
- Expected verdicts come from the project's own rules (tolerance +/-1.0 pp, 2% relative for amounts), so the key tests implementation consistency, not ground truth.
