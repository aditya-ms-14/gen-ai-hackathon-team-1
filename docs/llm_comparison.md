# LLM model comparison: PS 99 Citation Auditor

- Date: 2026-11-25 (Groq group 13:45, Gemini group 14:02)
- Command: `python scripts/compare_llms.py --modes <provider:model,...>` (two runs, merged here)
- Same 34 claims for every mode (PS sample 6, country demo 13, company report 15); one provider and model forced per mode, no fallback between providers. A call that gets no usable answer falls back to the deterministic regex parser or template text, so "answered" shows how much of each row the model itself produced.
- Calls spaced to stay under free-tier rate limits (Gemini 6.5 s, Groq 2.2 s apart); "mean LLM s/call" excludes that waiting.

| Mode | Claims found | Verdict accuracy | Flags | Explanations rejected by Citation Guard | LLM calls answered | Mean LLM s/call | Total time (s) |
|---|---|---|---|---|---|---|---|
| Baseline (regex, no LLM) | 34/34 | 34/34 (100.0%) | 5/5 | n/a (templates) | - | - | 0.5 |
| Groq `openai/gpt-oss-120b` | 34/34 | 30/34 (88.2%) | 5/5 | 4/35 | 18/41 | 0.64 | 110.6 |
| Groq `openai/gpt-oss-20b` | 34/34 | 32/34 (94.1%) | 5/5 | 0/34 | 8/40 | 0.29 | 93.5 |
| Groq `qwen/qwen3.8-27b` | 34/34 | 33/34 (97.1%) | 5/5 | 0/34 | 11/40 | 0.22 | 90.7 |
| Gemini `gemini-flash-lite-latest` | 34/34 | 33/34 (97.1%) | 5/5 | 1/34 | 40/40 | 4.10 | 405.1 |
| Gemini `gemini-3.5-flash` | not measurable | not measurable | - | 0/34 | 1/40 | - | 256.3 |

`gemini-3.5-flash`: Google returned 503 (overloaded) for 39 of 40 calls, so its 34/34 would really be the regex fallback; it is not scored.

## Mismatches

- gpt-oss-120b · PS sample · C01 "Carbon emissions reduced by 40%": expected Contradicted, got Unsupported.
- gpt-oss-120b · Country demo · C07, C08, C09 (Germany renewable energy, Brazil renewable electricity, Brazil forest area): expected Supported / Contradicted / Supported, got Unsupported.
- gpt-oss-20b · PS sample · C01 and C03 ("Renewable energy usage reached 75%"): got Unsupported.
- qwen3.8-27b · PS sample · C01: got Unsupported.
- gemini-flash-lite-latest: one mismatch (see the Gemini run output).

## Findings

- Every error is a claim-reading difference (wrong indicator or period in the claim spec, so no matching evidence is found), never arithmetic: Python computes every verdict in every mode. "Carbon emissions reduced by 40%" was missed by all three Groq models, most likely labelled CO2 instead of total GHG emissions.
- The Citation Guard caught 4 explanations from gpt-oss-120b and 1 from gemini-flash-lite-latest that stated numbers or verdict words the evidence does not support; template text replaced them.
- Low "answered" counts on Groq come from free-tier rate limits and from gpt-oss-120b (a reasoning model) sometimes returning empty text.

## Limits

- 34 claims is a small benchmark; one claim moves accuracy by about 3 points.
- The test reports use phrasings the regex parser handles, so the baseline is favoured; LLMs are expected to matter more on free-form real reports.
- Next: let the deterministic synonym match override the LLM's indicator, hide reasoning output for gpt-oss models, and repeat on published reports.
