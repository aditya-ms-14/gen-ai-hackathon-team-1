# LLM model comparison: PS 99 Citation Auditor

- Date: 2026-11-25. Command: `python scripts/compare_llms.py --modes <provider:model,...>` (Groq and Gemini runs, merged here).
- Same 34 claims for every mode (PS sample 6, country demo 13, company report 15). One provider and model forced per mode, no fallback between providers. A call that gets no usable answer falls back to the regex parser or template text, so **"answered" shows how much of each row the model itself produced**.
- Calls spaced to stay under free-tier rate limits (Gemini 6.5 s, Groq 2.2 s); "s/call" excludes that waiting.

## Current results (after the indicator fix, 14:23–14:40)

| Mode | Claims found | Verdict accuracy | First run (before fix) | Explanations rejected by Citation Guard | LLM calls answered | Mean LLM s/call |
|---|---|---|---|---|---|---|
| Baseline (regex, no LLM) | 34/34 | 34/34 (100%) | 100% | n/a (templates) | - | - |
| Groq `openai/gpt-oss-120b` | 34/34 | 34/34 (100%) | 88.2% | 2/34 | 22/40 | 0.74 |
| Groq `openai/gpt-oss-20b` | 34/34 | 34/34 (100%) | 94.1% | 0/34 | 4/40 | 0.35 |
| Groq `qwen/qwen3.8-27b` | 34/34 | 34/34 (100%) | 97.1% | 0/34 | 7/40 | 0.20 |
| Gemini `gemini-flash-lite-latest` | 34/34 | 34/34 (100%) | 97.1% | 2/34 | **40/40** | 2.53 |
| Gemini `gemini-3.5-flash` | not measurable | not measurable | not measurable | 0/34 | 0/40 | - |
| LM Studio (local) | not run | | | | | |

- `gemini-3.5-flash`: Google returned 503 "high demand" on every call, so its score would be the regex fallback; not scored.
- LM Studio: installed, but its local server was not started, so it was not measured.
- Gemini Flash-Lite is the cleanest LLM result: it answered every call itself and still matched all 34 verdicts. The Groq models answered fewer calls because of free-tier rate limits (and gpt-oss-120b sometimes returns empty text), so their 100% is partly the regex fallback.

## What changed between the runs

First run (13:45–14:02): every Groq model read "Carbon emissions reduced by 40%" as CO2 only, so the total-emissions evidence did not match and the claim came out Unsupported instead of Contradicted. Other first-run misses: gpt-oss-120b on Germany renewable energy (demo C07), Brazil renewable electricity (C08), Brazil forest area (C09); gpt-oss-20b on "Renewable energy usage reached 75%" (PS C03).

Fix (`llm_router/claims.py`, `_claim_from_llm`): when the claim's wording matches an indicator synonym, that mapping decides the indicator; the LLM's own indicator is used only when the synonyms find none. After the fix every model reproduced every expected verdict.

## Findings

- Errors were always claim reading (wrong indicator or period), never arithmetic: Python computes every verdict in every mode.
- The Citation Guard rejected 4 explanations in the first run (gpt-oss-120b) and 4 after the fix (gpt-oss-120b 2, Gemini Flash-Lite 2): numbers or verdict words not supported by the evidence, replaced with template text.
- Live LLM runs vary by provider and run: the India industry report gave 13/13 with Gemini Flash-Lite reading every page, and 10/13 in a mixed run where Groq read some pages before its quota ran out. Deterministic mode is always 61/61.

## Limits

- 34 claims is a small benchmark; one claim moves accuracy by about 3 points.
- The test reports use phrasings the regex parser handles, so the baseline is favoured.
