# LLM Router

One entry point for every LLM call in the Citation Auditor (Claim Parser, Evidence Resolver,
verdict narration). It tries providers in order and always returns validated JSON, or tells you
clearly that nothing worked.

```
Gemini (primary, cloud) ──fail/429──▶ Groq (cloud) ──fail/429──▶ LM Studio (local :1234) ──fail──▶ regex parser
```

| Situation | What the router does |
|---|---|
| Timeout, 5xx, network blip | Retry same provider with backoff (1s, 2s), then next provider |
| Reply isn't valid JSON / doesn't match schema | Retry immediately with a "fix your JSON" note, then next provider |
| 429 rate limit, bad key, LM Studio not running | Move to next provider at once and skip this one for 60s |
| Every LLM failed | Run the `fallback` you passed (e.g. `parse_claims_regex`); else raise `AllProvidersFailed` |

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env      # then paste your GEMINI_API_KEY and GROQ_API_KEY
```

**LM Studio (local fallback):** install LM Studio, download a small instruct model, load it, then
Developer tab → **Start Server** (port 1234). Good small models for JSON extraction:

| Model (search in LM Studio) | Size (Q4) | Notes |
|---|---|---|
| `qwen2.5-3b-instruct` | ~2 GB | Best JSON / number handling at this size, recommended |
| `llama-3.2-3b-instruct` | ~2 GB | Solid alternative |
| `phi-3.5-mini-instruct` | ~2.4 GB | Good reasoning, slightly slower |
| `qwen2.5-7b-instruct` | ~4.7 GB | If the machine has 16 GB RAM / a GPU, noticeably more accurate |

Set `LMSTUDIO_MODEL` to the id LM Studio shows, or leave it empty to use whatever model is loaded.

## Use it

```python
from llm_router import LLMRouter, AllProvidersFailed, parse_claims_regex

router = LLMRouter.from_env()          # reads .env

res = router.call_json(
    task="parse_claim",
    system="Extract quantitative sustainability claims.",
    user=report_text,
    schema=CLAIM_SCHEMA,               # any JSON Schema; output is validated against it
    fallback=parse_claims_regex,       # optional last resort, no LLM
)
res.data          # validated dict
res.provider      # "gemini" | "groq" | "lmstudio" | "regex"
res.used_fallback # True if the primary didn't answer -> show a badge / lower confidence
res.attempts      # per-try log: provider, status, latency, error

text = router.call_text(task="narrate_verdict", system="...", user="...").data
router.status()   # configured / cooling-down state of each provider, for the dashboard
```

## Test

```powershell
python -m pytest tests -q                    # offline, no keys or models needed
python scripts/smoke_test_router.py          # live: each provider, then the full chain
python scripts/smoke_test_router.py --only lmstudio
```
