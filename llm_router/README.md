# LLM Router

One package, one router, for every LLM call in the Citation Auditor. The LLM only parses claims,
reads passages and narrates verdicts; Python does all the math. Router by Sibi; the adapter, claim
parser and the hardening notes below were merged in on top.

```
Gemini (primary, cloud) --fail/429--> Groq (cloud) --fail/429--> LM Studio (local, 127.0.0.1:1234) --fail--> regex parser
```

| Situation | What the router does |
|---|---|
| Timeout | One retry (backoff), then next provider; the provider is skipped for 30 s |
| 5xx, network blip | Retry with backoff (1 s, 2 s), then next provider; skipped for 30 s |
| Reply isn't JSON / doesn't match the schema | Retry at once with a "fix your JSON" note, then next provider |
| 429, bad key / 404, LM Studio not running | Next provider at once; this one is skipped for 60 s (a `Retry-After` is honoured, capped at 10 min) |
| Every LLM failed | Run the `fallback` you passed (e.g. `parse_claims_regex`); else raise `AllProvidersFailed` |

A down LM Studio costs one 1.5 s connect attempt (Windows needs ~2 s to refuse a local connection),
then it is skipped until its cool-down ends. Nothing waits longer than the configured timeouts.

## Files

| File | Purpose |
|---|---|
| `router.py` | `LLMRouter`: fallback chain, retries, cool-downs, schema validation + repair prompt, per-provider stats |
| `providers.py` | `GeminiProvider` (native REST), `GroqProvider`, `LMStudioProvider` (OpenAI-compatible, auto-detects the loaded chat model) |
| `config.py` | `RouterConfig`, `.env` loader, env vars |
| `json_utils.py` | `extract_json` (chatter, fences, `<think>`, trailing commas), `schema_errors` |
| `errors.py` | `ProviderError`, `AllProvidersFailed`, `redact` (keys never reach logs or messages) |
| `adapter.py` | `TeamRouter`, `NullRouter`, `make_router()`: the interface the auditor pipeline uses |
| `claims.py` | Claim parser (regex + LLM) used by the pipeline; also `has_vague_term` |
| `regex_fallback.py` | `parse_claims_regex(text)`: dict view of `claims.parse_report`, for the router's `fallback=` hook |

## Providers and environment variables

Copy `.env.example` to `.env` (real environment variables win over `.env`). Keys are never logged.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER_ORDER` | `gemini,groq,lmstudio` | Fallback order |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | | Gemini; no key = skipped |
| `GEMINI_MODEL` | `gemini-3.5-flash` | |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | A trailing `/openai` (old `.env.example`) is stripped |
| `GEMINI_THINKING_BUDGET` | model default | `0` turns thinking off on flash models |
| `GROQ_API_KEY`, `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq; no key = skipped |
| `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | `localhost` is rewritten to `127.0.0.1` (IPv6 delay on Windows) |
| `LMSTUDIO_MODEL` | empty | Empty = the loaded chat model (embedding models are skipped) |
| `LLM_TIMEOUT_SECONDS` | 30 | Gemini / Groq request timeout (`GEMINI_TIMEOUT`, `GROQ_TIMEOUT` override); `LMSTUDIO_TIMEOUT` is 120 |
| `LLM_MAX_RETRIES` / `LLM_TIMEOUT_RETRIES` | 2 / 1 | Extra attempts / extra attempts after a timeout |
| `LLM_BACKOFF_BASE`, `LLM_BACKOFF_MAX` | 1, 8 | Seconds |
| `LLM_RATE_LIMIT_COOLDOWN`, `LLM_FAILURE_COOLDOWN` | 60, 30 | Seconds a provider is skipped |
| `LLM_MAX_TOKENS`, `LLM_TEMPERATURE` | 4096, 0 | |

Malformed numbers fall back to the default instead of crashing.

**LM Studio:** load a small instruct model (`qwen2.5-3b-instruct`, `llama-3.2-3b-instruct`,
`phi-3.5-mini-instruct`; `qwen2.5-7b-instruct` if you have 16 GB RAM or a GPU), then Developer tab,
Start Server. JSON mode is negotiated per server: `json_schema`, then `json_object`, then plain
prompting; a refused mode is remembered. A chat template that rejects the system role gets the
system prompt merged into the user message.

## Use it

For the auditor (never raises, every failure is `None`):

```python
from llm_router import make_router

llm = make_router()
llm.complete(system, user)                    # str | None            narration
llm.complete_json(system, user, schema=None)  # dict | None           JSON; a list reply -> {"items": [...]}
llm.status()                                  # {"gemini": "gemini-3.5-flash · 3 calls, 1 failed", ...}
llm.last_provider                             # "gemini" | "groq" | "lmstudio" | "regex" | None
llm.stats                                     # {"gemini": {"calls": 3, "failures": 1}, ...}
```

`make_router()` never raises. If the router cannot be built (e.g. an unknown name in
`LLM_PROVIDER_ORDER`) it returns a `NullRouter`: calls return `None` and `status()` explains why.

The router itself:

```python
from llm_router import LLMRouter, AllProvidersFailed, parse_claims_regex

router = LLMRouter.from_env()
res = router.call_json(system="...", user=text, schema=SCHEMA, task="parse_claim",
                       fallback=parse_claims_regex)      # optional, used only if every LLM fails
res.data; res.provider; res.used_fallback; res.attempts  # validated dict, who answered, per-try log
router.call_text(system="...", user="...").data
router.status()   # [{"provider", "model", "configured", "cooldown_s", "calls", "failures", "last_error"}]
```

## Claim parser

`claims.parse_pages({page: text}, report_year, llm)` and `parse_report(text, report_year, llm)` return
`Claim` objects. With an `llm` it makes one strict-JSON call per page; a returned claim is kept only
if its sentence is on the page and its numbers and years are in that sentence, otherwise the
deterministic regex parser (`parse_claim_regex`) decides. Without an LLM, or when it fails, regex alone is used.
`regex_fallback.parse_claims_regex` is the same parser as a dict, so there is one regex implementation.

## Test

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider llm_router          # offline: fakes and mocked HTTP only
.venv\Scripts\python.exe llm_router\scripts\smoke_test.py                     # live: each provider, then the full chain
.venv\Scripts\python.exe llm_router\scripts\smoke_test.py --only lmstudio
```

`tests/test_router.py` covers the router, providers (mocked HTTP), JSON extraction and config;
`tests/test_llm.py` covers the adapter, `make_router` and the claim parser. The smoke test needs real
keys or a running LM Studio; unconfigured or unreachable providers are reported as SKIPPED.
