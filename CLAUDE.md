# Team rules for Claude (PS 99 Citation Auditor)

Read `README.md` for setup and the owner of each file. Architecture: `docs/architecture.html`.

## Ownership
- Each module names its owner in its docstring. Only edit files your person owns, plus the shared ones they asked for.
- Shared contracts: `auditor/schemas.py` and `auditor/indicators.py`. Adding an optional field or a synonym is fine. Renaming or removing a field breaks the other laptop, so ask first.
- Work items are marked `TODO(Aditya)` / `TODO(Sibi)`.

## Design rules (don't break these)
- Python does every calculation. The LLM only parses claims, reads passages and narrates verdicts.
- The Evidence Ledger comes first; vector/keyword retrieval is only for a ledger miss.
- Every number used as evidence keeps source, page, locator and a verbatim quote. The Citation Guard re-finds that quote.
- No evidence → `Unsupported`. Never guess. Sources that disagree → `conflict` flag, never pick one.
- Every LLM call must handle `None` (both providers down) with a deterministic fallback.

## Before you say it works
- `python -m pytest -q` passes, and the app starts with `streamlit run ui/app.py`.
- Layout: one top-level folder per layer (`evidence_layer/`, `trust_layer/`, `llm_router/`, `ui/`) plus shared `auditor/`. Keep each branch's changes inside its own folder so branches merge cleanly.

## Git
- Commit and push only when your person asks.
- Commits use your person's own git name and email. Never Claude as author or co-author, no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" lines.
- Pull before you push (`git pull --rebase`); two laptops push to `main`.
