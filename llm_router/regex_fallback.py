"""Last-resort, no-LLM claim parser used when every provider has failed.

There is one regex parser in this package: ``claims.py``. This module only adapts its
output to the plain-dict shape the router's ``fallback=`` hook (and its schema check)
expects, so the router marks results from it with provider="regex" and the caller can
show a warning / lower confidence.

Pass it to the router as ``fallback=parse_claims_regex``.
"""

from __future__ import annotations

from .claims import parse_report

#: JSON Schema of what parse_claims_regex returns (handy as ``schema=`` for call_json).
CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "object"}}},
    "required": ["claims"],
}


def parse_claims_regex(text: str) -> dict:
    """``text`` -> {"claims": [claim dicts], "parser": "regex"} using claims.parse_report (no LLM).

    Each dict has the auditor.schemas.Claim fields (claim_id, text, indicator, metric,
    direction, value, unit, baseline_year, target_year, page, entity) with enums as strings.
    """
    claims = [c.model_dump(mode="json") for c in parse_report(text)]
    return {"claims": claims, "parser": "regex"}
