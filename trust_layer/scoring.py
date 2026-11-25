"""Confidence & Greenwash. Owner: Sibi.

Confidence measures evidence quality (source type, extraction method,
conflicts, guard result, corroboration), never retrieval similarity. 0 = no evidence.

Greenwash rules (each adds plain "greenwash" plus its own flag):
  greenwash:vague            vague claim, or a vague term with no number
  greenwash:no_target_year   "net zero" / "carbon neutral" / "climate neutral" with no year
  greenwash:offsets          relies on offsets, compensation or carbon credits
  greenwash:superlative      "best", "leading", "most sustainable", "greenest"... with no number
TODO(Sibi): tune weights on the answer key; add "missing baseline" greenwash rule.
"""
from __future__ import annotations

import re

from llm_router.claims import has_vague_term
from auditor.schemas import ExtractionMethod, Metric, SourceType, Status, Verdict
from trust_layer.units import normalize

AGREE_REL = 0.005  # same tolerance the verifier uses to call a conflict

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?")
_NEUTRAL = re.compile(r"\b(net[\s-]?zero|carbon[\s-]neutral(?:ity)?|climate[\s-]neutral(?:ity)?)\b", re.IGNORECASE)
_OFFSETS = re.compile(r"\b(offsets?|offsetting|offsetted|compensated|compensating|carbon[\s-]credits?)\b",
                      re.IGNORECASE)
_SUPERLATIVE = re.compile(
    r"\b(best(?![\s-]+(?:practices?|efforts?|available))|(?<!at )leading(?!\s+(?:to|up)\b)|"
    r"most\s+(?:sustainable|eco-friendly|environmentally\s+friendly|responsible)|greenest|cleanest|"
    r"number\s+one|no\.\s*1)\b", re.IGNORECASE)


def _has_number(text: str) -> bool:
    """A quantity in the text; a bare year is not a number that backs a claim."""
    return any(not _YEAR.fullmatch(n) for n in _NUMBER.findall(text))


def greenwash_flags(v: Verdict) -> list[str]:
    c = v.claim
    text = c.text or ""
    numberless = c.value is None and not _has_number(text)
    rules = []
    if c.metric == Metric.VAGUE or (has_vague_term(text) and c.value is None):
        rules.append("greenwash:vague")
    if _NEUTRAL.search(text) and not _YEAR.search(text):
        rules.append("greenwash:no_target_year")
    if _OFFSETS.search(text):
        rules.append("greenwash:offsets")
    if _SUPERLATIVE.search(text) and numberless:
        rules.append("greenwash:superlative")
    return ["greenwash", *rules] if rules else []


def greenwash(v: Verdict) -> bool:
    return bool(greenwash_flags(v))


def corroborated(v: Verdict) -> bool:
    """Every year used is backed by 2+ independent sources (distinct names) that agree."""
    by_year: dict[int, list] = {}
    for f in v.evidence:
        by_year.setdefault(f.year, []).append(f)
    if not by_year:
        return False
    for facts in by_year.values():
        if len({f.source for f in facts}) < 2:
            return False
        values = [normalize(f.value, f.unit, v.claim.indicator) for f in facts]
        nums = [x for x, _ in values]
        if len({u for _, u in values}) > 1 or max(nums) - min(nums) > AGREE_REL * max(abs(max(nums)), 1e-9):
            return False
    return True


def confidence(v: Verdict) -> float:
    """Trust in a Supported/Contradicted verdict. Unsupported has nothing to trust: 0."""
    if not v.evidence or v.status == Status.UNSUPPORTED:
        return 0.0
    score = 0.6
    methods = {f.method for f in v.evidence}
    if methods == {ExtractionMethod.TABLE}:
        score += 0.2
    if ExtractionMethod.LLM in methods:
        score -= 0.2
    if ExtractionMethod.TEXT in methods:
        score -= 0.1
    if any(f.source_type == SourceType.OFFICIAL for f in v.evidence):
        score += 0.2
    if corroborated(v):
        score += 0.1
    if v.conflict:
        score -= 0.3
    if "guard_failed" in v.flags:
        score -= 0.4
    return round(max(0.0, min(1.0, score)), 2)


def score(v: Verdict) -> Verdict:
    flags = list(v.flags)
    flags += [f for f in greenwash_flags(v) if f not in flags]
    return v.model_copy(update={"flags": flags, "confidence": confidence(v)})
