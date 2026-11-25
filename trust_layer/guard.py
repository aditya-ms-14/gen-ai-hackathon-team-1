"""Citation & Hallucination Guard. Owner: Sibi.

1. Every evidence quote must appear word for word in its source page,
   otherwise the verdict drops to Unsupported (flag "guard_failed").
2. Every number in the explanation must be a claim, computed, year or
   evidence value (or one of those rounded to 0/1/2 decimals, with or without
   thousands separators, %, or "million"/"billion"); digits inside source names,
   URLs, locators, series codes ("EN.GHG.CO2.MT.CE.AR5") and entity names are ignored.
3. The explanation must not state a different verdict word than the status
   ("supported" when the status is Contradicted).
If 2 or 3 fails, or 1 changed the status, the deterministic template replaces
the text (flag "narration_replaced").
"""
from __future__ import annotations

import re
import unicodedata

from trust_layer.explain import entity_name, template_explanation
from auditor.schemas import Status, Verdict
from trust_layer.units import normalize

# 1,234,567.8 | 1234.5 | 42 ; not glued to letters/digits/dots (CO2, AR5, p.1, v2.0)
_NUM = re.compile(r"(?<![A-Za-z\d.])(\d{1,3}(?:,\d{3})+(?![\d])|\d+)(\.\d+)?(?:\s*(thousand|million|billion|bn|trillion)\b)?",
                  re.IGNORECASE)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_SERIES = re.compile(r"\b[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+){2,}\b")  # EN.GHG.CO2.MT.CE.AR5
_SCOPE = re.compile(r"\bscopes?\s+[123](?:\s*(?:,|and|&|\+)\s*[123])*\b", re.IGNORECASE)

# Verdict words. Negated support ("not supported", "cannot be verified") counts as Unsupported;
# negated contradiction ("does not contradict") counts as nothing.
_NOT = (r"\b(?:not|never|cannot|can't|couldn't|isn't|wasn't|doesn't|didn't)\s+"
        r"(?:be\s+|been\s+|fully\s+|yet\s+|be\s+fully\s+)?")
_NEGATED = re.compile(_NOT + r"(?:supported|verified|confirmed|support|verify|confirm)\b", re.IGNORECASE)
_NEGATED_CONTRA = re.compile(_NOT + r"(?:contradicted|contradicts?)\b", re.IGNORECASE)
_WORDS = {
    Status.UNSUPPORTED: re.compile(r"\b(unsupported|unverified|unverifiable)\b", re.IGNORECASE),
    Status.CONTRADICTED: re.compile(r"\b(contradicted|contradicts?|contradictory|refuted|disproven|disproved)\b",
                                    re.IGNORECASE),
    Status.SUPPORTED: re.compile(r"\b(supported|verified|confirmed)\b", re.IGNORECASE),
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def quotes_ok(v: Verdict, source_pages: dict[str, dict[int, str]]) -> bool:
    for fact in v.evidence:
        pages = source_pages.get(fact.source)
        if not pages:
            return False
        haystack = pages.get(fact.page, "") if fact.page else " ".join(pages.values())
        if not fact.quote.strip() or _norm(fact.quote) not in _norm(haystack):
            return False
    return True


def _allowed(v: Verdict) -> list[float]:
    values = [v.claim.value, v.computed_value, v.claim.baseline_year, v.claim.target_year,
              *(f.value for f in v.evidence), *(f.year for f in v.evidence),
              *(normalize(f.value, f.unit, f.indicator)[0] for f in v.evidence)]  # same value in base units
    if v.calculation:
        values += [n for n, _ in _numbers(_strip_names(v.calculation, v))]
    out = set()
    for x in values:
        if x is not None:
            out |= {float(x), abs(float(x))}
    return sorted(out)


def _numbers(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for each number; '700 million' -> 700e6 (scaled values also accept 700)."""
    out = []
    for m in _NUM.finditer(text):
        whole, frac, scale = m.group(1).replace(",", ""), m.group(2) or "", m.group(3)
        value = float(whole + frac)
        decimals = max(len(frac) - 1, 0)
        if scale:
            out.append((value * _SCALE[scale.lower()], -1))  # -1: either form may match
        out.append((value, decimals))
    return out


def _strip_names(text: str, v: Verdict) -> str:
    """Blank out text whose digits are names, not claims."""
    names = {f.source for f in v.evidence} | {f.locator for f in v.evidence if f.locator}
    names |= {f.url for f in v.evidence if f.url}
    codes = {v.claim.entity, *(f.entity for f in v.evidence)} - {None}
    names |= codes | {entity_name(c) for c in codes}
    names = {n for n in names if n and re.search(r"\d", n) and re.search(r"[A-Za-z]", n)}  # never a bare number
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    text = _URL.sub(" ", text)
    text = _SERIES.sub(" ", text)
    return _SCOPE.sub(" ", text)


def _close(n: float, decimals: int, allowed: list[float]) -> bool:
    if decimals < 0:  # scaled form ("700 million"): exact-ish only
        return any(abs(n - x) <= 1e-6 * max(abs(x), 1.0) for x in allowed)
    tol = 0.5 * 10 ** -decimals + 1e-9  # "675" / "674.8" / "674.75" all match 674.7531
    return any(abs(n - x) <= tol for x in allowed)


def numbers_ok(text: str, v: Verdict) -> bool:
    allowed = _allowed(v)
    nums = _numbers(_strip_names(text, v))
    # a scaled pair ("700 million") passes if either reading is allowed
    i = 0
    while i < len(nums):
        n, d = nums[i]
        if d < 0:
            if not (_close(n, d, allowed) or _close(*nums[i + 1], allowed)):
                return False
            i += 2
            continue
        if not _close(n, d, allowed):
            return False
        i += 1
    return True


def stated_verdicts(text: str) -> set[Status]:
    """Verdict words the text uses, e.g. {Status.SUPPORTED} for 'the claim is supported'."""
    found = set()
    text = _NEGATED_CONTRA.sub(" ", text)
    if _NEGATED.search(text):
        found.add(Status.UNSUPPORTED)
        text = _NEGATED.sub(" ", text)
    for status, pattern in _WORDS.items():
        if pattern.search(text):
            found.add(status)
            text = pattern.sub(" ", text)  # "unsupported" must not also count as "supported"
    return found


def verdict_ok(text: str, v: Verdict) -> bool:
    return stated_verdicts(text) <= {v.status}


def apply_guard(v: Verdict, source_pages: dict[str, dict[int, str]]) -> Verdict:
    flags = list(v.flags)
    status = v.status
    if v.evidence and not quotes_ok(v, source_pages):
        status = Status.UNSUPPORTED
        flags.append("guard_failed")
    checked = v.model_copy(update={"status": status, "flags": flags})
    explanation = v.explanation
    if status != v.status or not numbers_ok(explanation, checked) or not verdict_ok(explanation, checked):
        explanation = template_explanation(checked)
        flags.append("narration_replaced")
    return checked.model_copy(update={"flags": flags, "explanation": explanation})
