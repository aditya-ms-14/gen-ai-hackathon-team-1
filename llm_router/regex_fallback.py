"""Last-resort, no-LLM claim parser used when every provider has failed.

It is deliberately simple: it finds sentences that contain a number and pulls
out value, unit, year and a percent-change direction. The output is
lower-quality than an LLM's, so the router marks results from it with
provider="regex" and the caller can show a warning / lower confidence.

Pass it to the router as ``fallback=parse_claims_regex`` (or wrap it to map
into the Claim Parser's own schema).
"""

from __future__ import annotations

import re

# Longest units first so "tCO2e" wins over "t".
_UNITS = [
    "million tonnes", "million tons", "tonnes CO2e", "tons CO2e", "tCO2e", "MtCO2e", "ktCO2e", "kgCO2e", "CO2e",
    "tonnes", "tons", "kg", "GWh", "MWh", "kWh", "GJ", "TJ", "MW", "GW", "m3", "megalitres", "ML", "litres", "liters",
    "employees", "hectares", "ha", "USD", "EUR", "INR", "crore", "lakh", "million", "billion", "%", "percent",
]
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_NUM = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
_VALUE_RE = re.compile(rf"(?<![\w.])(?P<num>{_NUM})\s*(?P<unit>{_UNIT_RE})?(?![\w])", re.IGNORECASE)
# Numbers that are part of a label, not a metric: "Scope 1", "SDG 13", "Tier 2", "Phase 3", "Section 4".
_LABEL_RE = re.compile(r"\b(scope|sdg|tier|phase|section|goal|page|table|figure|fig\.?|no\.?|category|cat\.?)\s*$", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:FY\s?)?(19[5-9]\d|20\d{2})\b", re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

_DOWN = re.compile(r"\b(reduc\w*|decreas\w*|cut|lower\w*|declin\w*|fell|fall\w*|drop\w*)\b", re.IGNORECASE)
_UP = re.compile(r"\b(increas\w*|grew|grow\w*|rais\w*|rose|ris\w*|improv\w*|expand\w*)\b", re.IGNORECASE)
_TARGET = re.compile(r"\b(target\w*|aim\w*|commit\w*|pledg\w*|goal|by 20\d{2}|will)\b", re.IGNORECASE)
_STOP = {"the", "a", "an", "our", "we", "in", "of", "by", "to", "and", "was", "were", "is", "has", "have", "from", "at", "for", "with", "total"}


def _indicator(sentence: str, start: int) -> str:
    """Best-guess indicator name: the few content words before the number."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", sentence[:start])
    words = [w for w in words if w.lower() not in _STOP and not _DOWN.fullmatch(w) and not _UP.fullmatch(w)]
    return " ".join(words[-4:]).lower()


def parse_claims_regex(text: str) -> dict:
    claims = []
    for sentence in _SENT_RE.split(text.strip()):
        sentence = sentence.strip()
        years = [int(y) for y in _YEAR_RE.findall(sentence)]
        for m in _VALUE_RE.finditer(sentence):
            if _LABEL_RE.search(sentence[: m.start()]):
                continue
            raw = m.group("num")
            value = float(raw.replace(",", ""))
            if int(value) in years and not m.group("unit") and float(int(value)) == value:
                continue  # it's the year itself, not a metric
            unit = (m.group("unit") or "").strip()
            if unit.lower() == "percent":
                unit = "%"
            direction = "decrease" if _DOWN.search(sentence) else "increase" if _UP.search(sentence) else None
            claims.append({
                "text": sentence,
                "indicator": _indicator(sentence, m.start()),
                "value": value,
                "unit": unit or None,
                "year": years[-1] if years else None,
                "baseline_year": years[0] if len(years) > 1 else None,
                "is_percent_change": unit == "%" and direction is not None,
                "direction": direction,
                "is_target": bool(_TARGET.search(sentence)),
            })
    return {"claims": claims, "parser": "regex"}
