"""Explanation writer: the LLM narrates an already-decided verdict. Owner: Sibi.

The LLM gets only the verdict JSON; the Citation Guard then checks every
number and verdict word it wrote. The template is used when no LLM is available
or the guard rejects the text, so the template itself must always pass the guard.
"""
from __future__ import annotations

from typing import Optional

import pycountry

from auditor.indicators import label
from auditor.schemas import Fact, Status, Verdict
from trust_layer.units import fmt_number

NARRATE_PROMPT = """You explain an environmental claim audit result in at most 2 sentences.
Use ONLY numbers that appear in the JSON. Do not change the status: name it with its exact word
(Supported, Contradicted or Unsupported) and never use the other two words. No speculation."""

# World Bank aggregate codes + short names where pycountry's official form reads badly.
_ENTITY_NAMES = {
    "WLD": "World", "EUU": "European Union", "EMU": "Euro area", "OED": "OECD members",
    "HIC": "High income", "LMC": "Lower middle income", "UMC": "Upper middle income", "LIC": "Low income",
    "USA": "United States", "GBR": "United Kingdom", "RUS": "Russia", "KOR": "South Korea",
    "PRK": "North Korea", "COD": "DR Congo", "CIV": "Côte d'Ivoire",
}


def entity_name(code: Optional[str]) -> str:
    """'DEU' -> 'Germany', 'WLD' -> 'World', None -> 'the reporting organisation'."""
    if not code:
        return "the reporting organisation"
    code = code.strip().upper()
    if code in _ENTITY_NAMES:
        return _ENTITY_NAMES[code]
    country = pycountry.countries.get(alpha_3=code)
    if country is None:
        return code
    return getattr(country, "common_name", None) or country.name


def _sentence(text: Optional[str]) -> str:
    text = (text or "").strip()
    return text if not text or text[-1] in ".!?" else text + "."


def citation(facts: list[Fact]) -> str:
    """'World Bank WDI, EN.GHG.CO2.MT.CE.AR5 · DEU · 2020 (https://…)'; one entry per distinct source row."""
    seen: list[str] = []
    for f in facts:
        part = f.source + (f", {f.locator}" if f.locator else "") + (f" ({f.url})" if f.url else "")
        if part not in seen:
            seen.append(part)
    return "; ".join(seen)


def template_explanation(v: Verdict) -> str:
    c = v.claim
    what = label(c.indicator) if c.indicator else "This claim"
    if c.entity:
        what += f" for {entity_name(c.entity)}"
    cite = citation(v.evidence)
    if "guard_failed" in v.flags:
        return f"{what}: the cited quote could not be re-found in its source ({cite}), so the claim cannot be verified."
    if v.conflict:
        return f"{what}: sources disagree, so the claim cannot be verified. {_sentence(v.calculation)}"
    if v.status == Status.UNSUPPORTED:
        text = f"{what}: not enough evidence to verify this claim. {_sentence(v.calculation)}".strip()
        return f"{text} Checked: {cite}." if cite else text
    unit = c.unit or ""
    claimed = f"{fmt_number(c.value)}{'' if unit in ('%', '') else ' '}{unit}"
    verb = "matches" if v.status == Status.SUPPORTED else "does not match"
    return f"{what}: the claim ({claimed}) {verb} the evidence. {_sentence(v.calculation)} Source: {cite}."


def narrate(v: Verdict, llm=None) -> str:
    if llm is None:
        return template_explanation(v)
    payload = v.model_dump_json(include={
        "status": True, "computed_value": True, "calculation": True, "claim": True,
        "evidence": {"__all__": {"source", "value", "unit", "year", "entity", "url"}},
    })
    text = llm.complete(NARRATE_PROMPT, payload)
    return text.strip() if text else template_explanation(v)
