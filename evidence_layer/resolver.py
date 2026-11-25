"""Evidence Resolver: ledger first, then official data, then RAG. Owner: Aditya.

1. Ledger lookup by (indicator, year, claim.entity).
2. Ledger miss + claim about a country/aggregate + use_official:
   evidence_layer.sources.fetch_official(indicator, entity, years) -> facts, added to
   the ledger (route "official").
3. Still missing: hybrid search for passages; if an LLM is available, ask it
   for {value, unit, year, quote} per missing year and keep the answer only if
   the quote is verbatim (normalised) in a passage and contains the value
   (Fact.method = LLM). No LLM / None reply / bad quote -> the year stays
   missing and the claim ends Unsupported. Never guess.
"""
from __future__ import annotations

import re
from typing import Optional

from auditor.indicators import label
from evidence_layer.ingest import norm
from evidence_layer.ledger import Ledger
from evidence_layer.retrieval import HybridIndex
from auditor.schemas import Chunk, Claim, Evidence, ExtractionMethod, Fact, Metric
from trust_layer.units import is_pp

LLM_PASSAGES = 3
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")

EXTRACT_PROMPT = """You read evidence passages from an environmental report and copy out one number.
Return {"value": number, "unit": string, "year": integer, "quote": string}.
- value: the {indicator} figure for {year}, exactly as the passage states it (no conversion, no arithmetic).
- quote: the shortest exact span copied character for character from ONE passage that contains the value.
- If no passage states it for that year, return {"value": null, "unit": null, "year": null, "quote": null}.
Never estimate or compute."""


def fill_years(claim: Claim, ledger: Ledger) -> Claim:
    """Default target = latest ledger year, baseline = the year before it."""
    if claim.indicator is None or claim.metric == Metric.VAGUE:
        return claim
    target = claim.target_year
    if target is None:
        known = ledger.years(claim.indicator, claim.entity)
        target = known[-1] if known else None
    baseline = claim.baseline_year
    if _is_change(claim) and baseline is None and target is not None:
        baseline = target - 1
    return claim.model_copy(update={"target_year": target, "baseline_year": baseline})


def _is_change(claim: Claim) -> bool:
    """% change, or a level change in percentage points: both need baseline + target."""
    return claim.metric == Metric.PCT_CHANGE or is_pp(claim.unit)


def needed_years(claim: Claim) -> list[int]:
    if _is_change(claim):
        return [y for y in (claim.baseline_year, claim.target_year) if y is not None]
    return [claim.target_year] if claim.target_year is not None else []


def resolve(claim: Claim, ledger: Ledger, index: HybridIndex, llm: Optional[object] = None,
            use_official: bool = True) -> Evidence:
    if claim.indicator is None or claim.metric == Metric.VAGUE:
        return Evidence(claim_id=claim.claim_id, route="none")

    years = needed_years(claim)

    def lookup() -> tuple[list[Fact], list[int]]:
        found = [f for year in years for f in ledger.lookup(claim.indicator, year, claim.entity)]
        covered = {f.year for f in found}
        return found, [y for y in years if y not in covered]

    facts, missing = lookup()
    if years and not missing:
        return Evidence(claim_id=claim.claim_id, facts=facts, route="ledger")

    official = False
    if claim.entity and use_official and missing:
        fetched = fetch_official(claim.indicator, claim.entity, missing)
        if fetched:
            ledger.add(fetched)
            facts, missing = lookup()
            official = True
            if not missing:
                return Evidence(claim_id=claim.claim_id, facts=facts, route="official")

    query = " ".join([claim.text, label(claim.indicator), *map(str, missing)])
    passages = index.search(query, claim.indicator, missing or None)
    if llm is not None and passages:
        for year in missing:
            fact = extract_with_llm(claim, year, passages[:LLM_PASSAGES], llm)
            if fact is not None:
                facts.append(fact)
    route = "rag" if passages else "official" if official else "none"
    return Evidence(claim_id=claim.claim_id, facts=facts, passages=passages, route=route)


# ---- official data -----------------------------------------------------------

def fetch_official(indicator: str, entity: str, years: list[int]) -> list[Fact]:
    """evidence_layer.sources.fetch_official, if that package exists; only facts that answer this exact question."""
    try:
        from evidence_layer.sources import fetch_official as fetch
    except ImportError:
        return []
    try:
        facts = fetch(indicator, entity, list(years)) or []
    except Exception:  # API down, timeout, bad payload: the claim just stays unresolved
        return []
    return [f for f in facts
            if isinstance(f, Fact) and f.indicator == indicator and f.entity == entity and f.year in years]


# ---- LLM reading a passage ------------------------------------------------------

def _to_float(x) -> Optional[float]:
    if isinstance(x, bool) or x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def _to_year(x) -> Optional[int]:
    v = _to_float(x)
    return int(v) if v is not None and v == int(v) else None


def _quote_has_value(quote: str, value: float) -> bool:
    return any(abs(float(n.replace(",", "")) - value) <= 1e-9 * max(1.0, abs(value)) for n in _NUM.findall(quote))


def accept_llm_answer(data: Optional[dict], claim: Claim, year: int, passages: list[Chunk]) -> Optional[Fact]:
    """Turn an LLM answer into a Fact only if every check passes; None otherwise."""
    if not isinstance(data, dict):
        return None
    value, quote = _to_float(data.get("value")), data.get("quote")
    unit = data.get("unit") if isinstance(data.get("unit"), str) else ""
    if value is None or not isinstance(quote, str) or not quote.strip():
        return None
    if _to_year(data.get("year")) != year or not _quote_has_value(quote, value):
        return None
    if unit and unit.lower().replace(" ", "") not in quote.lower().replace(" ", ""):
        return None  # the unit must be read from the quote too, or the verifier would compare wrong units
    q = norm(quote)
    for chunk in passages:
        text = norm(chunk.text)
        if q in text and str(year) in text:
            return Fact(
                fact_id=f"llm:{chunk.chunk_id}:{claim.indicator}:{year}", indicator=claim.indicator,
                value=value, unit=unit.strip(), year=year, source=chunk.source, page=chunk.page,
                locator=f"{chunk.chunk_id} (read by LLM)", quote=re.sub(r"\s+", " ", quote).strip(),
                method=ExtractionMethod.LLM, entity=claim.entity,
            )
    return None


def extract_with_llm(claim: Claim, year: int, passages: list[Chunk], llm) -> Optional[Fact]:
    if llm is None or not passages:
        return None
    system = EXTRACT_PROMPT.replace("{indicator}", label(claim.indicator)).replace("{year}", str(year))
    numbered = "\n\n".join(f"[Passage {i}] ({c.source}, page {c.page})\n{c.text}" for i, c in enumerate(passages, 1))
    user = f"Claim: {claim.text}\nIndicator: {label(claim.indicator)}\nYear: {year}\n\n{numbered}"
    try:
        data = llm.complete_json(system, user)
    except Exception:
        return None
    return accept_llm_answer(data, claim, year, passages)
