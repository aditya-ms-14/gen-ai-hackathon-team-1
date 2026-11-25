"""Claim Parser: report text / pages -> list[Claim]. Owner: Sibi.

LLM first (one strict-JSON call per page; every returned claim must be found verbatim on the
page and its numbers/years in its own text, otherwise it is dropped or re-parsed by regex).
The regex parser is deterministic, always available, and covers anything the LLM misses.

Entry points
    parse_pages(pages, report_year=None, llm=None)   {page_no: text} -> claims with Claim.page
    parse_report(text, report_year=None, llm=None)   one block of text (thin wrapper, page=None)
    parse_claim_regex(claim_id, text, report_year)   one sentence, no LLM

What the regex parser understands
    Text                                                       metric      value  unit        years / entity
    ---------------------------------------------------------  ----------  -----  ----------  ---------------------------
    "Carbon emissions reduced by 40%"                           pct_change  40     %  (down)   base = target-1, target = report year
    "Germany's CO2 emissions fell by 30% between 2010 and 2022" pct_change  30     %  (down)   2010 -> 2022, entity DEU
    "Renewable energy usage reached 75%"  (share indicator)     share       75     %           target = report year
    "Renewable electricity rose from 40% in 2015 to 55% in 2022" share      55     %  (up)     2015 -> 2022
    "... rose by 12 percentage points" / "12 pp" / "12 p.p."    pct_change  12     pp
    "We halved water use" / "by half"                           pct_change  50     %  (down)
    "doubled" / "tripled"                                       pct_change  100/200 % (up)
    "cut by a third" / "a quarter" / "two-thirds"               pct_change  33.3 / 25 / 66.7
    "since 2010"                                                            -      -           base 2010, target = report year
    "compared to 2019 levels" / "relative to 2015" / "vs 2019" / "than in 2019" / "2019 baseline"   base year
    "by 2030" (future target)                                               -      -           target 2030 (Unsupported later)
    "12,500 kL" / "1,200 tCO2e"                                 absolute    12500  kL / tCO2e
    "1.2 million tonnes"                                        absolute    1.2e6  tonnes
    "640 Mt CO2e" / "1.2 million tonnes of CO2e"                absolute    640    Mt CO2e
    "40 billion m3" / "40 billion cubic metres"                 absolute    40     billion m3
    "350 GWh", "2.1 TWh", "12 ML", "120 hectares"               absolute    as written
    "from 1,000 tCO2e to 700 tCO2e"                             absolute    700    tCO2e
    "reduced by 500 MWh" / "saved 500 MWh" (absolute change)    absolute    None   MWh         schema has no absolute-change metric
    "0.5 tCO2e per tonne" (intensity)                           not an amount
    "45% of our water is recycled" (non-share indicator)        share       45     %           indicator dropped: not that indicator
    "eco-friendly", "carbon neutral", "net zero", ...           vague       None
    Entity: first country / "EU" / "global" mentioned (auditor.entities); "our global ..." stays the organisation (None).

Page handling (parse_pages): lines are re-flowed into paragraphs (PDF line wraps and hyphenation
joined), split into sentences; headings, tables of contents, running titles and page numbers are
skipped; only checkable sentences (indicator + number) or vague/greenwash sentences are kept;
repeated sentences (summary boxes) are de-duplicated. "C01 ..." lines keep their ids.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from auditor.entities import country_code, find_country
from auditor.indicators import INDICATORS, is_share, match_indicator
from auditor.schemas import Claim, Direction, Metric

# ---- vocabulary (VAGUE_TERMS / has_vague_term are used by scoring.py) ------------
VAGUE_TERMS = ["eco-friendly", "environmentally friendly", "green", "sustainable", "carbon neutral",
               "net zero", "net-zero", "climate positive", "planet friendly", "clean", "responsible"]
_VAGUE = re.compile(r"\b(" + "|".join(re.escape(t) for t in VAGUE_TERMS) + r")\b", re.IGNORECASE)


def has_vague_term(text: str) -> bool:
    """Whole words only, so 'greenhouse' is not 'green'."""
    return bool(_VAGUE.search(text))


_CLAIM_ID = re.compile(r"^\s*(C\d{1,3})\b[\s:.\-–)]*", re.IGNORECASE)
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_NUM_RE = re.compile(r"(?<![\w.,])" + _NUM + r"(?!\d)")
_PP = re.compile(r"(?<![\w.,])" + _NUM + r"\s*(?:percentage[\s-]points?\b|percent(?:age)?\s+points?\b|"
                 r"per\s?cent\s+points?\b|%\s*points?\b|pp\b|p\.p\.|ppts?\b)", re.IGNORECASE)
_PCT = re.compile(r"(?<![\w.,])" + _NUM + r"\s*(?:%|percent\b|per\s?cent\b)(?!\s*points?\b)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<![\w.,$€£])(?:FY\s?)?((?:19|20)\d{2})(?![\w%])", re.IGNORECASE)
_Y = r"(?:FY\s?)?((?:19|20)\d{2})\b"
_BETWEEN = re.compile(r"\b(?:between|from)\s+" + _Y + r"\s*(?:and|to|until|through|-|–|—)\s*" + _Y, re.IGNORECASE)
_RANGE = re.compile(r"(?<![\w.,])((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\b")
_SINCE = re.compile(r"\bsince\s+(?:the\s+)?" + _Y, re.IGNORECASE)
_BASELINE = re.compile(r"\b(?:compared\s+(?:to|with)|relative\s+to|versus|vs\.?|against|from|below|above|on|over|"
                       r"than(?:\s+in)?|baseline(?:\s+year)?(?:\s+of)?)\s+(?:the\s+|a\s+|its\s+|our\s+|their\s+)?"
                       + _Y, re.IGNORECASE)
_BASELINE_AFTER = re.compile(r"\b((?:19|20)\d{2})\s+(?:levels?|baseline|base[\s-]year|values?)\b", re.IGNORECASE)

_QUALIFIER = r"(?:(?:about|around|approximately|approx\.?|nearly|almost|over|more\s+than|some|roughly|" \
             r"just\s+(?:over|under)|up\s+to|at\s+least)\s+)?"
_CHANGE_BEFORE = re.compile(
    r"(?:\bby|\b(?:increas|decreas|reduc|grew|grow|rose|rise|rising|fell|fall|drop|declin|cut|lower|higher|improv|"
    r"boost|jump|climb|slash|shr[aiu]nk|plung|surg|soar)\w*(?:\s+by)?|(?<!made\s)(?<!make\s)(?<!makes\s)\bup|\bdown|"
    r"\b(?:increase|decrease|reduction|rise|growth|drop|decline|cut|fall|improvement|change)s?\s+of)\s+"
    + _QUALIFIER + r"$", re.IGNORECASE)
_CHANGE_AFTER = re.compile(r"\s*(?:increase|decrease|reduction|rise|growth|drop|decline|cut|fall|improvement|lower|"
                           r"higher|less|more|fewer|below|above|down|up|year[\s-]on[\s-]year|change)\b", re.IGNORECASE)
_ABS_CHANGE_BEFORE = re.compile(
    r"(?:\bby|\b(?:sav|avoid|eliminat|divert)\w*|\b(?:reduc|decreas|cut|fell|drop|lower|increas|grew|rose)\w*|"
    r"\b(?:reduction|decrease|increase|cut|drop|fall|rise|saving)s?\s+of)\s+" + _QUALIFIER + r"$", re.IGNORECASE)
_SHARE_OF = re.compile(r"\s*of\b", re.IGNORECASE)
_FROM_TO = re.compile(r"\bfrom\s+" + _QUALIFIER + r"(?P<a>" + _NUM + r")(?:(?!\bfrom\b)[^;:]){0,60}?\bto\s+"
                      + _QUALIFIER + r"(?P<b>" + _NUM + r")", re.IGNORECASE)

_DECREASE = re.compile(r"\b(?:reduc\w*|decreas\w*|cut\w*|lower\w*|fell|fall\w*|drop\w*|declin\w*|less|fewer|"
                       r"sav\w*|down|halv\w*|below|shr[aiu]nk\w*|slash\w*|eliminat\w*|avoid\w*|minimi[sz]\w*)\b",
                       re.IGNORECASE)
_INCREASE = re.compile(r"\b(?:increas\w*|grew|grow\w*|rose|ris(?:e|es|en|ing)|improv\w*|boost\w*|more|higher|"
                       r"doubl\w*|tripl\w*|expand\w*|gain\w*|climb\w*|jump\w*|surg\w*|above|"
                       r"(?<!made\s)(?<!make\s)(?<!makes\s)up(?!\s+to\b))\b", re.IGNORECASE)

# (pattern, value, fixed direction or None = read it from the verb)
_MULTIPLIERS = [(re.compile(p, re.IGNORECASE), v, d) for p, v, d in [
    (r"\bhalv(?:ed|ing|es)\b|\bin\s+half\b", 50.0, Direction.DECREASE),
    (r"\bby\s+(?:a\s+|one\s+)?half\b", 50.0, None),
    (r"\bby\s+two[\s-]thirds\b", 66.7, None),
    (r"\bby\s+(?:a|one)[\s-]third\b", 33.3, None),
    (r"\bby\s+three[\s-]quarters\b", 75.0, None),
    (r"\bby\s+(?:a|one)[\s-]quarter\b", 25.0, None),
    (r"\bdoubl(?:ed|ing|es)\b", 100.0, Direction.INCREASE),
    (r"\btripl(?:ed|ing|es)\b", 200.0, Direction.INCREASE),
    (r"\bquadrupl(?:ed|ing|es)\b", 300.0, Direction.INCREASE),
]]

# ---- absolute units ---------------------------------------------------------------
_CO2E = r"(?:co2|co₂|carbon\s+dioxide)[\s-]?(?:e\b|eq\b\.?|equivalents?\b)"
_CO2 = r"(?:co2|co₂|carbon\s+dioxide)\b"
_TONNE = r"(?:t|tonnes?|tons?|metric\s+ton(?:ne)?s?)"
_OF = r"\s*(?:of\s+)?"
_UNIT_TABLE = [(re.compile(r"\s*" + p, re.IGNORECASE), canon) for p, canon in [
    (r"(?:g|giga)\s?" + _TONNE + _OF + _CO2E, "Gt CO2e"),
    (r"(?:m|mega)\s?" + _TONNE + _OF + _CO2E, "Mt CO2e"),
    (r"(?:k|kilo)\s?" + _TONNE + _OF + _CO2E, "kt CO2e"),
    (_TONNE + _OF + _CO2E, "tCO2e"),
    (r"(?:g|giga)\s?" + _TONNE + _OF + _CO2, "Gt CO2"),
    (r"(?:m|mega)\s?" + _TONNE + _OF + _CO2, "Mt CO2"),
    (r"(?:k|kilo)\s?" + _TONNE + _OF + _CO2, "kt CO2"),
    (_TONNE + _OF + _CO2, "tCO2"),
    (r"(?:gt|gigatonnes?)\b", "Gt"),
    (r"(?:mt|megatonnes?)\b", "Mt"),
    (r"(?:kt|kilotonnes?)\b", "kt"),
    (_TONNE + r"\b", "tonnes"),
    (r"(?:kl|kilolit(?:re|er)s?)\b", "kL"),
    (r"(?:ml|megalit(?:re|er)s?)\b", "ML"),
    (r"(?:gl|gigalit(?:re|er)s?)\b", "GL"),
    (r"(?:m3\b|m³|cubic\s+met(?:re|er)s?\b)", "m3"),
    (r"lit(?:re|er)s?\b", "L"),
    (r"kwh\b", "kWh"), (r"mwh\b", "MWh"), (r"gwh\b", "GWh"), (r"twh\b", "TWh"),
    (r"gj\b", "GJ"), (r"tj\b", "TJ"), (r"pj\b", "PJ"),
    (r"(?:hectares?|ha)\b", "ha"), (r"(?:km2|km²|square\s+kilomet(?:re|er)s?\b)", "km2"),
]]
_SCALE = re.compile(r"\s*(thousand|million|billion|trillion|mn|bn)\b", re.IGNORECASE)
_SCALE_NAME = {"mn": "million", "bn": "billion"}
_SCALE_FACTOR = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# scale word folded into the unit instead of multiplied out (units the verifier knows)
_SCALED_UNIT = {("thousand", "tCO2e"): "kt CO2e", ("million", "tCO2e"): "Mt CO2e", ("billion", "tCO2e"): "Gt CO2e",
                ("thousand", "tCO2"): "kt CO2", ("million", "tCO2"): "Mt CO2", ("billion", "tCO2"): "Gt CO2",
                ("billion", "m3"): "billion m3"}
_UNIT_FACTOR = {"Mt": (1e6, "tonnes"), "Gt": (1e9, "tonnes")}  # plain mass in tonnes
_INTENSITY = re.compile(r"\s*(?:per\b|/\s*[a-z€$£])", re.IGNORECASE)

# ---- page layout --------------------------------------------------------------------
_PAGE_NO = re.compile(r"^(?:page\s*)?[-–—|\s]*\d{1,4}[-–—|\s]*(?:(?:/|of)\s*\d{1,4})?$|^[ivxlcdm]{1,7}$", re.IGNORECASE)
_TOC = re.compile(r"(?:\.{3,}|…+|(?:\s?\.\s?){4,}|_{3,})\s*\d{1,4}$")
_TOC_TITLE = re.compile(r"^(?:table\s+of\s+)?contents$|^index$", re.IGNORECASE)
_SYMBOL_BULLET = re.compile(r"^[•●▪◦‣∙*–—-]\s+")
_NUMBER_BULLET = re.compile(r"^\(?(?:\d{1,2}|[a-z])[.)]\s+")
_ENDS_SENTENCE = re.compile(r"[.!?;][\"”’)\]]*$")
_SENT_END = re.compile(r"[.!?]+[\"”’)\]]*\s+(?=[\"“(\[]?[A-Z0-9•])")
_ABBREV = {"e.g", "i.e", "vs", "approx", "no", "nos", "fig", "figs", "inc", "ltd", "co", "corp", "plc", "dr", "mr",
           "mrs", "ms", "prof", "st", "p.a", "ca", "cf", "al", "incl", "excl", "est", "jan",
           "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_ORG_VOICE = re.compile(r"\b(?:we|our|ours|the company|the group)\b", re.IGNORECASE)
_ORG_WORDS = {"", "null", "none", "n/a", "org", "organisation", "organization", "company", "the company",
              "reporting organisation", "reporting organization", "self"}
MAX_LLM_CHARS = 8000

PAGE_PROMPT = f"""You extract environmental claims from one page of a sustainability report.
Return {{"claims": [...]}} with one item per claim:
  text: the claim sentence copied EXACTLY from the page (verbatim, without any "C01" id)
  indicator: one of {sorted(INDICATORS)} or null
  metric: "pct_change" (a change: "fell by 30%"), "share" (a level in %: "reached 75%"),
          "absolute" (an amount with a unit: "700 tCO2e"), or "vague" (no number: "eco-friendly")
  direction: "decrease" | "increase" | "none"
  value: the number as written (halved = 50, doubled = 100, by a third = 33.3), or null
  unit: "%" for percent, "pp" for percentage points, otherwise the unit as written
        ("tCO2e", "Mt CO2e", "kL", "billion m3", "MWh"), or null
  baseline_year, target_year: the years the claim compares or refers to (int), or null
  entity: ISO3 code of the country the claim is about ("DEU"), "WLD" for the world, "EUU" for the EU,
          null when it is about the reporting company
Include every sentence that states an environmental number, and vague green marketing claims.
Skip headings, tables of contents, page numbers and sentences with no environmental claim.
Use only what the page states. Never invent sentences, numbers or years."""

_NULLABLE = lambda t: {"type": [t, "null"]}  # noqa: E731
PAGE_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "indicator": _NULLABLE("string"),
            "metric": {"type": "string", "enum": [m.value for m in Metric]},
            "direction": {"type": "string", "enum": [d.value for d in Direction]},
            "value": _NULLABLE("number"),
            "unit": _NULLABLE("string"),
            "baseline_year": _NULLABLE("integer"),
            "target_year": _NULLABLE("integer"),
            "entity": _NULLABLE("string"),
        },
        "required": ["text", "metric"],
    }}},
    "required": ["claims"],
}


# ---- quantities ---------------------------------------------------------------------
@dataclass
class _Amount:
    value: float
    unit: str
    start: int
    end: int


@dataclass
class _Quantity:
    metric: Metric
    value: Optional[float]
    unit: Optional[str]
    start: int
    end: int
    direction: Optional[Direction] = None
    drop_indicator: bool = False


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _amounts(text: str, intensities: bool = False) -> list[_Amount]:
    """Every 'number [scale] unit' in text, normalised ('1.2 million tonnes' -> 1.2e6 tonnes).

    intensities=True returns only the '0.5 tCO2e per tonne' kind instead.
    """
    out = []
    for m in _NUM_RE.finditer(text):
        pos, scale = m.end(), None
        if sm := _SCALE.match(text, pos):
            scale = _SCALE_NAME.get(sm.group(1).lower(), sm.group(1).lower())
            pos = sm.end()
        for pattern, canon in _UNIT_TABLE:
            um = pattern.match(text, pos)
            if not um:
                continue
            if bool(_INTENSITY.match(text, um.end())) == intensities:  # "0.5 tCO2e per tonne" is not an amount
                value, unit = _num(m.group(1)), canon
                if scale and (scale, unit) in _SCALED_UNIT:
                    unit = _SCALED_UNIT[(scale, unit)]
                elif scale:
                    value *= _SCALE_FACTOR[scale]
                if unit in _UNIT_FACTOR:
                    factor, unit = _UNIT_FACTOR[unit]
                    value *= factor
                out.append(_Amount(round(value, 6), unit, m.start(), um.end()))
            break
    return out


def _is_change(text: str, start: int, end: int) -> bool:
    return bool(_CHANGE_BEFORE.search(text[max(0, start - 60):start]) or _CHANGE_AFTER.match(text, end))


def _multiplier(text: str) -> Optional[_Quantity]:
    for pattern, value, direction in _MULTIPLIERS:
        if m := pattern.search(text):
            return _Quantity(Metric.PCT_CHANGE, value, "%", m.start(), m.end(), direction)
    return None


def _quantity(text: str, indicator: Optional[str]) -> Optional[_Quantity]:
    """The number the claim is about, in priority order: from-to, %/pp, halved/doubled, amounts."""
    pcts, pps, amounts = list(_PCT.finditer(text)), list(_PP.finditer(text)), _amounts(text)

    for m in _FROM_TO.finditer(text):  # "from 40% (in 2015) to 55%" / "from 1,000 to 700 tCO2e"
        b = m.start("b")
        if pct := next((p for p in pcts if p.start() == b), None):
            a_val, b_val = _num(m.group("a")), _num(pct.group(1))
            direction = Direction.INCREASE if b_val > a_val else Direction.DECREASE if b_val < a_val else None
            return _Quantity(Metric.SHARE, b_val, "%", m.start(), pct.end(), direction)
        if amt := next((a for a in amounts if a.start == b), None):
            return _Quantity(Metric.ABSOLUTE, amt.value, amt.unit, m.start(), amt.end)

    ratios = sorted([(p.start(), p, "pp") for p in pps] + [(p.start(), p, "%") for p in pcts], key=lambda r: r[0])
    if ratios:
        _, p, unit = ratios[0]
        value = _num(p.group(1))
        if unit == "pp":
            return _Quantity(Metric.PCT_CHANGE, value, "pp", p.start(), p.end())
        if _is_change(text, p.start(), p.end()):
            return _Quantity(Metric.PCT_CHANGE, value, "%", p.start(), p.end())
        if is_share(indicator):
            return _Quantity(Metric.SHARE, value, "%", p.start(), p.end())
        if _SHARE_OF.match(text, p.end()):  # "45% of our water is recycled": not water consumption itself
            return _Quantity(Metric.SHARE, value, "%", p.start(), p.end(), drop_indicator=True)
        return _Quantity(Metric.PCT_CHANGE, value, "%", p.start(), p.end())

    if mult := _multiplier(text):
        return mult

    if amounts:
        a = amounts[0]
        if _ABS_CHANGE_BEFORE.search(text[max(0, a.start - 60):a.start]):
            # "reduced by 500 MWh" is a change in amount; the schema can't express it, so no value to check
            return _Quantity(Metric.ABSOLUTE, None, a.unit, a.start, a.end)
        return _Quantity(Metric.ABSOLUTE, a.value, a.unit, a.start, a.end)
    if intensity := _amounts(text, intensities=True):
        # a real number, but a ratio no total can confirm: uncheckable, yet not greenwash
        a = intensity[0]
        return _Quantity(Metric.ABSOLUTE, None, f"{a.unit} per unit", a.start, a.end)
    return None


def _has_quantity(text: str) -> bool:
    return bool(_PCT.search(text) or _PP.search(text) or _multiplier(text) or _amounts(text))


def _is_checkable(text: str) -> bool:
    return bool(match_indicator(text) and _has_quantity(text))


def _looks_like_claim(text: str) -> bool:
    return _is_checkable(text) or (has_vague_term(text) and len(text.split()) >= 4)


# ---- direction, years, entity ---------------------------------------------------------
def _direction(text: str, near: Optional[int] = None) -> Direction:
    """Direction word closest to the quantity (preceding words preferred); decrease wins ties."""
    hits = [(m, Direction.DECREASE) for m in _DECREASE.finditer(text)]
    hits += [(m, Direction.INCREASE) for m in _INCREASE.finditer(text)]
    if not hits:
        return Direction.NONE
    if near is None:
        return Direction.DECREASE if any(d == Direction.DECREASE for _, d in hits) else Direction.INCREASE

    def distance(hit):
        m, d = hit
        gap = near - m.end() if m.end() <= near else (m.start() - near) * 1.5 + 5
        return (gap, d != Direction.DECREASE)
    return min(hits, key=distance)[1]


def _years(text: str, skip: list[tuple[int, int]]) -> list[int]:
    out = []
    for m in _YEAR_RE.finditer(text):
        if not any(s <= m.start(1) < e for s, e in skip):
            out.append(int(m.group(1)))
    return out


def _baseline_target(text: str, years: list[int], report_year: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    if m := (_BETWEEN.search(text) or _RANGE.search(text)):
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)
    baseline = None
    for pattern in (_SINCE, _BASELINE, _BASELINE_AFTER):
        if (m := pattern.search(text)) and int(m.group(1)) in years:
            baseline = int(m.group(1))
            break
    others = [y for y in years if y != baseline]
    if baseline is not None:
        later = [y for y in others if y > baseline]
        return baseline, max(later) if later else (report_year if report_year and report_year > baseline else None)
    if len(set(years)) > 1:
        return min(years), max(years)
    return None, years[0] if years else report_year


def extract_entity(text: str) -> Optional[str]:
    """ISO3 / 'WLD' / 'EUU' the sentence is about; None = the reporting organisation."""
    return _org_guard(find_country(text), text)


def _org_guard(code: Optional[str], text: str) -> Optional[str]:
    """'Our global operations' is the organisation, not the world."""
    if code == "WLD" and _ORG_VOICE.search(text) and not re.search(r"\bworld\b", text, re.IGNORECASE):
        return None
    return code


# ---- one sentence -> Claim --------------------------------------------------------------
def parse_claim_regex(claim_id: str, text: str, report_year: Optional[int], page: Optional[int] = None) -> Claim:
    indicator = match_indicator(text)
    q = _quantity(text, indicator)
    skip = [(m.start(), m.end()) for m in list(_PCT.finditer(text)) + list(_PP.finditer(text))]
    skip += [(a.start, a.end) for a in _amounts(text)]
    baseline, target = _baseline_target(text, _years(text, skip), report_year)

    if q is None:
        metric, value, unit, direction = Metric.VAGUE, None, None, _direction(text)
    else:
        metric, value, unit = q.metric, q.value, q.unit
        direction = q.direction or _direction(text, q.start)
        if q.drop_indicator:
            indicator = None
    if metric == Metric.PCT_CHANGE and baseline is None and target is not None:
        baseline = target - 1
    return Claim(claim_id=claim_id, text=text, indicator=indicator, metric=metric, direction=direction,
                 value=value, unit=unit, baseline_year=baseline, target_year=target, page=page,
                 entity=extract_entity(text))


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    m = _NUM_RE.search(str(v))
    return _num(m.group(1)) if m else None


def _to_year(v: Any) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None and 1900 <= f <= 2100 and f == int(f) else None


def _fmt(v: float) -> str:
    return ("%f" % v).rstrip("0").rstrip(".")


def _canon_quantity(value: Optional[float], unit: Any) -> tuple[Optional[float], Optional[str]]:
    """LLM unit strings -> the regex parser's canonical units ('million tonnes' -> tonnes x 1e6)."""
    u = unit.strip() if isinstance(unit, str) else ""
    low = u.lower()
    if low in ("%", "percent", "per cent", "percentage"):
        return value, "%"
    if low in ("pp", "p.p.", "ppt", "pts", "points", "% points", "percentage point", "percentage points"):
        return value, "pp"
    if value is None or not u:
        return value, u or None
    amounts = _amounts(f"{_fmt(value)} {u}")
    if amounts and amounts[0].start == 0:
        return amounts[0].value, amounts[0].unit
    return value, u


def _llm_entity(raw: Any, text: str) -> Optional[str]:
    if isinstance(raw, str) and raw.strip().lower() not in _ORG_WORDS:
        if code := country_code(raw):
            return _org_guard(code, text)
    return extract_entity(text)


def _claim_from_llm(item: dict, text: str, report_year: Optional[int], page: Optional[int]) -> Claim:
    """LLM fields, but only if every number and year is grounded in the text; else the regex parse."""
    fallback = parse_claim_regex("", text, report_year, page)
    entity = _llm_entity(item.get("entity"), text)
    regex_result = fallback.model_copy(update={"entity": entity})
    try:
        metric = Metric(item.get("metric"))
    except ValueError:
        return regex_result
    raw_value = _to_float(item.get("value"))
    if metric == Metric.VAGUE:
        if fallback.value is not None:  # the text has a number the LLM ignored
            return regex_result
        value, unit = None, None
    else:
        if raw_value is None:
            return regex_result
        value, unit = _canon_quantity(raw_value, item.get("unit"))
    try:
        direction = Direction(item.get("direction") or "none")
    except ValueError:
        direction = fallback.direction
    # Rules first: when the wording matches an indicator synonym ("carbon emissions" -> total GHG), that
    # mapping wins. The LLM's own indicator is used only when the synonyms find none. Measured on
    # 2026-11-25: every Groq model read "Carbon emissions reduced by 40%" as CO2 and lost the claim.
    llm_indicator = item.get("indicator") if item.get("indicator") in INDICATORS else None
    indicator = fallback.indicator or llm_indicator
    baseline, target = _to_year(item.get("baseline_year")), _to_year(item.get("target_year"))
    if target is None:
        target = report_year
    if metric == Metric.PCT_CHANGE and baseline is None and target is not None:
        baseline = target - 1

    numbers = [_num(n) for n in _NUM_RE.findall(text)]
    value_ok = (raw_value is None or any(math.isclose(raw_value, n, rel_tol=1e-6) for n in numbers)
                or (value is not None and fallback.value is not None and math.isclose(value, fallback.value, rel_tol=1e-6)))
    text_years = set(_years(text, []))
    years_ok = (target is None or target in text_years or target == report_year) and \
               (baseline is None or baseline in text_years or (target is not None and baseline == target - 1))
    if not (value_ok and years_ok):
        return regex_result
    return Claim(claim_id="", text=text, indicator=indicator, metric=metric, direction=direction, value=value,
                 unit=unit, baseline_year=baseline, target_year=target, page=page, entity=entity)


# ---- page layout -> sentences --------------------------------------------------------
@dataclass
class _Unit:
    text: str
    explicit_id: Optional[str] = None


def detect_report_year(text: str) -> Optional[int]:
    """First year in the first few lines, e.g. 'COMPANY · 2025 REPORT'."""
    head = "\n".join(text.strip().splitlines()[:5])
    years = _years(head, [])
    return years[0] if years else None


def split_sentences(text: str) -> list[str]:
    """Sentence split that survives 'e.g.', 'U.S.', initials and decimals."""
    out, start = [], 0
    for m in _SENT_END.finditer(text):
        before = text[start:m.start()].split()
        word = before[-1].lower().rstrip(".") if before else ""
        if word in _ABBREV or (len(word) == 1 and word.isalpha()):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def _is_junk(line: str) -> bool:
    return bool(_PAGE_NO.match(line) or _TOC.search(line) or _TOC_TITLE.match(line))


# A line ending on one of these words is a wrapped sentence ("Scope 1 emissions fell by" / "12% ..."), never a heading.
_DANGLING = re.compile(
    r"\b(by|of|to|from|in|on|at|for|the|a|an|and|or|than|with|as|was|were|is|are|has|have|"
    r"fell|rose|dropped|declined|grew|reduced|increased|decreased|cut|reached|compared|between|since)$",
    re.IGNORECASE)


def _is_heading(line: str, next_line: str, wrapped: bool = False) -> bool:
    """Short line, no sentence punctuation, followed by a new block (blank / capitalised line).

    `wrapped`: the previous line was a full-width line of an unfinished sentence, so this one continues it.
    """
    if wrapped or _DANGLING.search(line):
        return False
    if _ENDS_SENTENCE.search(line) or len(line.split()) > 12 or len(line) > 90:
        return False
    return not next_line or not next_line[0].islower()


def _join(a: str, b: str) -> str:
    if a.endswith("-") and len(a) > 1 and a[-2].isalpha() and b[:1].islower():
        return a[:-1] + b  # "emis-" + "sions"
    return a + " " + b


def _segment(text: str) -> list[_Unit]:
    """Page text -> sentences (and 'C01 ...' lines), without headings, TOC lines or page numbers."""
    lines = [line.strip() for line in text.splitlines()]
    units: list[_Unit] = []
    block: Optional[list] = None  # [explicit_id, text, started_as_bullet]

    def flush() -> None:
        nonlocal block
        if block and block[1].strip():
            if block[0]:
                units.append(_Unit(block[1].strip(), block[0]))
            else:
                units.extend(_Unit(s) for s in split_sentences(block[1]))
        block = None

    for i, line in enumerate(lines):
        next_line = next((n for n in lines[i + 1:i + 2]), "")
        if not line or _is_junk(line):
            flush()
            continue
        if m := _CLAIM_ID.match(line):
            flush()
            block = [m.group(1).upper(), line[m.end():].strip(), False]
            continue
        if m := _SYMBOL_BULLET.match(line):
            flush()
            block = [None, line[m.end():].strip(), True]
            continue
        if m := _NUMBER_BULLET.match(line):
            content = line[m.end():].strip()
            flush()
            if _is_heading(content, next_line) and not _is_checkable(content):
                continue  # "1. Introduction"
            block = [None, content, True]
            continue
        is_open = block is not None and not _ENDS_SENTENCE.search(block[1])
        if is_open and (line[0].islower() or not line[0].isalpha()):
            block[1] = _join(block[1], line)  # wrapped line
            continue
        wrapped = is_open and i > 0 and len(lines[i - 1]) >= 50  # previous line ran to the margin
        if _is_heading(line, next_line, wrapped) or (is_open and _is_checkable(block[1]) and _is_checkable(line)):
            flush()
            if _is_checkable(line):  # call-out such as "Emissions down 30% since 2019"
                block = [None, line, False]
            continue
        if block is None or (is_open and block[2]):  # a capitalised line after a bullet starts a new block
            flush()
            block = [None, line, False]
        else:
            block[1] = _join(block[1], line)
    flush()
    return units


def split_claims(text: str) -> list[tuple[str, str]]:
    """(claim_id, sentence) pairs. Uses 'C01 ...' ids when present."""
    drafts = [(u.explicit_id, Claim(claim_id="", text=u.text)) for u in _segment(text)
              if u.explicit_id or _looks_like_claim(u.text)]
    return [(c.claim_id, c.text) for c in _assign_ids(drafts)]


# ---- LLM batch extraction ---------------------------------------------------------------
_NORM_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_NORM_MAP)).strip().casefold()


def _windows(lines: list[str], limit: int = MAX_LLM_CHARS) -> list[str]:
    out, cur = [], ""
    for line in lines:
        if cur and len(cur) + len(line) > limit:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    return out + ([cur] if cur else [])


def _llm_page_claims(page: Optional[int], text: str, units: list[_Unit], report_year: Optional[int],
                     llm) -> list[tuple[str, Claim]]:
    """One LLM call per page (per ~8k chars). Only claims whose text is on the page survive."""
    lines = [f"{u.explicit_id} {u.text}" if u.explicit_id else u.text for u in units]
    haystacks = (_norm(text), _norm("\n".join(lines)))
    found: list[tuple[str, Claim]] = []
    for window in _windows(lines):
        try:
            data = llm.complete_json(PAGE_PROMPT, f"Report year: {report_year}\nPage: {page or 1}\n---\n{window}",
                                     schema=PAGE_SCHEMA, name="claims")
        except Exception:
            data = None
        items = data if isinstance(data, list) else (data or {}).get("claims") or (data or {}).get("items") or []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            claim_text = _CLAIM_ID.sub("", item["text"]).strip()
            needle = _norm(claim_text).rstrip(".;, ")
            if len(needle) < 8 or not any(needle in h for h in haystacks):
                continue  # invented or paraphrased: dropped
            found.append((needle, _claim_from_llm(item, claim_text, report_year, page)))
    return found


def _page_claims(page: Optional[int], text: str, report_year: Optional[int], llm) -> list[tuple[Optional[str], Claim]]:
    units = _segment(text)
    llm_claims = _llm_page_claims(page, text, units, report_year, llm) if llm is not None and units else []
    if not llm_claims:
        return [(u.explicit_id, parse_claim_regex("", u.text, report_year, page))
                for u in units if u.explicit_id or _looks_like_claim(u.text)]

    norm_units = [_norm(u.text).rstrip(".;, ") for u in units]
    by_unit: dict[int, list[Claim]] = {}
    loose: list[Claim] = []
    for needle, claim in llm_claims:
        idx = next((i for i, nu in enumerate(norm_units) if nu and (needle in nu or nu in needle)), None)
        if idx is None:
            loose.append(claim)
        else:
            by_unit.setdefault(idx, []).append(claim)

    out: list[tuple[Optional[str], Claim]] = []
    for i, unit in enumerate(units):
        if i in by_unit:
            claims = by_unit[i]
            if len(claims) == 1:
                claims[0].text = unit.text  # keep the whole sentence as written
            out.append((unit.explicit_id, claims[0]))
            out.extend((None, c) for c in claims[1:])
        elif unit.explicit_id or _is_checkable(unit.text):  # never lose a number the LLM skipped
            out.append((unit.explicit_id, parse_claim_regex("", unit.text, report_year, page)))
    return out + [(None, c) for c in loose]


def _assign_ids(drafts: list[tuple[Optional[str], Claim]]) -> list[Claim]:
    """Explicit ids ('C03') kept; the rest numbered C01, C02, ... skipping ids already used."""
    reserved = {wid for wid, _ in drafts if wid}
    taken: set[str] = set()
    n, out = 0, []
    for wid, claim in drafts:
        if wid and wid not in taken:
            cid = wid
        else:
            n += 1
            while f"C{n:02d}" in reserved or f"C{n:02d}" in taken:
                n += 1
            cid = f"C{n:02d}"
        taken.add(cid)
        claim.claim_id = cid
        out.append(claim)
    return out


def _dedup_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _parse(pages: list[tuple[Optional[int], str]], report_year: Optional[int], llm) -> list[Claim]:
    drafts: list[tuple[Optional[str], Claim]] = []
    seen: set[str] = set()
    for page, text in pages:
        for wid, claim in _page_claims(page, text, report_year, llm):
            key = _dedup_key(claim.text)
            if key in seen and not wid:
                continue  # repeated in a summary box / on another page
            seen.add(key)
            drafts.append((wid, claim))
    return _assign_ids(drafts)


# ---- public -----------------------------------------------------------------------
def parse_pages(pages: dict[int, str], report_year: Optional[int] = None, llm=None) -> list[Claim]:
    """Real reports: {page_no: text} -> claims with Claim.page set, in page order."""
    ordered = sorted((p, t) for p, t in pages.items() if t and t.strip())
    if not report_year and ordered:
        report_year = detect_report_year(ordered[0][1])
    return _parse(ordered, report_year, llm)


def parse_report(text: str, report_year: Optional[int] = None, llm=None) -> list[Claim]:
    """One block of text (no page numbers): same parser as parse_pages, Claim.page stays None."""
    report_year = report_year or detect_report_year(text)
    return _parse([(None, text)], report_year, llm)
