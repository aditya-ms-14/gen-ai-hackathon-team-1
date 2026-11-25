"""Shared contracts for the whole pipeline.

Both laptops code against these models. Change a field only after telling
the other side; adding an optional field is safe, renaming one is not.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Metric(str, Enum):
    PCT_CHANGE = "pct_change"  # "reduced by 40%" (baseline year -> target year)
    SHARE = "share"            # "renewables reached 75%" (a level, not a change)
    ABSOLUTE = "absolute"      # "emitted 700 tCO2e in 2025"
    VAGUE = "vague"            # "eco-friendly", no number to check


class Direction(str, Enum):
    DECREASE = "decrease"
    INCREASE = "increase"
    NONE = "none"


class Status(str, Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    UNSUPPORTED = "Unsupported"  # not enough evidence; NOT the same as false


class ExtractionMethod(str, Enum):
    TABLE = "table"  # CSV / XLSX cell or PDF table cell
    TEXT = "text"    # regex over prose
    LLM = "llm"      # LLM read a retrieved passage (lowest trust)


class SourceType(str, Enum):
    OFFICIAL = "official"  # government / regulator / third-party audited
    COMPANY = "company"    # the company's own data
    UNKNOWN = "unknown"


class Fact(BaseModel):
    """One number in the Evidence Ledger, with enough provenance to re-check it."""

    fact_id: str
    indicator: str                  # canonical id from indicators.py, e.g. "ghg_emissions"
    value: float
    unit: str                       # as written in the source: "tCO2e", "kL", "%"
    year: int
    source: str                     # file name, or dataset name for official APIs ("World Bank WDI")
    page: Optional[int] = None      # 1-based PDF page
    locator: Optional[str] = None   # "row 2", "table 1, row 4", "EN.GHG.CO2.MT.CE.AR5 · DEU · 2020"
    quote: str                      # exact source text; Citation Guard re-finds it
    method: ExtractionMethod = ExtractionMethod.TABLE
    source_type: SourceType = SourceType.UNKNOWN
    entity: Optional[str] = None    # who the number is about: ISO3 country ("DEU"), "WLD", "EUU"; None = the reporting org
    url: Optional[str] = None       # clickable citation link (official datasets)


class Chunk(BaseModel):
    """A page-tagged passage in the Hybrid Index."""

    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None
    indicators: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)


class Claim(BaseModel):
    """One claim from the draft report, as a checkable spec."""

    claim_id: str                        # "C01"
    text: str                            # claim as written
    indicator: Optional[str] = None
    metric: Metric = Metric.VAGUE
    direction: Direction = Direction.NONE
    value: Optional[float] = None        # 40 for "40%"
    unit: Optional[str] = None           # "%" for changes/shares, else the absolute unit
    baseline_year: Optional[int] = None
    target_year: Optional[int] = None
    page: Optional[int] = None
    entity: Optional[str] = None         # same codes as Fact.entity; None = the reporting org


class Evidence(BaseModel):
    """What the Evidence Resolver found for one claim."""

    claim_id: str
    facts: list[Fact] = Field(default_factory=list)
    passages: list[Chunk] = Field(default_factory=list)
    route: str = "none"  # "ledger" | "rag" | "none"


class Verdict(BaseModel):
    """Final result for one claim: what the dashboard and report show."""

    claim: Claim
    status: Status
    computed_value: Optional[float] = None
    calculation: Optional[str] = None    # "(700 - 1000) / 1000 × 100 = -30.0%"
    evidence: list[Fact] = Field(default_factory=list)
    conflict: bool = False               # two sources disagree for the same indicator + year
    confidence: float = 0.0              # 0..1 from evidence quality, never similarity
    flags: list[str] = Field(default_factory=list)  # "greenwash", "conflict", "guard_failed", ...
    explanation: str = ""
    route: Optional[str] = None          # where evidence came from: "ledger" | "rag" | "official" | "none"
