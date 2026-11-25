"""Evidence Ingestion. Owner: Aditya.

Turns evidence files into Evidence Ledger facts and Hybrid Index chunks,
keeping file + page + row for every value.

- CSV / XLSX tables in "wide" (indicator, unit, 2024, 2025...) or "long"
  (indicator, year, value, unit) layout. Optional "entity" / "country" column
  -> ISO3 code (auditor.entities.country_code); blank or unknown = the reporting org.
- PDF tables via pdfplumber (ruled tables first, text-aligned tables as a
  fallback), same wide/long logic, quote = the row's cells.
- Prose facts via regex ("Total GHG emissions in 2025 were 700 tCO2e."), only
  when the sentence names one indicator, one year and one number with a unit.
- PDF/TXT page text -> chunks.

Every fact's quote is checked the way the Citation Guard checks it
(whitespace-collapsed, lower-cased substring of read_pages(source)[page]);
a fact whose quote can't be re-found is dropped, never emitted.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .entities import country_code, find_country
from .indicators import INDICATORS, is_share, match_indicator
from .schemas import Chunk, ExtractionMethod, Fact, SourceType

TABLE_EXTS = {".csv", ".xlsx", ".xls"}
TEXT_EXTS = {".pdf", ".txt", ".md"}
_YEAR = re.compile(r"^(?:FY\s?)?((?:19|20)\d{2})(?:\.0)?\*?$", re.IGNORECASE)  # header / cell that is a year
_YEARS_IN_TEXT = re.compile(r"\b(?:19|20)\d{2}\b")

# One number followed by a unit, e.g. "700 tCO2e", "40,000 kL", "75%".
_UNIT_WORDS = ("ktCO2e|MtCO2e|kgCO2e|tCO2e|tCO2|t CO2e|tonnes CO2e|tonnes|tonne|tons|kt|Mt|t|"
               "kL|ML|m3|m³|megalitres|MWh|GWh|kWh|GJ|TJ|hectares|ha")
_QTY = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
                  r"(%|percent\b|(?:" + _UNIT_WORDS + r")(?![\w₂]))", re.IGNORECASE)
# Sentences that state a change, a target, a bound or an intensity are not levels: skip them.
_NOT_A_LEVEL = re.compile(
    r"\b(?:by\s+\d|(?:up|down)\s+\d|reduc\w*|decreas\w*|increas\w*|fell|rose|cut\w*|drop\w*|declin\w*|"
    r"grew|grow\w*|sav\w*|avoid\w*|compared|versus|vs|target\w*|goal\w*|aim\w*|plan(?:s|ned|ning)?|will|expect\w*|"
    r"forecast\w*|project\w*|estimat\w*|intensity|per|less than|more than|over|under|below|above|up to|"
    r"at least|at most|nearly|almost|approximately|around|about|roughly)\b|~", re.IGNORECASE)
_MAX_SENTENCE = 300


def ingest_file(path: str | Path, source_type: SourceType = SourceType.UNKNOWN) -> tuple[list[Fact], list[Chunk]]:
    """One evidence file -> (ledger facts, index chunks)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in TABLE_EXTS:
        return _table_facts(p, source_type), []
    if ext in TEXT_EXTS:
        pages = read_pages(p)
        facts = _pdf_table_facts(p, pages, source_type) if ext == ".pdf" else []
        facts += _dedupe(facts, _prose_facts(p.name, pages, source_type))
        return facts, chunk_pages(p, pages)
    raise ValueError(f"Unsupported evidence file type: {p.name}")


def read_pages(path: str | Path) -> dict[int, str]:
    """Page number (1-based) -> text. Tables become one canonical CSV 'page'."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        import pymupdf

        with pymupdf.open(p) as doc:
            return {i + 1: page.get_text() for i, page in enumerate(doc)}
    if ext in TABLE_EXTS:
        return {1: _read_table(p).to_csv(index=False)}
    return {1: p.read_text(encoding="utf-8")}


def norm(text: str) -> str:
    """The Citation Guard's normalisation (guard._norm): NFKC, straight quotes, collapse whitespace, lower-case."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


# ---- tables (CSV / XLSX / PDF) ---------------------------------------------

def _read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(p, dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _number(cell: str) -> Optional[float]:
    cleaned = str(cell).replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _year(cell: str) -> Optional[int]:
    m = _YEAR.match(str(cell or "").strip())
    return int(m.group(1)) if m else None


def _source_type(cell: str, default: SourceType) -> SourceType:
    try:
        return SourceType(str(cell).strip().lower()) if str(cell).strip() else default
    except ValueError:
        return default


_UNIT_ONLY = re.compile(r"(?:%|" + _UNIT_WORDS + r")", re.IGNORECASE)


def _unit_in_label(name: str) -> str:
    """'GHG emissions (tCO2e)' -> 'tCO2e' when the table has no unit column."""
    m = re.search(r"\(([^()]+)\)\s*$", name)
    return m.group(1).strip() if m and _UNIT_ONLY.fullmatch(m.group(1).strip()) else ""


def _is_header(row: list[str]) -> bool:
    cells = {str(c or "").strip().lower() for c in row}
    return any(_year(c) for c in row) or {"year", "value"} <= cells


def _rows_to_facts(header: list[str], rows: list[list[str]], *, source: str, page: Optional[int],
                   quote_for: Callable[[int, list[str], int], Optional[str]],
                   locator_for: Callable[[int], str], id_for: Callable[[int, int], str],
                   default_source_type: SourceType, method: ExtractionMethod) -> list[Fact]:
    """Wide or long table rows -> facts. quote_for(row_no, cells, value_col) may veto a fact by returning None."""
    header = [str(h or "").strip() for h in header]
    cols = {h.lower(): j for j, h in enumerate(header) if h}
    year_cols = [(j, y) for j, h in enumerate(header) if (y := _year(h))]
    long = not year_cols and "year" in cols and "value" in cols
    if not year_cols and not long:
        return []
    ind_j = cols.get("indicator", cols.get("metric", 0))
    unit_j, st_j = cols.get("unit"), cols.get("source_type")
    ent_j = cols.get("entity", cols.get("country"))
    facts: list[Fact] = []

    for i, raw in enumerate(rows):
        row = [str(c if c is not None else "").strip() for c in raw] + [""] * max(0, len(header) - len(raw))
        indicator = match_indicator(row[ind_j])
        if not indicator:
            continue
        unit = row[unit_j] if unit_j is not None else _unit_in_label(row[ind_j])
        entity = country_code(row[ent_j]) if ent_j is not None and row[ent_j] else None
        source_type = _source_type(row[st_j], default_source_type) if st_j is not None else default_source_type
        if long:
            cells = [(cols["value"], y)] if (y := _year(row[cols["year"]])) else []
        else:
            cells = year_cols
        for j, year in cells:
            value = _number(row[j])
            if value is None:
                continue  # blank cell = no data, which makes the claim Unsupported
            quote = quote_for(i, row, j)
            if not quote:
                continue
            facts.append(Fact(
                fact_id=id_for(i, year), indicator=indicator, value=value,
                unit=unit or ("%" if "%" in row[j] else ""), year=year, source=source, page=page,
                locator=locator_for(i), quote=quote, method=method, source_type=source_type, entity=entity,
            ))
    return facts


def _table_facts(p: Path, default_source_type: SourceType) -> list[Fact]:
    df = _read_table(p)
    lines = df.to_csv(index=False).splitlines()  # line i+1 is row i: used as the quote
    return _rows_to_facts(
        list(df.columns), df.values.tolist(), source=p.name, page=1,
        quote_for=lambda i, row, j: lines[i + 1], locator_for=lambda i: f"row {i + 2}",
        id_for=lambda i, year: f"{p.name}#r{i + 2}-{year}",
        default_source_type=default_source_type, method=ExtractionMethod.TABLE,
    )


_TEXT_TABLES = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def _pdf_table_facts(p: Path, pages: dict[int, str], default_source_type: SourceType) -> list[Fact]:
    """pdfplumber tables -> facts, keeping only rows whose quote is on the pymupdf page text."""
    try:
        import pdfplumber
    except ImportError:
        return []
    facts: list[Fact] = []
    try:
        pdf = pdfplumber.open(p)
    except Exception:
        return []
    with pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            haystack = norm(pages.get(page_no, ""))
            page_facts = _page_table_facts(p.name, page_no, _tables(page), haystack, default_source_type)
            if not page_facts and len(_YEARS_IN_TEXT.findall(haystack)) >= 2:
                text_tables = [t for t in _tables(page, _TEXT_TABLES) if len(t) >= 2 and max(map(len, t)) >= 2]
                page_facts = _page_table_facts(p.name, page_no, text_tables, haystack, default_source_type)
            facts += page_facts
    return facts


def _tables(page, settings: Optional[dict] = None) -> list[list[list[str]]]:
    try:
        return page.extract_tables(settings) if settings else page.extract_tables()
    except Exception:
        return []


def _page_table_facts(name: str, page_no: int, tables: list[list[list[str]]], haystack: str,
                      default_source_type: SourceType) -> list[Fact]:
    facts: list[Fact] = []
    for t_no, table in enumerate(tables, start=1):
        rows = [[str(c if c is not None else "").strip() for c in row] for row in table if row]
        head = next((k for k, row in enumerate(rows[:4]) if _is_header(row)), None)
        if head is None:
            continue

        def quote_for(i: int, row: list[str], j: int) -> Optional[str]:
            return _row_quote(row, j, haystack)

        facts += _rows_to_facts(
            rows[head], rows[head + 1:], source=name, page=page_no, quote_for=quote_for,
            locator_for=lambda i, t_no=t_no, head=head: f"page {page_no}, table {t_no}, row {head + i + 2}",
            id_for=lambda i, year, t_no=t_no, head=head: f"{name}#p{page_no}-t{t_no}-r{head + i + 2}-{year}",
            default_source_type=default_source_type, method=ExtractionMethod.TABLE,
        )
    return facts


def _row_quote(row: list[str], value_j: int, haystack: str) -> Optional[str]:
    """Longest quote built from the row's cells that is on the page and contains the value."""
    cells = [c for c in row if c]
    label, value = (cells[0] if cells else ""), row[value_j]
    candidates = [
        " ".join(cells),                                    # the whole row
        " ".join(c for c in row[: value_j + 1] if c),       # label .. value
        f"{label} {value}",
    ]
    for quote in candidates:
        if quote and value in quote and norm(quote) in haystack:
            return re.sub(r"\s+", " ", quote).strip()
    return None


# ---- prose facts -----------------------------------------------------------

def _indicators_in(text: str) -> set[str]:
    """Every indicator named in text; a phrase inside a longer one ("emissions" in "co2 emissions") is ignored."""
    lowered = text.lower()
    spans = [(m.start(), m.end(), ind)
             for ind, spec in INDICATORS.items() for syn in spec["synonyms"]
             for m in re.finditer(r"\b" + re.escape(syn) + r"\b", lowered)]
    return {ind for s, e, ind in spans
            if not any(s2 <= s and e <= e2 and (e2 - s2) > (e - s) for s2, e2, _ in spans)}


def prose_fact(sentence: str) -> Optional[tuple[str, float, str, int]]:
    """(indicator, value, unit, year) when the sentence states exactly one level, else None."""
    if len(sentence) > _MAX_SENTENCE or _NOT_A_LEVEL.search(sentence):
        return None
    qty = list(_QTY.finditer(sentence))
    if len(qty) != 1:
        return None
    m = qty[0]
    rest = sentence[: m.start()] + " " + sentence[m.end():]
    years = {int(y) for y in _YEARS_IN_TEXT.findall(rest)}
    indicators = _indicators_in(rest)
    if len(years) != 1 or len(indicators) != 1:
        return None
    indicator, unit = indicators.pop(), m.group(2)
    if unit.lower() in ("%", "percent") and not is_share(indicator):
        return None  # "emissions were 30%" is a change or a share of something else, not a level
    unit = "%" if unit.lower() == "percent" else unit
    return indicator, float(m.group(1).replace(",", "")), unit, years.pop()


def _prose_facts(name: str, pages: dict[int, str], default_source_type: SourceType) -> list[Fact]:
    facts: list[Fact] = []
    for page, text in pages.items():
        flat = re.sub(r"\s+", " ", text).strip()
        haystack = norm(text)
        for k, sentence in enumerate(re.split(r"(?<=[.!?])\s+", flat), start=1):
            parsed = prose_fact(sentence)
            if not parsed or norm(sentence) not in haystack:
                continue
            indicator, value, unit, year = parsed
            facts.append(Fact(
                fact_id=f"{name}#p{page}-s{k}", indicator=indicator, value=value, unit=unit, year=year,
                source=name, page=page, locator=f"page {page}, sentence {k}", quote=sentence.strip(),
                method=ExtractionMethod.TEXT, source_type=default_source_type, entity=find_country(sentence),
            ))
    return facts


def _dedupe(existing: list[Fact], new: list[Fact]) -> list[Fact]:
    """Drop prose facts that repeat a table fact on the same page."""
    seen = {(f.indicator, f.year, f.entity, f.value, f.page) for f in existing}
    return [f for f in new if (f.indicator, f.year, f.entity, f.value, f.page) not in seen]


# ---- chunks ----------------------------------------------------------------

def chunk_pages(p: Path, pages: Optional[dict[int, str]] = None, max_chars: int = 700) -> list[Chunk]:
    """Paragraph chunks, each tagged with its page, indicator and years."""
    chunks: list[Chunk] = []
    for page, text in (pages if pages is not None else read_pages(p)).items():
        buf = ""
        paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
        for para in paragraphs + [""]:
            if para and len(buf) + len(para) < max_chars:
                buf = f"{buf}\n\n{para}" if buf else para
                continue
            if buf:
                chunks.append(Chunk(
                    chunk_id=f"{p.name}#p{page}-{len(chunks)}", text=buf, source=p.name, page=page,
                    indicators=[ind] if (ind := match_indicator(buf)) else [],
                    years=sorted({int(y) for y in _YEARS_IN_TEXT.findall(buf)}),
                ))
            buf = para
    return chunks
