"""Evidence layer: entity-aware ledger, PDF tables, prose facts, hybrid search,
official-data hook and LLM passage reading. Deterministic: fake LLM, fake sources."""
import sys
import types

import pymupdf
import pytest

from evidence_layer import retrieval
from evidence_layer.ingest import ingest_file, norm, read_pages
from evidence_layer.ledger import Ledger
from evidence_layer.resolver import resolve
from evidence_layer.retrieval import HybridIndex, rrf
from auditor.schemas import Chunk, Claim, Direction, ExtractionMethod, Fact, Metric, SourceType


def _fact(value, year, entity=None, source="a.csv"):
    return Fact(fact_id=f"{source}-{year}-{entity}", indicator="ghg_emissions", value=value, unit="tCO2e",
                year=year, source=source, quote=str(value), entity=entity)


def _claim(**kw):
    base = dict(claim_id="X", text="Emissions reduced by 30%", indicator="ghg_emissions", metric=Metric.PCT_CHANGE,
                direction=Direction.DECREASE, value=30, unit="%", baseline_year=2024, target_year=2025)
    return Claim(**{**base, **kw})


def _guard_ok(fact: Fact, path) -> bool:
    """The Citation Guard's check, done here so a regression in ingest can't slip past it."""
    return norm(fact.quote) in norm(read_pages(path)[fact.page])


def _pdf(path, pages):
    """pages: list of callables(page) drawing onto a new page."""
    doc = pymupdf.open()
    for draw in pages:
        draw(doc.new_page())
    doc.save(path)
    return path


def _text(*lines):
    def draw(page):
        for n, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * n), line, fontsize=11)
    return draw


def _grid(rows, ruled=True):
    def draw(page):
        page.insert_text((72, 60), "Table 2: Environmental performance", fontsize=11)
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(72 + c * 110, 80 + r * 22, 182 + c * 110, 102 + r * 22)
                if ruled:
                    page.draw_rect(rect, color=(0, 0, 0), width=0.5)
                page.insert_text((rect.x0 + 4, rect.y1 - 7), cell, fontsize=10)
    return draw


TABLE = [["Indicator", "Unit", "2024", "2025"],
         ["GHG emissions", "tCO2e", "1,000", "700"],
         ["Water consumption", "kL", "50000", "40000"],
         ["Waste", "tonnes", "800", ""]]


# ---- 1. entities -------------------------------------------------------------

def test_ledger_lookup_matches_entity():
    ledger = Ledger()
    ledger.add([_fact(700, 2025), _fact(600_000, 2025, "DEU"), _fact(5_000_000, 2025, "USA")])
    assert [f.value for f in ledger.lookup("ghg_emissions", 2025)] == [700]
    assert [f.value for f in ledger.lookup("ghg_emissions", 2025, "DEU")] == [600_000]
    assert ledger.lookup("ghg_emissions", 2025, "FRA") == []
    assert ledger.years("ghg_emissions", "DEU") == [2025]


def test_csv_country_column_becomes_entity(tmp_path):
    csv = tmp_path / "countries.csv"
    csv.write_text("indicator,country,unit,2024,2025\n"
                   "Emissions,Germany,tCO2e,900,800\n"
                   "Emissions,,tCO2e,1000,700\n", encoding="utf-8")
    facts, _ = ingest_file(csv)
    by_entity = {(f.entity, f.year): f.value for f in facts}
    assert by_entity == {("DEU", 2024): 900, ("DEU", 2025): 800, (None, 2024): 1000, (None, 2025): 700}
    assert all(_guard_ok(f, csv) for f in facts)


def test_xlsx_long_layout_with_entity_and_unit_in_label(tmp_path):
    import pandas as pd

    xlsx = tmp_path / "wdi.xlsx"
    pd.DataFrame({"Indicator": ["GHG emissions (tCO2e)", "GHG emissions (tCO2e)"], "Entity": ["DE", "World"],
                  "Year": [2020, 2020], "Value": [600, 40000]}).to_excel(xlsx, index=False)
    facts, _ = ingest_file(xlsx)
    assert {(f.entity, f.year, f.value, f.unit) for f in facts} == {("DEU", 2020, 600, "tCO2e"),
                                                                      ("WLD", 2020, 40000, "tCO2e")}
    assert all(_guard_ok(f, xlsx) for f in facts)


def test_resolver_uses_claim_entity():
    ledger = Ledger()
    ledger.add([_fact(1000, 2024), _fact(700, 2025), _fact(900, 2024, "DEU"), _fact(800, 2025, "DEU")])
    ev = resolve(_claim(entity="DEU"), ledger, HybridIndex(dense=False), use_official=False)
    assert ev.route == "ledger" and {f.value for f in ev.facts} == {900, 800}
    ev = resolve(_claim(), ledger, HybridIndex(dense=False))
    assert {f.value for f in ev.facts} == {1000, 700}


# ---- 2. official data hook -----------------------------------------------------

@pytest.fixture
def fake_sources(monkeypatch):
    calls = []

    def fetch_official(indicator, entity, years):
        calls.append((indicator, entity, list(years)))
        return [Fact(fact_id=f"wdi-{entity}-{y}", indicator=indicator, value=v, unit="MtCO2e", year=y,
                     source="World Bank WDI", locator=f"EN.GHG · {entity} · {y}", quote=f"{entity} {y} {v}",
                     source_type=SourceType.OFFICIAL, entity=entity)
                for y, v in {2024: 750.0, 2025: 690.0}.items() if y in years]

    module = types.ModuleType("evidence_layer.sources")
    module.fetch_official = fetch_official
    monkeypatch.setitem(sys.modules, "evidence_layer.sources", module)
    return calls


def test_resolver_fetches_official_data_on_ledger_miss(fake_sources):
    ledger = Ledger()
    ev = resolve(_claim(entity="DEU"), ledger, HybridIndex(dense=False))
    assert ev.route == "official"
    assert fake_sources == [("ghg_emissions", "DEU", [2024, 2025])]
    assert {(f.year, f.value) for f in ev.facts} == {(2024, 750.0), (2025, 690.0)}
    assert len(ledger.lookup("ghg_emissions", 2025, "DEU")) == 1  # cached in the ledger
    assert resolve(_claim(entity="DEU"), ledger, HybridIndex(dense=False)).route == "ledger"
    assert len(fake_sources) == 1


def test_official_hook_respects_switch_and_org_claims(fake_sources):
    assert resolve(_claim(entity="DEU"), Ledger(), HybridIndex(dense=False), use_official=False).route == "none"
    assert resolve(_claim(), Ledger(), HybridIndex(dense=False)).route == "none"  # org claim: no country data
    assert fake_sources == []


def test_resolver_works_without_sources_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "evidence_layer.sources", None)  # import raises ImportError
    ev = resolve(_claim(entity="DEU"), Ledger(), HybridIndex(dense=False))
    assert ev.route == "none" and ev.facts == []


# ---- 3. PDF tables ---------------------------------------------------------------

@pytest.mark.parametrize("ruled", [True, False], ids=["ruled", "text-aligned"])
def test_pdf_table_rows_become_facts(tmp_path, ruled):
    pdf = _pdf(tmp_path / "esg.pdf", [_text("Introduction."), _grid(TABLE, ruled)])
    facts, _ = ingest_file(pdf)
    table = {(f.indicator, f.year): f for f in facts if f.method == ExtractionMethod.TABLE}
    assert {k: (f.value, f.unit) for k, f in table.items()} == {
        ("ghg_emissions", 2024): (1000, "tCO2e"), ("ghg_emissions", 2025): (700, "tCO2e"),
        ("water_consumption", 2024): (50000, "kL"), ("water_consumption", 2025): (40000, "kL"),
        ("waste", 2024): (800, "tonnes"),
    }
    for f in table.values():
        assert f.page == 2 and "table 1" in f.locator and _guard_ok(f, pdf)


# ---- 4. prose facts --------------------------------------------------------------

def test_prose_sentence_becomes_text_fact(tmp_path):
    pdf = _pdf(tmp_path / "esg.pdf", [
        _text("Company overview."),
        _text("Total GHG emissions in 2025 were 700 tCO2e.",
              "Emissions fell by 30% in 2025 compared with 2024.",
              "We plan to cut water use to 30,000 kL by 2030."),
    ])
    facts, _ = ingest_file(pdf)
    assert len(facts) == 1
    f = facts[0]
    assert (f.indicator, f.year, f.value, f.unit, f.page, f.method) == \
        ("ghg_emissions", 2025, 700, "tCO2e", 2, ExtractionMethod.TEXT)
    assert f.quote == "Total GHG emissions in 2025 were 700 tCO2e." and _guard_ok(f, pdf)


# ---- 5. hybrid retrieval -----------------------------------------------------------

def test_rrf_fuses_ranks():
    assert rrf([[0, 1, 2], [2, 0]]) == [0, 2, 1]


def _three_page_pdf(tmp_path):
    return _pdf(tmp_path / "report.pdf", [
        _text("About the company and its governance in 2025."),
        _text("Freshwater use in 2025 across all plants was reduced through recycling."),
        _text("Greenhouse gas output in 2025: our Scope 1 and Scope 2 footprint."),
    ])


def test_bm25_only_search_is_page_tagged(tmp_path):
    _, chunks = ingest_file(_three_page_pdf(tmp_path))
    index = HybridIndex(dense=False)
    index.add(chunks)
    hits = index.search("scope 1 footprint 2025", indicator="ghg_emissions", years=[2025])
    assert hits and hits[0].page == 3 and index.last_mode == "bm25"


def test_hybrid_search_finds_the_right_page(tmp_path):
    if not retrieval.models_available():
        pytest.skip("sentence-transformers models not downloadable here")
    _, chunks = ingest_file(_three_page_pdf(tmp_path))
    index = HybridIndex()
    index.add(chunks)
    hits = index.search("carbon emissions in 2025", indicator="ghg_emissions", years=[2025])
    assert index.last_mode == "hybrid"
    assert hits and hits[0].page == 3 and hits[0].source == "report.pdf"


def test_search_falls_back_to_bm25_when_models_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", lambda: None)
    _, chunks = ingest_file(_three_page_pdf(tmp_path))
    index = HybridIndex()
    index.add(chunks)
    hits = index.search("scope 1 footprint 2025", indicator="ghg_emissions", years=[2025])
    assert hits and hits[0].page == 3 and index.last_mode == "bm25"


# ---- 6. LLM passage reading ----------------------------------------------------------

class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete_json(self, system, user):
        self.calls += 1
        return self.reply


PASSAGE = Chunk(chunk_id="esg.pdf#p4-0", source="esg.pdf", page=4, indicators=["ghg_emissions"], years=[2025],
                text="In 2025 our total greenhouse gas emissions came to\n700 tCO2e, audited by a third party.")


def _rag_setup():
    ledger = Ledger()
    ledger.add([_fact(1000, 2024)])
    index = HybridIndex(dense=False)
    index.add([PASSAGE])
    return ledger, index


def test_llm_extraction_accepts_verbatim_quote():
    ledger, index = _rag_setup()
    llm = FakeLLM({"value": 700, "unit": "tCO2e", "year": 2025, "quote": "emissions came to 700 tCO2e"})
    ev = resolve(_claim(text="Greenhouse gas emissions reduced by 30%"), ledger, index, llm)
    llm_facts = [f for f in ev.facts if f.method == ExtractionMethod.LLM]
    assert ev.route == "rag" and llm.calls == 1 and len(llm_facts) == 1
    f = llm_facts[0]
    assert (f.value, f.unit, f.year, f.source, f.page) == (700, "tCO2e", 2025, "esg.pdf", 4)
    assert norm(f.quote) in norm(PASSAGE.text)


@pytest.mark.parametrize("reply", [
    {"value": 700, "unit": "tCO2e", "year": 2025, "quote": "total emissions were 700 tCO2e"},  # invented quote
    {"value": 650, "unit": "tCO2e", "year": 2025, "quote": "emissions came to 700 tCO2e"},     # value not in quote
    {"value": 700, "unit": "tCO2e", "year": 2024, "quote": "emissions came to 700 tCO2e"},     # wrong year
    {"value": None, "unit": None, "year": None, "quote": None},                                # "not stated"
    None,                                                                                       # both providers down
], ids=["invented", "value", "year", "null", "none"])
def test_llm_extraction_rejects_unverifiable_answers(reply):
    ledger, index = _rag_setup()
    ev = resolve(_claim(text="Greenhouse gas emissions reduced by 30%"), ledger, index, FakeLLM(reply))
    assert [f.year for f in ev.facts] == [2024]
    assert ev.passages and ev.passages[0].page == 4


def test_no_llm_means_no_extraction():
    ledger, index = _rag_setup()
    ev = resolve(_claim(text="Greenhouse gas emissions reduced by 30%"), ledger, index, None)
    assert ev.route == "rag" and [f.year for f in ev.facts] == [2024]
