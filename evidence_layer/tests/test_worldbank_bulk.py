"""World Bank WDI bulk CSV as an evidence upload (synthetic small file in the same format)."""
import pytest

from auditor.indicators import match_indicator
from auditor.schemas import Claim, Direction, Metric, SourceType, Status, Verdict
from evidence_layer import ingest
from evidence_layer.ingest import ingest_file, read_pages
from evidence_layer.sources.worldbank import WorldBank
from evidence_layer.worldbank_bulk import SERIES_MAP, is_worldbank_bulk
from trust_layer.guard import quotes_ok

HEADER = '"Country Name","Country Code","Indicator Name","Indicator Code","2019","2020","2021","2022",'


def _write(tmp_path, rows, name="API_6_DS2_en_csv_v2_1.csv"):
    p = tmp_path / name
    body = "\n".join(rows)
    p.write_text('﻿"Data Source","World Development Indicators",\n\n"Last Updated Date","2026-11-11",\n\n'
                 f"{HEADER}\n{body}\n", encoding="utf-8")
    return p


@pytest.fixture
def wb(tmp_path):
    return _write(tmp_path, [
        '"Germany","DEU","Total greenhouse gas emissions excluding LULUCF (Mt CO2e)","EN.GHG.ALL.MT.CE.AR5","800.5","700.25","","600",',
        '"Germany","DEU","Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)","EN.GHG.CO2.MT.CE.AR5","650","","","500",',
        '"World","WLD","Renewable energy consumption (% of total final energy consumption)","EG.FEC.RNEW.ZS","17.1","17.5","18","",',
        '"European Union","EUU","Forest area (% of land area)","AG.LND.FRST.ZS","39.9","40","40.1","40.2",',
        '"India","IND","Population, total","SP.POP.TOTL","1000","1100","1200","1300",',
    ])


def test_detects_format(wb, tmp_path):
    assert is_worldbank_bulk(wb)
    plain = tmp_path / "plain.csv"
    plain.write_text("indicator,unit,2024\nScope 1 emissions,tCO2e,100\n")
    assert not is_worldbank_bulk(plain)


def test_facts(wb):
    facts, chunks = ingest_file(wb)
    assert chunks == []
    by = {(f.indicator, f.entity, f.year): f for f in facts}
    assert by[("ghg_emissions", "DEU", 2020)].value == 700.25
    assert by[("ghg_emissions", "DEU", 2020)].unit == "Mt CO2e"
    assert by[("co2_emissions", "DEU", 2019)].value == 650
    assert by[("renewable_share", "WLD", 2021)].unit == "%"
    assert by[("forest_area_share", "EUU", 2022)].value == 40.2
    assert ("renewable_share", "WLD", 2022) not in by            # blank cell = no data
    assert ("ghg_emissions", "DEU", 2021) not in by
    assert all(f.source_type == SourceType.OFFICIAL and f.entity for f in facts)
    assert len(facts) == 3 + 2 + 3 + 4


def test_unmapped_series_ignored(wb):
    facts, _ = ingest_file(wb)
    assert not any("POP" in f.fact_id or f.entity == "IND" for f in facts)
    assert "Population" not in read_pages(wb)[1]


def test_quotes_pass_citation_guard(wb):
    facts, _ = ingest_file(wb)
    pages = {wb.name: read_pages(wb)}
    for f in facts:
        v = Verdict(claim=Claim(claim_id="C", text="t", metric=Metric.SHARE, direction=Direction.NONE),
                    status=Status.SUPPORTED, evidence=[f])
        assert quotes_ok(v, pages), f.quote
    assert len({f.quote for f in facts}) == len(facts)               # one canonical line per fact


def test_every_mapped_code_matches_its_indicator():
    expected = {code: ind for ind, codes in WorldBank.SERIES.items() for code in codes}
    assert set(SERIES_MAP) == set(expected)
    for code, (text, unit) in SERIES_MAP.items():
        assert match_indicator(text) == expected[code], code
        assert unit == WorldBank.UNITS[code]


def test_table_is_parsed_once(wb, monkeypatch):
    calls = []
    real = ingest.read_worldbank_bulk
    monkeypatch.setattr(ingest, "read_worldbank_bulk", lambda p: calls.append(p) or real(p))
    ingest._TABLE_CACHE.clear()
    ingest_file(wb)
    read_pages(wb)
    assert len(calls) == 1


def test_normal_csv_unchanged(tmp_path):
    p = tmp_path / "ev.csv"
    p.write_text("entity,indicator,year,value,unit\nGermany,Scope 1 emissions,2024,100,tCO2e\nWLD,Emissions,2024,5,tCO2e\n")
    facts, _ = ingest_file(p)
    assert [(f.entity, f.value, f.unit) for f in facts] == [("DEU", 100, "tCO2e"), (None, 5, "tCO2e")]
