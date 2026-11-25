"""Official data sources: offline tests only (data/official snapshots, network blocked)."""
import csv
import re
from pathlib import Path

import pytest
import requests

import auditor.sources as sources
from auditor.sources import _common, fetch_official, set_offline, snapshot_pages
from auditor.sources._common import HEADER, OFFICIAL_DIR, Row
from auditor.schemas import Direction, ExtractionMethod, Metric, SourceType

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
TOL_PP = 1.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _no_network(*args, **kwargs):
    raise requests.ConnectionError("network disabled in tests")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(requests, "get", _no_network)
    set_offline(True)
    yield
    set_offline(False)
    for connector in sources.CONNECTORS:
        connector._down_until = 0.0


CASES = [
    ("co2_emissions", "DEU", [2010, 2022]),
    ("ghg_emissions", "WLD", [1990, 2023]),
    ("ghg_emissions", "EUU", [1990, 2022]),
    ("renewable_share", "IND", [2021]),
    ("renewable_electricity_share", "BRA", [2021]),
    ("water_withdrawal", "FRA", [2010, 2020]),
    ("forest_area_share", "BRA", [2022]),
    ("energy_consumption", "CHN", [2010, 2022]),   # Our World in Data
    ("waste", "GBR", [2016, 2022]),                # UNEP via SDG API
    ("waste", "EUU", [2010, 2022]),                # EEA index
]


@pytest.mark.parametrize("indicator,entity,years", CASES)
def test_facts_are_cited_and_quotes_are_in_snapshot_pages(indicator, entity, years):
    facts = fetch_official(indicator, entity, years)
    assert sorted(f.year for f in facts) == years
    assert len({f.source for f in facts}) == 1, "facts must come from one source"
    pages = snapshot_pages()
    for f in facts:
        assert f.indicator == indicator and f.entity == entity
        assert f.url and f.url.startswith("https://")
        assert f.locator and f.locator.endswith(f" · {entity} · {f.year}")
        assert f.page == 1
        assert f.method == ExtractionMethod.TABLE and f.source_type == SourceType.OFFICIAL
        assert _norm(f.quote) in _norm(pages[f.source][f.page or 1])
        assert f.quote.split(",")[3] == _common.fmt(f.value)


def test_snapshot_pages_cover_every_source():
    pages = snapshot_pages()
    assert set(pages) == {c.name for c in sources.CONNECTORS}
    assert all(1 in p and p[1].count("\n") > 10 for p in pages.values())


def test_country_names_and_codes_give_the_same_facts():
    assert fetch_official("co2_emissions", "Germany", [2015]) == fetch_official("co2_emissions", "DEU", [2015])
    assert fetch_official("ghg_emissions", "World", [2020]) == fetch_official("ghg_emissions", "WLD", [2020])


def test_priority_and_best_partial():
    # World Bank first when it covers every year.
    assert {f.source for f in fetch_official("co2_emissions", "DEU", [2010, 2022])} == {"World Bank WDI"}
    # WDI renewables stop at 2021 -> EEA (same final-energy definition) before OWID.
    assert {f.source for f in fetch_official("renewable_share", "DEU", [2023])} == {"European Environment Agency"}
    # No source has 2023 water withdrawals -> best partial (2010 only).
    assert [f.year for f in fetch_official("water_withdrawal", "FRA", [2010, 2023])] == [2010]


@pytest.mark.parametrize("args", [
    ("biodiversity", "BRA", [2015]),
    ("not_an_indicator", "DEU", [2020]),
    ("co2_emissions", "Atlantis", [2020]),
    ("co2_emissions", None, [2020]),
    ("co2_emissions", "DEU", []),
    ("co2_emissions", "DEU", ["not a year"]),
    (None, None, None),
    ("waste", "CHN", [2016, 2020]),
])
def test_failures_return_empty_list(args):
    assert fetch_official(*args) == []


def test_broken_connector_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("source exploded")

    for connector in sources.CONNECTORS:
        monkeypatch.setattr(connector, "fetch", boom)
    assert fetch_official("co2_emissions", "DEU", [2020]) == []


def test_network_failure_falls_back_to_snapshot(monkeypatch, tmp_path):
    set_offline(False)
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)   # no cached API responses either
    facts = fetch_official("co2_emissions", "DEU", [2010, 2022])
    assert [f.value for f in facts] == [814.962, 658.0886]
    assert facts[0].quote == "EN.GHG.CO2.MT.CE.AR5,DEU,2010,814.962,Mt CO2e"


def test_snapshot_files_are_small_and_canonical():
    files = sorted(OFFICIAL_DIR.glob("*.csv"))
    assert {p.stem for p in files} >= {c.slug for c in sources.CONNECTORS}
    assert sum(p.stat().st_size for p in OFFICIAL_DIR.iterdir() if p.is_file()) < 2 * 1024 * 1024
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == HEADER
        assert lines[1:] == sorted(set(lines[1:])), path.name
        for line in lines[1:]:
            row = Row.make(*line.split(","))
            assert row is not None and row.line() == line, line


# ---------------------------------------------------------------- demo dataset

def _demo_claims():
    from auditor.claims import detect_report_year, parse_claim_regex, split_claims
    from auditor.entities import find_country

    text = (SAMPLES / "demo_country_report.txt").read_text(encoding="utf-8")
    year = detect_report_year(text)
    claims = []
    for claim_id, sentence in split_claims(text):
        claim = parse_claim_regex(claim_id, sentence, year)
        claims.append(claim.model_copy(update={"entity": claim.entity or find_country(sentence)}))
    return claims


def _answer_key():
    with open(SAMPLES / "demo_answer_key.csv", encoding="utf-8", newline="") as fh:
        return {row["claim_id"]: row for row in csv.DictReader(fh)}


def _years(claim):
    if claim.metric == Metric.PCT_CHANGE:
        return [claim.baseline_year, claim.target_year]
    return [claim.target_year]


def _expected(claim):
    """The project's rules, recomputed here from the snapshot (independent of verifier.py)."""
    if claim.metric == Metric.VAGUE or claim.value is None or not claim.indicator or not claim.entity:
        return "Unsupported", None
    by_year = {f.year: f.value for f in fetch_official(claim.indicator, claim.entity, _years(claim))}
    if any(y not in by_year for y in _years(claim)):
        return "Unsupported", None
    if claim.metric == Metric.PCT_CHANGE:
        base, target = by_year[claim.baseline_year], by_year[claim.target_year]
        computed = (target - base) / base * 100
        claimed = -claim.value if claim.direction == Direction.DECREASE else claim.value
        diff = abs(abs(computed) - claim.value) if claim.direction == Direction.NONE else abs(computed - claimed)
    else:
        computed = by_year[claim.target_year]
        diff = abs(computed - claim.value)
    return ("Supported" if diff <= TOL_PP else "Contradicted"), round(computed, 2)


def test_demo_answer_key_matches_snapshot_values():
    key = _answer_key()
    claims = _demo_claims()
    assert [c.claim_id for c in claims] == list(key)
    statuses = set()
    for claim in claims:
        row = key[claim.claim_id]
        status, computed = _expected(claim)
        statuses.add(status)
        assert status == row["expected_status"], (claim.claim_id, computed)
        if row["expected_computed"]:
            assert computed == pytest.approx(float(row["expected_computed"]), abs=0.01), claim.claim_id
        else:
            assert computed is None, claim.claim_id
    assert statuses == {"Supported", "Contradicted", "Unsupported"}


def test_demo_claims_through_verifier_and_scoring():
    from auditor.schemas import Evidence
    from auditor.scoring import score
    from auditor.verifier import verify

    key = _answer_key()
    for claim in _demo_claims():
        row = key[claim.claim_id]
        facts = fetch_official(claim.indicator, claim.entity, _years(claim)) if claim.indicator else []
        verdict = score(verify(claim, Evidence(claim_id=claim.claim_id, facts=facts)))
        assert verdict.status.value == row["expected_status"], (claim.claim_id, verdict.calculation)
        if row["expected_computed"]:
            assert verdict.computed_value == pytest.approx(float(row["expected_computed"]), abs=0.01)
        for flag in filter(None, row["expected_flags"].split(";")):
            assert flag in verdict.flags, claim.claim_id
