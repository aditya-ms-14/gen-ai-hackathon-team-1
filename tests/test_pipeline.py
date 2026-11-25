"""Runs the full pipeline on the problem-statement sample, with no LLM (deterministic)."""
import csv
from pathlib import Path

import pytest

from trust_layer.guard import apply_guard
from evidence_layer.ingest import read_pages
from auditor.pipeline import run_audit
from auditor.schemas import Claim, Direction, Evidence, Fact, Metric, Status
from trust_layer.units import normalize
from trust_layer.verifier import verify

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
EVIDENCE = SAMPLES / "evidence_2024_2025.csv"


@pytest.fixture(scope="module")
def result():
    return run_audit(SAMPLES / "sample_report_2025.txt", [EVIDENCE], use_llm=False)


@pytest.fixture(scope="module")
def by_id(result):
    return {v.claim.claim_id: v for v in result.verdicts}


def test_matches_answer_key(by_id):
    with open(SAMPLES / "answer_key.csv", encoding="utf-8") as fh:
        key = {row["claim_id"]: row for row in csv.DictReader(fh)}
    assert set(by_id) == set(key)
    for cid, row in key.items():
        v = by_id[cid]
        assert v.status.value == row["expected_status"], f"{cid}: {v.calculation}"
        if row["expected_computed"]:
            assert v.computed_value == pytest.approx(float(row["expected_computed"])), cid
        for flag in filter(None, row["expected_flags"].split(";")):
            assert flag in v.flags, cid


def test_template_explanations_pass_the_guard(by_id):
    assert not any("narration_replaced" in v.flags for v in by_id.values())


def test_checked_verdicts_cite_their_source(by_id):
    for v in by_id.values():
        if v.status != Status.UNSUPPORTED:
            assert v.evidence and all(f.source and f.quote and f.locator for f in v.evidence)
            assert v.confidence > 0


def test_guard_rejects_fabricated_quote(by_id):
    v = by_id["C02"]
    fake = v.evidence[0].model_copy(update={"quote": "Water,kL,50000,10000"})
    checked = apply_guard(v.model_copy(update={"evidence": [fake]}), {EVIDENCE.name: read_pages(EVIDENCE)})
    assert checked.status == Status.UNSUPPORTED
    assert "guard_failed" in checked.flags


def test_guard_replaces_narration_with_invented_number(by_id):
    v = by_id["C01"].model_copy(update={"explanation": "Emissions fell 55% thanks to new solar panels."})
    checked = apply_guard(v, {EVIDENCE.name: read_pages(EVIDENCE)})
    assert "55" not in checked.explanation
    assert "narration_replaced" in checked.flags
    assert checked.status == Status.CONTRADICTED


def test_conflicting_sources_are_flagged_not_picked():
    def fact(value, source):
        return Fact(fact_id=source, indicator="ghg_emissions", value=value, unit="tCO2e", year=2025,
                    source=source, quote=str(value))

    base = Fact(fact_id="b", indicator="ghg_emissions", value=1000, unit="tCO2e", year=2024, source="a.csv", quote="1000")
    claim = Claim(claim_id="X", text="Emissions reduced by 30%", indicator="ghg_emissions", metric=Metric.PCT_CHANGE,
                  direction=Direction.DECREASE, value=30, unit="%", baseline_year=2024, target_year=2025)
    v = verify(claim, Evidence(claim_id="X", facts=[base, fact(700, "a.csv"), fact(650, "b.pdf")]))
    assert v.conflict and v.status == Status.UNSUPPORTED


def test_pdf_chunks_are_page_tagged_and_searchable(tmp_path):
    import pymupdf

    from evidence_layer.ingest import ingest_file
    from evidence_layer.retrieval import HybridIndex

    pdf = tmp_path / "esg.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Introduction and company overview.")
    doc.new_page().insert_text((72, 72), "Total GHG emissions in 2025 were 700 tCO2e.")
    doc.save(pdf)

    facts, chunks = ingest_file(pdf)
    index = HybridIndex()
    index.add(chunks)
    hits = index.search("carbon emissions 2025", indicator="ghg_emissions", years=[2025])
    assert hits and hits[0].page == 2 and 2025 in hits[0].years


def test_unit_normalisation():
    assert normalize(50, "ML") == (50_000, "kL")
    assert normalize(1.2, "ktCO2e") == (1_200, "tCO2e")
