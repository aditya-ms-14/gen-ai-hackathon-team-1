"""The two fictional industry reports (data/samples/industry) audited against the World Bank extract CSV."""
import csv
from pathlib import Path

import pytest

from auditor.pipeline import run_audit
from scripts.evaluate import load_key, match

D = Path(__file__).resolve().parent.parent / "data" / "samples" / "industry"
EXTRACT = D / "worldbank_environment_extract.csv"
CASES = [("industry_report_1_india.pdf", "industry_report_1_answer_key.csv"),
         ("industry_report_2_europe.pdf", "industry_report_2_answer_key.csv")]


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def audit(request):
    pdf, key = request.param
    res = run_audit(D / pdf, [EXTRACT], use_llm=False, use_official=False)
    key_rows = load_key(D / key)
    return res, key_rows, match(key_rows, res.verdicts)


def test_extract_is_small_and_in_bulk_format():
    assert EXTRACT.stat().st_size < 300_000
    assert EXTRACT.read_text(encoding="utf-8").startswith('"Data Source"')


def test_matches_answer_key(audit):
    res, key, m = audit
    assert not res.errors
    for row in key:
        v = m[row["claim_id"]]
        assert v is not None, f"not extracted: {row['source']}"
        assert v.status.value == row["expected_status"], (row["claim_id"], v.status, v.calculation)
        exp_flags = [f for f in row["expected_flags"].split(";") if f]
        assert all(f in v.flags for f in exp_flags), (row["claim_id"], v.flags)
        assert "guard_failed" not in v.flags
        if row["expected_computed"]:
            assert v.computed_value is not None
            assert abs(v.computed_value - float(row["expected_computed"])) <= 0.05, row["claim_id"]


def test_checked_verdicts_cite_world_bank_extract(audit):
    res, key, m = audit
    checked = [v for v in res.verdicts if v.status.value in ("Supported", "Contradicted")]
    assert len(checked) == sum(r["expected_status"] in ("Supported", "Contradicted") for r in key)
    for v in checked:
        assert v.evidence
        assert all(f.source == EXTRACT.name and f.locator and f.quote and f.source_type.value == "official"
                   for f in v.evidence), v.claim.text


def test_reports_say_they_are_fictional():
    import pymupdf
    for pdf, _ in CASES:
        with pymupdf.open(D / pdf) as doc:
            assert "fictional" in doc[0].get_text().lower()
            for page in doc:
                assert "Fictional demonstration document for testing the PS 99 Citation Auditor" in page.get_text()
