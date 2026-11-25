"""Headless Streamlit checks for the UI, plus the audit component payload and the PDF export."""
import re
from pathlib import Path

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from ui import report
from evidence_layer.ingest import read_pages
from auditor.pipeline import run_audit
from ui import audit_view

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "ui" / "app.py"
SAMPLES = ROOT / "data" / "samples"
REPORT, EVIDENCE = SAMPLES / "sample_report_2025.txt", SAMPLES / "evidence_2024_2025.csv"


@pytest.fixture(scope="module")
def sample():
    return run_audit(REPORT, [EVIDENCE], use_llm=False)


def _page_html(at) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def test_sample_button_runs_audit_and_every_view_renders():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    at.toggle(key="use_llm").set_value(False).run()  # deterministic: regex parser + templates
    at.button(key="run_sample").click().run()
    assert not at.exception

    result = at.session_state["result"]
    assert len(result.verdicts) == 6
    assert result.counts() == {"Supported": 2, "Contradicted": 1, "Unsupported": 3}
    assert at.session_state["view"] == "audit"

    page = _page_html(at)
    for value, label in [(6, "Claims found"), (2, "Supported"), (1, "Contradicted"), (3, "Unsupported"),
                         (1, "Greenwash")]:
        assert re.search(rf"<b[^>]*>{value}</b><span>{label}</span>", page), label

    at.button(key="nav_export").click().run()
    assert not at.exception
    assert at.session_state["view"] == "export"
    assert at.session_state["pdf"][1].startswith(b"%PDF")

    at.button(key="nav_upload").click().run()
    assert not at.exception
    assert "✓ Verify + score" in _page_html(at)  # step bar shows every step done


def test_audit_component_locates_claims_and_marks_values(sample):
    payload = audit_view.build_payload(sample.verdicts, read_pages(REPORT), {EVIDENCE.name: EVIDENCE},
                                       REPORT.name)
    located = {s["c"] for page in payload["report"]["pages"] for para in page["paras"] for s in para if "c" in s}
    assert located == set(range(6)) and payload["unplaced"] == []

    c01 = next(c for c in payload["claims"] if c["id"] == "C01")
    assert c01["s"] == "bad" and c01["route"] == "ledger" and c01["score"] > 0
    marked = [s["t"] for s in c01["groups"][0]["quoteSegs"] if s.get("m")]
    assert marked == ["1000", "700"]
    assert c01["groups"][0]["snippet"]["segs"][0]["t"].startswith("indicator,unit")  # header row kept
    c06 = next(c for c in payload["claims"] if c["id"] == "C06")
    assert c06["s"] == "warn" and c06["route"] == "none" and "greenwash" in c06["flags"]


def test_audit_component_escapes_text_from_user_files(sample):
    evil_text = '</script><img src=x onerror=alert(1)> & "quoted"'
    evil = sample.verdicts[0].model_copy(update={"claim": sample.verdicts[0].claim.model_copy(update={"text": evil_text})})
    html = audit_view.render(audit_view.build_payload([evil], {1: "Intro. " + evil_text}), "dark")
    assert "</script><img" not in html and "<img src=x" not in html
    assert html.count("</script>") == 2  # only the template's own data + code tags
    assert 'data-theme="dark"' in html


def test_pdf_evidence_opens_as_page_image_and_urls_as_links(sample, tmp_path):
    pdf = tmp_path / "annex.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page().insert_text((72, 120), "Total water withdrawal in 2023 was 4.9 million cubic metres.")
        doc.save(pdf)
    base = sample.verdicts[0]
    fact = base.evidence[0].model_copy(update={"source": pdf.name, "page": 2, "value": 4_900_000,
                                               "quote": "Total water withdrawal in 2023 was 4.9 million cubic metres."})
    official = fact.model_copy(update={"source": "World Bank WDI", "page": None, "url": "https://data.worldbank.org/x"})
    payload = audit_view.build_payload([base.model_copy(update={"evidence": [fact, official]})], {}, {pdf.name: pdf})
    by_source = {g["source"]: g for g in payload["claims"][0]["groups"]}
    assert by_source[pdf.name]["open"]["kind"] == "image"
    assert payload["images"][f"{pdf.name}#p2"].startswith("data:image/jpeg;base64,")
    assert [s["t"] for s in by_source[pdf.name]["quoteSegs"] if s.get("m")] == ["4.9 million"]
    assert by_source["World Bank WDI"]["open"] == {"kind": "url", "href": "https://data.worldbank.org/x"}


def test_pdf_export(sample):
    pdf = report.to_pdf(sample.verdicts, subtitle=REPORT.name)
    assert pdf.startswith(b"%PDF")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = " ".join(page.get_text() for page in doc)
    text = " ".join(text.split())
    for cid in ["C01", "C02", "C03", "C04", "C05", "C06"]:
        assert cid in text
    assert "Contradicted" in text and "Emissions,tCO2e,1000,700" in text
    assert "evidence_2024_2025.csv p.1 row 2" in text
