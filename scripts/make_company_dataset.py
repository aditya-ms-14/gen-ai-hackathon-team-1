r"""Build the fictional "Greenfield Manufacturing Ltd" evaluation dataset (reproducible).

    set PYTHONPATH=<repo root>
    .venv\Scripts\python.exe scripts/make_company_dataset.py

Writes to data/samples/: company_report_2024.pdf, company_evidence_tables.pdf,
company_evidence.xlsx, company_answer_key.csv.  Expected verdicts are computed here with the
project's rules (% change = (target - baseline) / baseline * 100, levels compared directly,
tolerance +/-1.0 percentage point for %/shares, 2 % relative for absolute amounts).
"""
from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
import pymupdf

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"
TOL_PP, TOL_REL = 1.0, 0.02

YEARS = (2022, 2023, 2024)
# indicator label, unit, values for YEARS.  Labels use names auditor/indicators.py recognises.
TABLE = [
    ("Scope 1 emissions", "tCO2e", (12000, 11200, 10560)),
    ("Water consumption", "kL", (210000, 200000, 164000)),
    ("Renewable electricity", "%", (30, 38, 45)),
    ("Energy consumption", "MWh", (90000, 88000, 85500)),
    ("Waste generated", "tonnes", (5000, 4800, 4400)),
    ("Freshwater withdrawal", "kL", (400000, 380000, 360000)),
]
XLSX_OVERRIDE = {("Freshwater withdrawal", 2024): 395000}  # the deliberate conflict (PDF says 360,000)


def val(label: str, year: int) -> float:
    return next(v[YEARS.index(year)] for l, _, v in TABLE if l == label)


def pct(label: str, b: int, t: int) -> float:
    return round((val(label, t) - val(label, b)) / val(label, b) * 100, 2)


def claims() -> list[tuple]:
    """(page, sentence, expected status, expected computed, expected flags, why)"""
    g = pct("Scope 1 emissions", 2022, 2024)
    w = pct("Water consumption", 2023, 2024)
    e = pct("Energy consumption", 2022, 2024)
    wa = pct("Waste generated", 2022, 2024)
    return [
        (2, "Scope 1 emissions fell by 12% from 2022 to 2024.", "Supported", g, "",
         f"12000 -> 10560 tCO2e is {g}%; claimed -12% (within 1.0 pp)"),
        (2, "Water consumption was reduced by 18% compared with 2023.", "Supported", w, "",
         f"200000 -> 164000 kL is {w}%; claimed -18% (within 1.0 pp)"),
        (2, "Renewable electricity reached 45% of total electricity use in 2024.", "Supported", 45.0, "",
         "2024 renewable electricity share is 45%; claimed 45%"),
        (2, "Energy consumption fell by 5% from 2022 to 2024.", "Supported", e, "",
         f"90000 -> 85500 MWh is {e}%; claimed -5% (within 1.0 pp)"),
        (3, "Waste generated decreased by 20% compared with 2022.", "Contradicted", wa, "",
         f"5000 -> 4400 tonnes is {wa}%; claimed -20%"),
        (3, "Renewable electricity rose from 30% in 2022 to 55% in 2024.", "Contradicted", 45.0, "",
         "2024 renewable electricity share is 45%; claimed 55%"),
        (3, "Waste generated totalled 4,900 tonnes in 2024.", "Contradicted", 4400.0, "",
         "recorded 2024 waste is 4400 tonnes; claimed 4900 (11% off, beyond 2% tolerance)"),
        (3, "Scope 1 emissions were 10,560 tCO2e in 2024.", "Supported", 10560.0, "",
         "recorded 2024 Scope 1 emissions are 10560 tCO2e; claimed 10560"),
        (3, "Freshwater withdrawal fell by 10% from 2022 to 2024.", "Unsupported", None, "conflict",
         "PDF table says 360000 kL for 2024, XLSX says 395000 kL: sources disagree"),
        (4, "CO2 emissions from logistics fell by 25% compared with 2022.", "Unsupported", None, "",
         "no CO2 emissions indicator in any evidence file"),
        (4, "Water consumption fell by 30% compared with 2019.", "Unsupported", None, "",
         "no 2019 water consumption data"),
        (4, "Land restored reached 120 hectares in 2024.", "Unsupported", None, "",
         "no land restored data in any evidence file"),
        (4, "Renewable electricity will reach 80% by 2030.", "Unsupported", None, "",
         "future target, nothing to check yet"),
        (4, "Our operations are eco-friendly and environmentally responsible.", "Unsupported", None, "greenwash",
         "vague claim with no number or baseline"),
        (4, "We are committed to net zero.", "Unsupported", None, "greenwash",
         "net zero with no target year"),
    ]


FILLER = {
    2: ["Greenfield Manufacturing Ltd makes precision metal components for the automotive and rail sectors.",
        "This chapter summarises our operational performance for the reporting year.",
        "Our environmental data are collected monthly at each of our three sites."],
    3: ["Our site teams review resource use every quarter with the sustainability committee.",
        "The following results build on the programmes described earlier in this report.",
        "Capital investment in process upgrades continued across the plant during the year."],
    4: ["We continue to engage suppliers and local communities on environmental topics.",
        "The final chapter sets out our outlook and commitments.",
        "Thank you to every employee who contributed to these results."],
}
TITLES = {2: "Operational Performance", 3: "Resource Use and Waste", 4: "Outlook and Commitments"}


def build_report(path: Path, cl: list[tuple]) -> None:
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 200), "Greenfield Manufacturing Ltd", fontsize=24)
    p.insert_text((72, 240), "Sustainability Report 2024", fontsize=18)
    p.insert_text((72, 300), "A fictional company created for evaluation purposes.", fontsize=11)
    for pg in (2, 3, 4):
        page = doc.new_page()
        page.insert_text((72, 80), TITLES[pg], fontsize=16)
        sents = [s for (n, s, *_) in cl if n == pg]
        fill = FILLER[pg]
        paras = [" ".join([fill[0]] + sents[0:2]), " ".join([fill[1]] + sents[2:4]),
                 " ".join([fill[2]] + sents[4:6])]
        y = 120
        for para in paras:
            page.insert_textbox(pymupdf.Rect(72, y, 523, y + 150), para, fontsize=11)
            y += 150
    doc.save(path)


def build_evidence_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 70), "Greenfield Manufacturing Ltd - Environmental Data Tables 2022-2024", fontsize=13)
    page.insert_text((72, 100), "Table 1. Key environmental indicators", fontsize=11)
    cols = [72, 232, 292, 372, 452, 523]
    rows = [["Indicator", "Unit", "2022", "2023", "2024"]] + [
        [lab, unit, *[str(v) if unit == "%" else f"{v:,}" for v in vals]] for lab, unit, vals in TABLE]
    rh, y0 = 26, 120
    for i, row in enumerate(rows):
        y = y0 + i * rh
        for j, cell in enumerate(row):
            r = pymupdf.Rect(cols[j], y, cols[j + 1], y + rh)
            page.draw_rect(r, color=(0, 0, 0), width=0.8, fill=(0.9, 0.9, 0.9) if i == 0 else None)
            page.insert_text((r.x0 + 4, r.y0 + 17), cell, fontsize=10)
    doc.save(path)


def build_xlsx(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Indicators"
    ws.append(["indicator", "unit", *YEARS])
    for lab, unit, vals in TABLE:
        ws.append([lab, unit, *[XLSX_OVERRIDE.get((lab, y), v) for y, v in zip(YEARS, vals)]])
    wb.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cl = claims()
    build_report(OUT / "company_report_2024.pdf", cl)
    build_evidence_pdf(OUT / "company_evidence_tables.pdf")
    build_xlsx(OUT / "company_evidence.xlsx")
    with open(OUT / "company_answer_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "expected_status", "expected_computed", "expected_flags", "why", "source"])
        for i, (pg, sent, status, comp, flags, why) in enumerate(cl, 1):
            w.writerow([f"C{i:02d}", status, "" if comp is None else comp, flags, why,
                        f"company_report_2024.pdf p.{pg} | {sent}"])
    print(f"wrote {len(cl)} claims to {OUT}")


if __name__ == "__main__":
    main()
