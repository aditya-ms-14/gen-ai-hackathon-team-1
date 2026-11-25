r"""Build two fictional industry-style reports from the real World Bank WDI bulk CSV (reproducible).

    set PYTHONPATH=<repo root>
    .venv\Scripts\python.exe scripts/make_industry_reports.py <API_6_DS2_en_csv_v2_*.csv or the .zip>

Writes to data/samples/industry/: two PDF reports, two answer keys (company_answer_key.csv format),
worldbank_environment_extract.csv (same bulk format, only the countries and 6 series used) and README.md.
Every figure in a claim is computed from the data with code. Supported claims state the true figure rounded
to a whole number (|true - claimed| <= 0.6); Contradicted claims are 15-20 points too high; Unsupported claims
name a year with no data or an indicator the data lacks; one sentence per report is vague (greenwash).
The publishers are fictional and every page says so.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import zipfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence_layer.sources.worldbank import WorldBank  # noqa: E402
from evidence_layer.worldbank_bulk import SERIES_MAP, read_worldbank_bulk  # noqa: E402

OUT = ROOT / "data" / "samples" / "industry"
FOOTER = "Fictional demonstration document for testing the PS 99 Citation Auditor. Figures from World Bank WDI."
NAMES = {"IND": "India", "WLD": "the world", "DEU": "Germany", "FRA": "France", "EUU": "the European Union",
         "ITA": "Italy", "POL": "Poland"}
EXTRACT_ENTITIES = ["IND", "WLD", "DEU", "FRA", "EUU", "ITA", "POL"]
IND_NAME = {"ghg_emissions": "greenhouse gas emissions", "co2_emissions": "CO2 emissions",
            "water_withdrawal": "freshwater withdrawals"}
LEVEL = {  # indicator -> (phrase, "of what")
    "renewable_share": ("Renewable energy consumption", "of total energy demand"),
    "renewable_electricity_share": ("Renewable electricity", "of total electricity output"),
    "forest_area_share": ("Forest area", "of land area"),
}
IND_OF_SERIES = {codes[0]: ind for ind, codes in WorldBank.SERIES.items()}


class Data:
    def __init__(self, path: Path):
        df = read_worldbank_bulk(path)
        self.v = {(r.entity, IND_OF_SERIES[r.series], int(r.year)): float(r.value) for r in df.itertuples()}

    def get(self, e, ind, y):
        return self.v[(e, ind, y)]

    def pct(self, e, ind, y0, y1):
        a, b = self.get(e, ind, y0), self.get(e, ind, y1)
        return (b - a) / a * 100


def subject(e: str, ind: str) -> str:
    name = IND_NAME[ind]
    return {"IND": f"India's {name}", "WLD": f"Global {name}", "EUU": f"The European Union's {name}"}.get(
        e, f"{NAMES[e]}'s {name}")


def make_claims(d: Data, spec: list[tuple]) -> list[dict]:
    """spec items: (kind, status, entity, indicator, a, b); see the branches for what a / b mean."""
    out = []
    for kind, status, e, ind, a, b in spec:
        c = dict(kind=kind, status=status, flags="", computed=None, entity=e)
        if kind == "change":          # a, b = base year, target year
            true = d.pct(e, ind, a, b)
            mag = round(abs(true)) + (0 if status == "Supported" else (15 if true > 0 else 20))
            verb = "rose" if true > 0 else "fell"
            c["text"] = f"{subject(e, ind)} {verb} by {mag}% between {a} and {b}."
            c["computed"] = round(true, 2)
            c["why"] = (f"{d.get(e, ind, a):.1f} -> {d.get(e, ind, b):.1f} is {true:+.1f}%; claimed "
                        f"{'+' if true > 0 else '-'}{mag}%" + (" (within 1.0 pp)" if status == "Supported" else ""))
        elif kind == "level":         # b = year
            true = d.get(e, ind, b)
            claimed = round(true) if status == "Supported" else round(true) + 15
            phrase, of = LEVEL[ind]
            c["text"] = f"{phrase} in {NAMES[e]} reached {claimed}% {of} in {b}."
            c["computed"] = round(true, 2)
            c["why"] = f"{b} value is {true:.1f}%; claimed {claimed}%"
        elif kind == "nodata":        # a = year with no data, b = claimed %
            phrase, of = LEVEL[ind]
            c["text"] = f"{phrase} in {NAMES[e]} reached {b}% {of} in {a}."
            c["why"] = f"World Bank WDI has no {a} value for this series"
        elif kind == "waste":         # a = base year, b = claimed %
            c["text"] = f"{NAMES[e]}'s industrial waste fell by {b}% between {a} and 2022."
            c["why"] = "the World Bank file has no industrial waste series"
        else:                         # vague: the sentence is in `ind`
            c["text"] = ind
            c["flags"] = "greenwash"
            c["why"] = "vague claim with no number or baseline"
        out.append(c)
    return out


SPEC1 = [  # India
    ("change", "Supported", "IND", "ghg_emissions", 2010, 2022),
    ("change", "Supported", "IND", "co2_emissions", 2005, 2022),
    ("change", "Contradicted", "IND", "co2_emissions", 2015, 2022),
    ("change", "Supported", "WLD", "ghg_emissions", 2000, 2022),
    ("level", "Supported", "IND", "renewable_share", None, 2021),
    ("level", "Supported", "IND", "renewable_electricity_share", None, 2021),
    ("level", "Contradicted", "IND", "renewable_share", None, 2021),
    ("nodata", "Unsupported", "IND", "renewable_share", 2024, 38),
    ("change", "Supported", "IND", "water_withdrawal", 2000, 2022),
    ("level", "Supported", "IND", "forest_area_share", None, 2022),
    ("level", "Contradicted", "IND", "forest_area_share", None, 2024),
    ("waste", "Unsupported", "IND", None, 2010, 15),
    ("vague", "Unsupported", "IND",
     "India's heavy industry is committed to clean and sustainable manufacturing for the decades ahead.", None, None),
]
SPEC2 = [  # Europe
    ("change", "Supported", "DEU", "ghg_emissions", 2010, 2022),
    ("change", "Contradicted", "DEU", "co2_emissions", 2010, 2022),
    ("change", "Supported", "FRA", "co2_emissions", 2000, 2022),
    ("change", "Supported", "EUU", "ghg_emissions", 2005, 2022),
    ("change", "Supported", "ITA", "ghg_emissions", 2005, 2022),
    ("level", "Supported", "DEU", "renewable_electricity_share", None, 2021),
    ("level", "Contradicted", "POL", "renewable_electricity_share", None, 2021),
    ("level", "Supported", "FRA", "renewable_share", None, 2021),
    ("nodata", "Unsupported", "EUU", "renewable_share", 2024, 30),
    ("change", "Supported", "DEU", "water_withdrawal", 2005, 2022),
    ("level", "Supported", "EUU", "forest_area_share", None, 2022),
    ("level", "Contradicted", "ITA", "renewable_share", None, 2021),
    ("waste", "Unsupported", "DEU", None, 2010, 12),
    ("vague", "Unsupported", "EUU",
     "European manufacturers are building a clean and sustainable industrial future together.", None, None),
]

# Filler sentences: no digits, no vague-marketing words, so the parser finds only the generated claims.
FILL = {
    1: {
        "intro": ["Heavy industry sits at the centre of the national energy transition.",
                  "Steel, cement, fertiliser and power producers account for much of the fuel demand of the country.",
                  "This review gathers public statistics so that members can compare their own progress with the national picture."],
        "Emissions": ["Industrial output has grown quickly over the past two decades, and fuel use has grown with it.",
                      "Members report that efficiency gains in kilns and furnaces have slowed the rise but not stopped it.",
                      "Process emissions from cement and steel remain the hardest part of the inventory to reduce."],
        "Energy": ["Renewable supply is expanding fastest in the power sector, where solar and wind projects are tendered every year.",
                   "Industrial heat is harder to supply from renewable sources, so progress there is slower.",
                   "Members are signing long-term supply contracts to bring more renewable power into factories."],
        "Water": ["Water is a shared constraint for factories, farms and cities in many river basins.",
                  "Plants in dry regions are investing in recycling loops and condensate recovery."],
        "Land": ["Plantations and restoration schemes around industrial sites are a visible part of member programmes.",
                 "Land use planning is increasingly part of the permitting process for new plants."],
        "Outlook": ["Members will use this review as a common reference point when they set their own targets.",
                    "The forum plans to repeat the exercise when new national statistics are released."],
    },
    2: {
        "intro": ["European manufacturers operate under the most developed carbon pricing system in the world.",
                  "This report follows the main climate indicators for the largest producing economies of the region.",
                  "All figures are taken from public national and regional statistics."],
        "Emissions": ["Industrial regions have shifted away from coal and towards gas and electricity over several decades.",
                      "The pace of change differs by country and depends on the fuel mix that each economy started with.",
                      "Energy prices and trade conditions also shaped output and fuel demand in recent years."],
        "Energy": ["Wind and solar power now supply a growing part of factory electricity across the region.",
                   "Nuclear and hydropower keep a large role in some countries, while others rely more on new wind capacity.",
                   "Several associations are working on shared standards for renewable power contracts."],
        "Water": ["Cooling and process water remain essential for chemicals, paper and metal producers.",
                  "Drought episodes in recent summers have pushed plants to measure their intake more closely."],
        "Land": ["Forests cover a large part of the continent and provide timber, storage for carbon and recreation.",
                 "Industrial sites increasingly report on the land they restore after closure."],
        "Outlook": ["The association will review these indicators again when the next statistical release is published.",
                    "Members are invited to compare their own data with the national and regional figures shown here."],
    },
}


def split_sections(claims: list[dict]) -> dict[str, list[str]]:
    sec = {"Emissions": [], "Energy": [], "Water": [], "Land": [], "Outlook": []}
    for c in claims:
        t = c["text"].lower()
        if c["kind"] in ("waste", "vague"):
            s = "Outlook"
        elif "greenhouse" in t or "co2" in t:
            s = "Emissions"
        elif "freshwater" in t:
            s = "Water"
        elif "forest" in t:
            s = "Land"
        else:
            s = "Energy"
        sec[s].append(c["text"])
    return sec


CSS = """body{font-family:sans-serif;font-size:10.5pt;line-height:1.45;color:#1d2a35}
h2{font-size:15pt;color:%s;margin:14px 0 4px 0;border-bottom:1.5px solid %s}
p{margin:0 0 8px 0;text-align:left}.lead{font-size:11.5pt;color:#33444f}"""


def paras(fill: list[str], claims: list[str]) -> list[str]:
    """Interleave filler and claim sentences into short paragraphs (one claim per sentence)."""
    out, f, cl = [], list(fill), list(claims)
    while cl or f:
        part = [f.pop(0)] if f else []
        part += [cl.pop(0) for _ in range(min(2, len(cl)))]
        if cl and f:
            part.append(f.pop(0))
        out.append(" ".join(part))
    return out


def build_pdf(path: Path, title: str, publisher: str, subtitle: str, colour: tuple, hexcol: str,
              fill: dict, sec: dict[str, list[str]]) -> None:
    doc = pymupdf.open()
    W, H = pymupdf.paper_size("a4")
    css = CSS % (hexcol, hexcol)

    def put(page, html: str, rect, css_=None):
        spare, _ = page.insert_htmlbox(rect, html, css=css if css_ is None else css_, scale_low=1)
        if spare < 0:
            raise RuntimeError(f"text does not fit on a page of {path.name}")

    def footer(page, n):
        page.draw_line((50, H - 52), (W - 50, H - 52), color=(0.7, 0.7, 0.7), width=0.6)
        page.insert_text((50, H - 38), FOOTER, fontsize=7.2, color=(0.35, 0.35, 0.35))
        page.insert_text((W - 62, H - 24), f"Page {n}", fontsize=8, color=(0.35, 0.35, 0.35))

    def block(name: str, heading: str) -> str:
        return f"<h2>{heading}</h2>" + "".join(f"<p>{p}</p>" for p in paras(fill[name], sec.get(name, [])))

    body = pymupdf.Rect(50, 60, W - 50, H - 62)
    p1 = doc.new_page(width=W, height=H)
    p1.draw_rect(pymupdf.Rect(0, 0, W, 215), color=None, fill=colour)
    put(p1, f"""<div style="color:white">
        <p style="font-size:9.5pt;margin:0 0 10px 0">{publisher.upper()}</p>
        <p style="font-size:27pt;font-weight:bold;line-height:1.1;margin:0 0 12px 0">{title}</p>
        <p style="font-size:12pt;margin:0">{subtitle}</p></div>""", pymupdf.Rect(50, 40, W - 50, 205), css_="")
    put(p1, "<p style='font-size:8.5pt;color:#8a1f1f'><b>Fictional publisher and fictional document.</b> "
            "Prepared only to test an automated claim checker. Not a real organisation's report.</p>",
        pymupdf.Rect(50, 225, W - 50, 262), css_="")
    exec_html = ("<h2>Executive summary</h2>"
                 f"<p class='lead'>{' '.join(fill['intro'][:2])}</p><p>{fill['intro'][2]}</p>"
                 + block("Emissions", "1. Emissions"))
    put(p1, exec_html, pymupdf.Rect(50, 270, W - 50, H - 62))
    footer(p1, 1)
    p2 = doc.new_page(width=W, height=H)
    put(p2, block("Energy", "2. Energy") + block("Water", "3. Water"), body)
    footer(p2, 2)
    p3 = doc.new_page(width=W, height=H)
    put(p3, block("Land", "4. Land") + block("Outlook", "5. Outlook"), body)
    footer(p3, 3)
    doc.set_metadata({"title": title, "author": publisher, "subject": "Fictional test document"})
    doc.save(path)


def write_key(path: Path, claims: list[dict], pdf_name: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "expected_status", "expected_computed", "expected_flags", "why", "source"])
        for i, c in enumerate(claims, 1):
            w.writerow([f"C{i:02d}", c["status"], "" if c["computed"] is None else c["computed"], c["flags"],
                        c["why"], f"{pdf_name} | {c['text']}"])


def write_extract(src: Path, dst: Path) -> int:
    codes = set(SERIES_MAP)
    lines = src.read_text(encoding="utf-8-sig").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith('"Country Name"'))
    keep = []
    for ln in lines[head + 1:]:
        if ln.strip():
            row = next(csv.reader([ln]))
            if row[1] in EXTRACT_ENTITIES and row[3] in codes:
                keep.append(ln)
    dst.write_text("\n".join(lines[:head + 1] + keep) + "\n", encoding="utf-8")
    return len(keep)


def find_csv(arg: Path, tmp: Path) -> Path:
    if arg.suffix.lower() != ".zip":
        return arg
    with zipfile.ZipFile(arg) as z:
        name = next(n for n in z.namelist() if Path(n).name.startswith("API_") and n.endswith(".csv"))
        data = z.read(name)           # read as data only; nothing from the archive is executed
    p = tmp / "wb.csv"
    p.write_bytes(data)
    return p


REPORTS = [
    ("industry_report_1_india.pdf", "industry_report_1_answer_key.csv", SPEC1, FILL[1],
     "Indian Industry Decarbonisation Review 2024", "Bharat Industrial Climate Forum (fictional)",
     "Emissions, energy, water and land: how Indian industry is changing", (0.10, 0.32, 0.27), "#1c5240"),
    ("industry_report_2_europe.pdf", "industry_report_2_answer_key.csv", SPEC2, FILL[2],
     "European Manufacturing Climate Progress Report 2024", "European Manufacturers Climate Association (fictional)",
     "Germany, France, Italy, Poland and the EU: indicators for industry", (0.12, 0.22, 0.42), "#1f3a6b"),
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = find_csv(Path(sys.argv[1]), Path(tmp))
        d = Data(src)
        n = write_extract(src, OUT / "worldbank_environment_extract.csv")
    summary = []
    for pdf, key, spec, fill, title, pub, sub, col, hexcol in REPORTS:
        claims = make_claims(d, spec)
        build_pdf(OUT / pdf, title, pub, sub, col, hexcol, fill, split_sections(claims))
        write_key(OUT / key, claims, pdf)
        counts = {s: sum(c["status"] == s for c in claims) for s in ("Supported", "Contradicted", "Unsupported")}
        summary.append((pdf, key, title, counts, len(claims)))
        for i, c in enumerate(claims, 1):
            print(f"{pdf[:17]} C{i:02d} {c['status']:12} {c['computed']}  {c['text']}")
    L = ["# Industry-style demo reports (fictional publishers, real World Bank data)", "",
         "Both reports are invented documents for testing the PS 99 Citation Auditor. The publishers do not exist. "
         "Every checkable figure comes from the World Bank World Development Indicators (WDI) bulk download, and "
         "some claims are deliberately wrong or unsupported. Rebuild with "
         "`python scripts/make_industry_reports.py <WDI bulk .csv or .zip>`.", "", "## Files", "",
         "| File | What it is |", "|---|---|"]
    for pdf, key, title, c, k in summary:
        L += [f"| `{pdf}` | \"{title}\" (3 pages, prose only) |", f"| `{key}` | answer key for it ({k} claims) |"]
    L += [f"| `worldbank_environment_extract.csv` | {n} rows of the WDI bulk CSV in the same format as the 22.9 MB "
          "download: 7 entities x 6 series. Upload it as the evidence file. |", "",
          "## Which evidence to upload", "",
          "Upload `worldbank_environment_extract.csv` with either report. The full WDI bulk CSV gives the same "
          "verdicts. With no evidence file the app falls back to its official-data connectors.", "",
          "## Expected verdict counts", "",
          "| Report | Supported | Contradicted | Unsupported (of which greenwash) |", "|---|---|---|---|"]
    for pdf, key, title, c, k in summary:
        L.append(f"| `{pdf}` | {c['Supported']} | {c['Contradicted']} | {c['Unsupported']} (1) |")
    L += ["", "Tolerance is +/-1.0 percentage point for % changes and share levels. Unsupported claims name a year "
          "the data does not cover (renewable share ends in 2021), an indicator it lacks (industrial waste), "
          "or are vague (greenwash)."]
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("extract rows:", n)


if __name__ == "__main__":
    main()
