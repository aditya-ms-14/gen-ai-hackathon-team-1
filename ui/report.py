"""Audit Report export. Owner: Sibi (last hour); PDF export added with the UI.

Done:  table rows, CSV, Markdown, PDF (PyMuPDF Story: HTML -> A4 pages, with links).
"""
from __future__ import annotations

import html as _html
from datetime import date
from typing import Optional

import pandas as pd

from auditor.indicators import label
from auditor.schemas import Verdict


def to_rows(verdicts: list[Verdict]) -> list[dict]:
    rows = []
    for v in verdicts:
        first = v.evidence[0] if v.evidence else None
        rows.append({
            "Claim": v.claim.claim_id,
            "Text": v.claim.text,
            "Indicator": label(v.claim.indicator),
            "Status": v.status.value,
            "Calculation": v.calculation or "",
            "Confidence": v.confidence,
            "Flags": ", ".join(v.flags),
            "Source": f"{first.source} p.{first.page} {first.locator or ''}".strip() if first else "",
            "Explanation": v.explanation,
        })
    return rows


def to_dataframe(verdicts: list[Verdict]) -> pd.DataFrame:
    return pd.DataFrame(to_rows(verdicts))


def to_csv(verdicts: list[Verdict]) -> bytes:
    return to_dataframe(verdicts).to_csv(index=False).encode("utf-8")


def to_markdown(verdicts: list[Verdict]) -> str:
    lines = ["# Environmental Claim Audit", ""]
    for v in verdicts:
        lines += [f"## {v.claim.claim_id}: {v.status.value}", f"> {v.claim.text}", "",
                  f"- Calculation: {v.calculation or 'n/a'}", f"- Confidence: {v.confidence}",
                  f"- Flags: {', '.join(v.flags) or 'none'}"]
        lines += [f"- Evidence: {f.source} p.{f.page} {f.locator or ''}: `{f.quote}`" for f in v.evidence]
        lines += [f"- {v.explanation}", ""]
    return "\n".join(lines)


# ---- PDF ------------------------------------------------------------------------
_PDF_COLOURS = {"Supported": ("#2d6a4f", "#dcebdf"), "Contradicted": ("#b3261e", "#f6dcd8"),
                "Unsupported": ("#9a6a00", "#f6e8c3")}
_PDF_CSS = """
body {font-family: serif; font-size: 10pt; color: #1d2320; line-height: 1.4}
h1 {font-size: 20pt; margin: 0 0 4pt 0}
h2 {font-size: 12.5pt; margin: 14pt 0 4pt 0; padding-top: 6pt; border-top: 1px solid #d9d2c3}
.meta {font-family: monospace; font-size: 8pt; color: #6b706a; margin: 0 0 10pt 0}
.lbl {font-family: monospace; font-size: 7.5pt; color: #6b706a; margin: 8pt 0 2pt 0}
.claim {font-size: 11.5pt; font-style: italic; margin: 2pt 0 6pt 0}
.calc {font-family: monospace; font-size: 8.5pt; margin: 0}
.quote {font-family: monospace; font-size: 8.5pt; background-color: #f4efe6; padding: 4pt; margin: 2pt 0 0 0}
.cite {font-family: monospace; font-size: 7.5pt; color: #6b706a; margin: 2pt 0 4pt 0}
.expl {margin: 4pt 0 0 0}
table {border-collapse: collapse; width: 100%}
th, td {font-family: monospace; font-size: 8pt; text-align: left; padding: 2pt 4pt; border-bottom: 1px solid #d9d2c3}
th {color: #6b706a; font-weight: normal}
"""


def _e(text) -> str:
    return _html.escape("" if text is None else str(text), quote=True)


def _num(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.4f}".rstrip("0").rstrip(".")


def to_pdf_html(verdicts: list[Verdict], title: str = "Environmental Claim Audit", subtitle: str = "") -> str:
    """HTML body for the PDF. Every value is escaped: claim text and quotes come from user files."""
    counts = {s: sum(v.status.value == s for v in verdicts) for s in _PDF_COLOURS}
    greenwash = sum("greenwash" in v.flags for v in verdicts)
    meta = " · ".join(x for x in [subtitle, f"generated {date.today().isoformat()}", f"{len(verdicts)} claims",
                                  *(f"{k} {n}" for k, n in counts.items()), f"Greenwash {greenwash}"] if x)
    failed = sum("guard_failed" in v.flags for v in verdicts)
    guard = (f"{failed} claim(s) had a quote the Citation Guard could not re-find; they are marked Unsupported."
             if failed else "Every cited quote was re-found word for word in its source by the Citation Guard.")
    parts = [f"<h1>{_e(title)}</h1>", f'<p class="meta">{_e(meta)}</p>',
             f'<p class="meta">Numbers are computed in Python, not by the language model. {_e(guard)}</p>']
    for v in verdicts:
        fg, bg = _PDF_COLOURS.get(v.status.value, ("#6b706a", "#eeeeee"))
        route = v.route or ("none" if not v.evidence else "ledger")
        parts.append(
            f'<h2>{_e(v.claim.claim_id)} &#183; <span style="color:{fg};background-color:{bg}">'
            f"&#160;{_e(v.status.value)}&#160;</span></h2>"
            f'<p class="claim">&#8220;{_e(v.claim.text)}&#8221;</p>'
            f'<p class="lbl">CALCULATION (NO LLM MATH)</p><p class="calc">{_e(v.calculation or "n/a")}</p>'
            f'<p class="lbl">EVIDENCE SCORE {round((v.confidence or 0) * 100)}% &#183; VIA {_e(route).upper()}'
            f' &#183; FLAGS: {_e(", ".join(v.flags) or "none")}</p>'
        )
        if v.evidence:
            rows = "".join(
                f"<tr><td>{_e(label(f.indicator))}{' · ' + _e(f.entity) if f.entity else ''}</td><td>{f.year}</td>"
                f"<td>{_e(_num(f.value))}</td><td>{_e(f.unit)}</td>"
                f"<td>{_e(f.source)}{f' p.{f.page}' if f.page else ''} {_e(f.locator or '')}</td></tr>"
                for f in v.evidence)
            parts.append("<table><tr><th>Indicator</th><th>Year</th><th>Value</th><th>Unit</th><th>Source</th></tr>"
                         f"{rows}</table>")
            seen = set()
            for f in v.evidence:
                key = (f.source, f.page, f.quote)
                if key in seen:
                    continue
                seen.add(key)
                link = f' &#183; <a href="{_e(f.url)}">{_e(f.url)}</a>' if f.url and f.url.lower().startswith(("http://", "https://")) else ""
                parts.append(f'<p class="lbl">EVIDENCE QUOTE</p><p class="quote">{_e(f.quote)}</p>'
                             f'<p class="cite">{_e(f.source)}{f" p.{f.page}" if f.page else ""}'
                             f' {_e(f.locator or "")}{link}</p>')
        else:
            parts.append('<p class="lbl">EVIDENCE</p><p class="calc">No supporting evidence found.</p>')
        if v.explanation:
            parts.append(f'<p class="expl">{_e(v.explanation)}</p>')
    return "".join(parts)


def to_pdf(verdicts: list[Verdict], title: str = "Environmental Claim Audit", subtitle: str = "") -> bytes:
    """A4 PDF with every verdict, calculation, evidence quote and citation."""
    import pymupdf

    story = pymupdf.Story(html=to_pdf_html(verdicts, title, subtitle), user_css=_PDF_CSS)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (48, 54, -48, -54)
    doc = story.write_with_links(lambda rect_num, filled: (mediabox, where, None))
    for i, page in enumerate(doc, start=1):
        page.insert_text((48, mediabox.height - 28), f"PS 99 Citation Auditor · page {i} of {doc.page_count}",
                         fontsize=7, fontname="helv", color=(0.42, 0.44, 0.42))
    doc.set_metadata({"title": title, "creator": "PS 99 Citation Auditor"})
    try:
        doc.subset_fonts()
    except Exception:
        pass
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data
