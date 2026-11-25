"""Audit view: report text with clickable claims + claim detail, as one HTML component. Owner: UI.

build_payload() turns verdicts into plain JSON data (all text stays text; the
browser inserts it with textContent, never innerHTML). render() injects that
JSON into ui/audit_component.html with <, > and & escaped, so text from
uploaded files can never close the <script> tag or become markup.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from auditor.schemas import ExtractionMethod, Fact, Verdict

from .theme import FONTS_URL, PALETTES

UI_DIR = Path(__file__).resolve().parent
TEMPLATE = UI_DIR / "audit_component.html"
STATUS_CLASS = {"Supported": "ok", "Contradicted": "bad", "Unsupported": "warn"}
TABLE_EXTS = {".csv", ".xlsx", ".xls"}
MAX_PAGE_IMAGES = 12
MAX_PAGE_TEXT = 6000
PANEL_MAX = 720  # px; the iframe is a little taller

# A number that is not glued to letters ("tCO2e" is not 2), optionally scaled by a word.
_NUM = re.compile(r"(?<![A-Za-z\d.])(\d[\d,]*(?:\.\d+)?)(?:\s*(million|billion|thousand)\b)?", re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


# ---- small helpers -----------------------------------------------------------
def status_class(v: Verdict) -> str:
    return STATUS_CLASS.get(v.status.value, "warn")


def infer_route(v: Verdict) -> str:
    """verdict.route when the pipeline sets it, otherwise inferred from the evidence."""
    if getattr(v, "route", None):
        return str(v.route)
    if not v.evidence:
        return "none"
    if any(f.method == ExtractionMethod.LLM for f in v.evidence):
        return "rag"
    return "ledger"


def fmt_value(x: Optional[float]) -> str:
    if x is None:
        return ""
    if float(x).is_integer():
        return f"{int(x):,}"
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def _label(indicator: Optional[str]) -> str:
    try:
        from auditor.indicators import label

        return label(indicator) or (indicator or "")
    except Exception:
        return indicator or ""


def _words_pattern(text: str) -> Optional[re.Pattern]:
    words = text.split()
    return re.compile(r"\s+".join(re.escape(w) for w in words), re.I) if words else None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def mark_values(text: str, values: Iterable[float], base: Optional[dict] = None) -> list[dict]:
    """Split text into segments, marking every number equal to one of `values`."""
    vals = [float(v) for v in values if v is not None]
    extra = base or {}

    def hit(number: str, scale: Optional[str] = None) -> bool:
        try:
            x = float(number.replace(",", ""))
        except ValueError:
            return False
        x *= _SCALE[scale.lower()] if scale else 1.0
        return any(_close(x, v) for v in vals)

    spans: list[tuple[int, int]] = []
    for m in _NUM.finditer(text):
        if m.group(2) and hit(m.group(1), m.group(2)):    # "4.9 million"
            spans.append((m.start(), m.end()))
        elif hit(m.group(1)):                               # "1,000" or "1000"
            spans.append((m.start(1), m.end(1)))
        elif "," in m.group(1):                              # CSV row "1000,700": check each cell
            offset = m.start(1)
            for part in m.group(1).split(","):
                if part and hit(part):
                    spans.append((offset, offset + len(part)))
                offset += len(part) + 1

    segs: list[dict] = []
    pos = 0
    for start, end in spans:
        if start > pos:
            segs.append({"t": text[pos:start], **extra})
        segs.append({"t": text[start:end], "m": 1, **extra})
        pos = end
    if pos < len(text):
        segs.append({"t": text[pos:], **extra})
    return segs


# ---- report text with claim spans ------------------------------------------
def _unwrap_pdf(text: str) -> str:
    """Join PDF hard-wrapped lines; keep a break after sentence-final punctuation."""
    text = text.replace("\r\n", "\n")
    return re.sub(r"(?<![.!?:;\n])[ \t]*\n(?![ \t]*\n)", " ", text)


def locate_claims(pages: dict[int, str], verdicts: list[Verdict]) -> tuple[dict[int, list[tuple[int, int, int]]], list[int]]:
    """Find each claim's text in the report. Returns {page: [(start, end, claim_index)]}, unplaced indexes."""
    spans: dict[int, list[tuple[int, int, int]]] = {p: [] for p in pages}
    unplaced: list[int] = []
    for idx, v in enumerate(verdicts):
        text = (v.claim.text or "").strip()
        variants = [text, text.rstrip(".;:,!")]
        order = list(pages)
        if v.claim.page in pages:
            order.remove(v.claim.page)
            order.insert(0, v.claim.page)
        placed = False
        for variant in dict.fromkeys(variants):
            pat = _words_pattern(variant)
            if not pat:
                continue
            for page in order:
                for m in pat.finditer(pages[page]):
                    if all(m.end() <= s or m.start() >= e for s, e, _ in spans[page]):
                        spans[page].append((m.start(), m.end(), idx))
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            unplaced.append(idx)
    return {p: sorted(s) for p, s in spans.items()}, unplaced


def _paragraphs(text: str, spans: list[tuple[int, int, int]]) -> list[list[dict]]:
    pieces: list[dict] = []
    pos = 0
    for start, end, idx in spans:
        if start > pos:
            pieces.append({"t": text[pos:start]})
        pieces.append({"t": text[start:end], "c": idx})
        pos = end
    if pos < len(text):
        pieces.append({"t": text[pos:]})

    paras: list[list[dict]] = [[]]
    for piece in pieces:
        if "c" in piece:
            paras[-1].append(piece)
            continue
        for k, part in enumerate(re.split(r"\n[ \t]*\n", piece["t"])):
            if k:
                paras.append([])
            if part:
                paras[-1].append({"t": part})
    out = []
    for para in paras:
        if not para or all("c" not in s and not s["t"].strip() for s in para):
            continue
        if "c" not in para[0]:
            para[0] = {"t": para[0]["t"].lstrip()}
        if "c" not in para[-1]:
            para[-1] = {"t": para[-1]["t"].rstrip()}
        out.append([s for s in para if s["t"]])
    return out


def report_block(report_pages: dict[int, str], verdicts: list[Verdict], is_pdf: bool = False,
                 name: str = "") -> tuple[dict, list[int], dict[int, int]]:
    pages = {int(p): (_unwrap_pdf(t) if is_pdf else t.replace("\r\n", "\n")) for p, t in (report_pages or {}).items()}
    spans, unplaced = locate_claims(pages, verdicts)
    claim_page = {idx: p for p, s in spans.items() for _, _, idx in s}
    block = {"name": name, "multi": len(pages) > 1,
             "pages": [{"page": p, "paras": _paragraphs(pages[p], spans[p])} for p in sorted(pages)]}
    return block, unplaced, claim_page


# ---- evidence groups, snippets, source pages ---------------------------------
def _find_quote(page_text: str, quote: str) -> Optional[tuple[int, int]]:
    pat = _words_pattern(quote)
    m = pat.search(page_text) if pat else None
    return (m.start(), m.end()) if m else None


def _snippet(page_text: str, quote: str, values: list[float], is_table: bool) -> Optional[dict]:
    """Two lines of context around the quote (tables keep their header row)."""
    span = _find_quote(page_text, quote)
    if not span:
        return None
    start, end = span
    before_lines = page_text[:start].split("\n")  # last item: text on the quote's own line
    after_lines = page_text[end:].split("\n")
    keep_before, lead = before_lines[-3:], len(before_lines) > 3
    if is_table and lead:  # keep the header row so the columns (years) stay readable
        keep_before = [before_lines[0], "…", *keep_before] if len(before_lines) > 4 else before_lines
        lead = False
    keep_after, tail = after_lines[:3], len(after_lines) > 3
    before = "\n".join(line if len(line) <= 220 else "…" + line[-220:] for line in keep_before)
    after = "\n".join(line if len(line) <= 220 else line[:220] + "…" for line in keep_after)
    segs = [{"t": before}] if before else []
    segs += mark_values(page_text[start:end], values, {"q": 1})
    if after:
        segs.append({"t": after})
    return {"lead": lead, "tail": tail, "segs": segs}


def _page_text_view(page_text: str, quote: str, values: list[float]) -> Optional[dict]:
    if not page_text:
        return None
    span = _find_quote(page_text, quote)
    if span and len(page_text) > MAX_PAGE_TEXT:
        lo = max(0, span[0] - MAX_PAGE_TEXT // 2)
        page_text, span = page_text[lo:lo + MAX_PAGE_TEXT], (span[0] - lo, span[1] - lo)
    elif len(page_text) > MAX_PAGE_TEXT:
        page_text = page_text[:MAX_PAGE_TEXT]
    if not span:
        return {"kind": "text", "segs": [{"t": page_text}]}
    s, e = span
    return {"kind": "text", "segs": [{"t": page_text[:s]}, *mark_values(page_text[s:e], values, {"q": 1}),
                                     {"t": page_text[e:]}]}


def render_pdf_page(path: Path, page: int, needles: list[str], zoom: float = 1.4) -> Optional[str]:
    """Cited PDF page as a base64 JPEG data URI, with the quote highlighted (in memory only)."""
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            if not 1 <= page <= doc.page_count:
                return None
            pg = doc[page - 1]
            for needle in needles:
                if not needle.strip():
                    continue
                rects = pg.search_for(needle)[:20]
                for rect in rects:
                    pg.add_highlight_annot(rect)
                if rects and needle is needles[0]:
                    break  # the full quote was found; skip the per-value fallback
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
            data = pix.tobytes("jpeg", jpg_quality=75)
    except Exception:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _groups(v: Verdict, sources: dict[str, Path], source_pages: dict[str, dict[int, str]],
            images: dict[str, str]) -> list[dict]:
    grouped: dict[tuple, list[Fact]] = {}
    for f in v.evidence:
        grouped.setdefault((f.source, f.page, f.quote), []).append(f)
    out = []
    for (source, page, quote), facts in grouped.items():
        values = [f.value for f in facts]
        path = sources.get(source)
        is_table = bool(path and path.suffix.lower() in TABLE_EXTS)
        pages = source_pages.get(source) or {}
        page_text = (pages.get(page) if page else "\n".join(pages.values())) or ""
        locators = list(dict.fromkeys(f.locator for f in facts if f.locator))

        url = next((f.url for f in facts if f.url and re.match(r"^https?://", f.url, re.I)), None)
        open_target: Optional[dict] = None
        if url:
            open_target = {"kind": "url", "href": url}
        elif path and path.suffix.lower() == ".pdf" and page:
            key = f"{source}#p{page}"
            if key not in images and len(images) < MAX_PAGE_IMAGES:
                needles = [quote] + [fmt_value(x) for x in values] + [str(x) for x in values]
                if (uri := render_pdf_page(path, page, list(dict.fromkeys(needles)))):
                    images[key] = uri
            if key in images:
                open_target = {"kind": "image", "key": key}
        if open_target is None and page_text:
            open_target = _page_text_view(page_text, quote, values)

        out.append({
            "source": source, "page": page, "pageLabel": f"p.{page}" if page else "—",
            "locator": ", ".join(locators), "quote": quote,
            "quoteSegs": mark_values(quote, values),
            "snippet": _snippet(page_text, quote, values, is_table) if page_text else None,
            "open": open_target,
        })
    return out


def _rows(v: Verdict) -> list[list[str]]:
    rows = []
    for f in v.evidence:
        ind = _label(f.indicator) + (f" · {f.entity}" if f.entity else "")
        cite = " · ".join(x for x in [f.source, f"p.{f.page}" if f.page else "", f.locator or ""] if x)
        rows.append([ind, str(f.year), fmt_value(f.value), f.unit or "", cite])
    return rows


def build_payload(verdicts: list[Verdict], report_pages: dict[int, str],
                  sources: Optional[dict[str, Path]] = None, report_name: str = "",
                  is_pdf: bool = False) -> dict:
    """Everything the component shows, as plain JSON-able data."""
    sources = {k: Path(p) for k, p in (sources or {}).items()}
    source_pages: dict[str, dict[int, str]] = {}
    for name, path in sources.items():
        try:
            from evidence_layer.ingest import read_pages

            source_pages[name] = read_pages(path)
        except Exception:
            source_pages[name] = {}

    block, unplaced, claim_page = report_block(report_pages, verdicts, is_pdf, report_name)
    images: dict[str, str] = {}
    claims = []
    for idx, v in enumerate(verdicts):
        claims.append({
            "id": v.claim.claim_id, "text": v.claim.text, "status": v.status.value, "s": status_class(v),
            "route": infer_route(v), "score": float(v.confidence or 0.0),
            "calc": v.calculation or "", "explanation": v.explanation or "",
            "flags": list(v.flags), "rows": _rows(v),
            "groups": _groups(v, sources, source_pages, images),
            "guard": "failed" if "guard_failed" in v.flags else ("passed" if v.evidence else "n/a"),
            "page": v.claim.page or (claim_page.get(idx) if block["multi"] else None),
        })
    return {"report": block, "claims": claims, "unplaced": unplaced, "images": images}


# ---- HTML ---------------------------------------------------------------------
def json_for_script(obj) -> str:
    """JSON that is safe inside <script>: no <, > or & characters survive."""
    s = json.dumps(obj, ensure_ascii=False)
    return (s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
             .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _vars(kind: str) -> str:
    return ";".join(f"--{k}:{v}" for k, v in PALETTES[kind].items())


def render(payload: dict, theme: Optional[str] = None) -> str:
    attr = f' data-theme="{theme}"' if theme in PALETTES else ""
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__THEME_ATTR__", attr)
            .replace("__FONTS_URL__", FONTS_URL)
            .replace("__LIGHT_VARS__", _vars("light"))
            .replace("__DARK_VARS__", _vars("dark"))
            .replace("__PANEL_MAX__", str(PANEL_MAX))
            .replace("__DATA__", json_for_script(payload)))


def component_height(payload: dict) -> int:
    return PANEL_MAX + 40
