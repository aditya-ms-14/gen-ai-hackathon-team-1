"""Streamlit UI: 1 · Upload -> 2 · Audit -> 3 · Export, following the team's HTML design.

Run:  streamlit run ui/app.py   (from the repo root, so .streamlit/config.toml is picked up)
Streamlit Community Cloud: main file path = ui/app.py.
The audit view (report with clickable claims + claim detail) is one HTML component: ui/audit_view.py.
"""
from __future__ import annotations

import html
import inspect
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]  # repo root: makes auditor/, evidence_layer/, ... importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from auditor.pipeline import run_audit  # noqa: E402
from evidence_layer.ingest import read_pages  # noqa: E402
from ui import audit_view, report, theme  # noqa: E402
RUNTIME = ROOT / "data" / "runtime"
SAMPLES = ROOT / "data" / "samples"
SAMPLE_REPORT, SAMPLE_EVIDENCE = SAMPLES / "sample_report_2025.txt", SAMPLES / "evidence_2024_2025.csv"
DEMO_REPORT = SAMPLES / "demo_country_report.txt"
VIEWS = {"upload": "1 · Upload", "audit": "2 · Audit", "export": "3 · Export"}
STEPS = ["Ingest evidence", "Build ledger + index", "Parse claims", "Resolve evidence", "Verify + score"]
PROVIDERS = {"gemini": "Gemini API", "groq": "Groq", "lmstudio": "LM Studio"}

st.set_page_config(page_title="Citation Auditor · PS 99", page_icon="🌿", layout="wide")
THEME = theme.theme_type()
theme.inject(THEME)
st.session_state.setdefault("view", "upload")


# ---- helpers ------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def set_view(view: str) -> None:
    st.session_state["view"] = view


def supports_official() -> bool:
    try:
        return "use_official" in inspect.signature(run_audit).parameters
    except (TypeError, ValueError):
        return False


@st.cache_data(ttl=60, show_spinner=False)
def llm_status() -> dict[str, str]:
    try:
        from llm_router import make_router

        return {str(k): str(v) for k, v in make_router().status().items()}
    except Exception as exc:  # router mid-edit or misconfigured: say so, don't crash the page
        return {"llm": f"status unavailable ({exc.__class__.__name__})"}


def model_line(use_llm: bool) -> str:
    if not use_llm:
        return "Model: LLM off · regex claim parser + template narration"
    parts = [f"{PROVIDERS.get(k, k)} ({v})" for k, v in llm_status().items()] or ["none configured"]
    return "Model: " + parts[0] + ("" if len(parts) == 1 else " · fallback " + " · ".join(parts[1:]))


def status_counts(verdicts) -> dict[str, int]:
    counts = {s: sum(v.status.value == s for v in verdicts) for s in ("Supported", "Contradicted", "Unsupported")}
    counts["Greenwash"] = sum("greenwash" in v.flags for v in verdicts)
    return counts


def stat_tile(cls: str, n: int, label: str) -> str:
    attr = f' class="{cls}"' if cls else ""
    return f'<div class="ps-stat"><b{attr}>{n}</b><span>{esc(label)}</span></div>'


def _size(obj) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def steps_html(result=None, running: bool = False) -> str:
    if result is None:
        first = "● " + STEPS[0] + (" · running…" if running else "")
        cells = f'<div class="a">{esc(first)}</div>' + "".join(f"<div>{esc(s)}</div>" for s in STEPS[1:])
        return f'<div class="ps-steps" aria-label="Pipeline progress">{cells}</div>'
    verdicts = result.verdicts
    counts = status_counts(verdicts)
    n_files = len(st.session_state.get("evidence_paths", []))
    facts, chunks = _size(getattr(result, "ledger", None)), _size(getattr(result, "index", None))
    details = [
        f"{n_files} file(s)" if n_files else "official data",
        " · ".join(x for x in [f"{facts} facts" if facts is not None else "",
                               f"{chunks} chunks" if chunks is not None else ""] if x),
        f"{len(verdicts)} claims",
        f"{sum(bool(v.evidence) for v in verdicts)} with evidence",
        f"{counts['Supported']} / {counts['Contradicted']} / {counts['Unsupported']}",
    ]
    cells = "".join(f'<div class="d">✓ {esc(s)}<small>{esc(d)}</small></div>' for s, d in zip(STEPS, details))
    return f'<div class="ps-steps" aria-label="Pipeline progress">{cells}</div>'


def save_upload(upload, folder: str) -> Path:
    target = RUNTIME / folder
    target.mkdir(parents=True, exist_ok=True)
    path = target / Path(upload.name).name  # never trust a client path
    path.write_bytes(upload.getvalue())
    return path


def start_audit(report_path: Path, evidence_paths: list[Path], label: str, official: bool = False) -> None:
    params = inspect.signature(run_audit).parameters
    takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
    year = int(st.session_state.get("report_year", 0) or 0)
    kwargs = {"report_year": year or None, "use_llm": bool(st.session_state.get("use_llm", True))}
    if official and "use_official" in params:
        kwargs["use_official"] = True
    kwargs = {k: v for k, v in kwargs.items() if k in params or takes_any}
    steps_slot.markdown(steps_html(running=True), unsafe_allow_html=True)
    try:
        with st.spinner("Extracting claims, resolving evidence, verifying…"):
            result = run_audit(report_path, evidence_paths, **kwargs)
    except Exception as exc:  # show the failure, keep the app alive
        steps_slot.markdown(steps_html(), unsafe_allow_html=True)
        st.error(f"Audit failed: {exc}")
        return
    if official and "use_official" not in params:
        label += " (official data not wired into run_audit yet)"
    st.session_state.update(result=result, report_path=str(report_path), run_label=label,
                            evidence_paths=[str(p) for p in evidence_paths],
                            run_id=st.session_state.get("run_id", 0) + 1, view="audit")
    for key in ("payload", "pdf"):
        st.session_state.pop(key, None)
    st.rerun()


def get_payload(result) -> dict:
    run_id = st.session_state.get("run_id", 0)
    cached = st.session_state.get("payload")
    if cached and cached[0] == run_id:
        return cached[1]
    report_path = Path(st.session_state.get("report_path", ""))
    pages = getattr(result, "report_pages", None)  # if the pipeline ever carries them
    if not pages:
        try:
            pages = read_pages(report_path)
        except Exception:
            pages = {}
    sources = {Path(p).name: Path(p) for p in st.session_state.get("evidence_paths", [])}
    payload = audit_view.build_payload(result.verdicts, pages, sources, report_path.name,
                                       is_pdf=report_path.suffix.lower() == ".pdf")
    st.session_state["payload"] = (run_id, payload)
    return payload


def get_pdf(result) -> Optional[bytes]:
    run_id = st.session_state.get("run_id", 0)
    cached = st.session_state.get("pdf")
    if cached and cached[0] == run_id:
        return cached[1]
    try:
        data = report.to_pdf(result.verdicts, subtitle=Path(st.session_state.get("report_path", "")).name)
    except Exception as exc:
        st.error(f"PDF export failed: {exc}")
        data = None
    st.session_state["pdf"] = (run_id, data)
    return data


def no_result() -> None:
    md('<h1 class="ps-h1">No audit yet.</h1>'
       '<p class="ps-sub">Upload a report and its evidence, or run the problem-statement sample.</p>')
    st.button("← 1 · Upload", key="to_upload", on_click=set_view, args=("upload",))


# ---- header + nav -------------------------------------------------------------------
with st.container(key="ps_header"):
    brand, nav = st.columns([3, 2], vertical_alignment="center")
    brand.markdown('<div class="ps-brand">Citation Auditor<small>PS 99 · environmental reports</small></div>',
                   unsafe_allow_html=True)
    with nav.container(horizontal=True, horizontal_alignment="right", gap="small"):
        for view_key, view_label in VIEWS.items():
            st.button(view_label, key=f"nav_{view_key}", on_click=set_view, args=(view_key,),
                      type="primary" if st.session_state["view"] == view_key else "secondary")

view = st.session_state["view"]
result = st.session_state.get("result")

# ---- 1 · Upload ---------------------------------------------------------------------
if view == "upload":
    md('<h1 class="ps-h1">Check every claim against its evidence.</h1>'
       '<p class="ps-sub">Upload the draft report and the evidence files behind it. '
       "Numbers are verified by code, not by the language model.</p>")
    left, right = st.columns(2, gap="medium")
    with left.container(key="drop_report"):
        md('<div class="ps-drop"><div class="ps-drop-icon">▤</div><b>Draft report</b>'
           '<p class="ps-mono ps-mute">PDF · TXT · MD · drag here</p></div>')
        report_file = st.file_uploader("Draft report", type=["pdf", "txt", "md"], key="report_file",
                                       label_visibility="collapsed")
    with right.container(key="drop_evidence"):
        md('<div class="ps-drop"><div class="ps-drop-icon">▦</div><b>Evidence files</b>'
           '<p class="ps-mono ps-mute">PDF · XLSX · CSV · TXT · multiple</p></div>')
        evidence_files = st.file_uploader("Evidence files", type=["csv", "xlsx", "pdf", "txt"],
                                          accept_multiple_files=True, key="evidence_files",
                                          label_visibility="collapsed")

    steps_slot = st.empty()
    steps_slot.markdown(steps_html(result), unsafe_allow_html=True)

    with st.container(horizontal=True, vertical_alignment="center", key="run_row"):
        go = st.button("Verify all claims →", type="primary", key="verify")
        run_sample = st.button("Run problem-statement sample", key="run_sample")
        run_demo = st.button("Run country demo", key="run_demo") if DEMO_REPORT.exists() else False
        md(f'<span class="ps-mono ps-mute">{esc(model_line(st.session_state.get("use_llm", True)))}</span>')

    with st.expander("Settings"):
        st.toggle("Use the LLM for claim parsing and narration (Gemini → LM Studio)", value=True, key="use_llm")
        st.number_input("Report year (0 = detect from the report)", min_value=0, max_value=2100, value=0,
                        key="report_year")

    if go:
        if report_file and evidence_files:
            start_audit(save_upload(report_file, "report"), [save_upload(f, "evidence") for f in evidence_files],
                        label=Path(report_file.name).name)
        elif report_file and supports_official():
            start_audit(save_upload(report_file, "report"), [], label=f"{Path(report_file.name).name} · official data",
                        official=True)
        else:
            st.warning("Upload a report and at least one evidence file, or run the sample.")
    elif run_sample:
        start_audit(SAMPLE_REPORT, [SAMPLE_EVIDENCE], label="problem-statement sample")
    elif run_demo:
        start_audit(DEMO_REPORT, [], label="country demo · official data", official=True)

# ---- 2 · Audit ----------------------------------------------------------------------
elif view == "audit":
    if result is None:
        no_result()
    else:
        for err in getattr(result, "errors", None) or []:
            st.error(err)
        verdicts = result.verdicts
        counts = status_counts(verdicts)
        tiles = [("", len(verdicts), "Claims found"), ("ok", counts["Supported"], "Supported"),
                 ("bad", counts["Contradicted"], "Contradicted"), ("warn", counts["Unsupported"], "Unsupported"),
                 ("warn", counts["Greenwash"], "Greenwash")]
        md('<div class="ps-stats">' + "".join(stat_tile(*t) for t in tiles) + "</div>")
        facts, chunks = _size(getattr(result, "ledger", None)), _size(getattr(result, "index", None))
        provider = getattr(result, "llm_provider", None) or "none (regex parser + templates)"
        md(f'<p class="ps-mono ps-mute">{esc(st.session_state.get("run_label", ""))} · LLM used: {esc(provider)}'
           + (f" · ledger facts: {facts}" if facts is not None else "")
           + (f" · index chunks: {chunks}" if chunks is not None else "") + "</p>")

        payload = get_payload(result)
        # Our own template; user text inside it is JSON-escaped and rendered with textContent (ui/audit_view.py).
        # st.iframe replaces st.components.v1.html, which newer Streamlit versions remove.
        view_html, view_height = audit_view.render(payload, THEME), audit_view.component_height(payload)
        if hasattr(st, "iframe"):
            st.iframe(view_html, height=view_height)
        else:
            components.html(view_html, height=view_height, scrolling=True)

        md('<div class="ps-h3">All verdicts</div>')
        st.dataframe(report.to_dataframe(verdicts), hide_index=True, width="stretch")

# ---- 3 · Export ---------------------------------------------------------------------
else:
    if result is None:
        no_result()
    else:
        md('<h1 class="ps-h1">Audit report ready.</h1>'
           '<p class="ps-sub">PDF, CSV or Markdown with every verdict, calculation and source citation.</p>')
        verdicts = result.verdicts
        stem = Path(st.session_state.get("report_path", "audit")).stem or "audit"
        rows = [
            ("PDF", "Narrated verdicts, calculations, evidence quotes, citations and source links",
             get_pdf(result), f"{stem}_audit.pdf", "application/pdf", "dl_pdf", "primary"),
            ("CSV", "Claim · verdict · evidence · score", report.to_csv(verdicts), f"{stem}_audit.csv",
             "text/csv", "dl_csv", "secondary"),
            ("Markdown", "The full audit as plain text", report.to_markdown(verdicts), f"{stem}_audit.md",
             "text/markdown", "dl_md", "secondary"),
        ]
        spec = [1, 3.4, 1.2]
        with st.container(key="export_panel"):
            head = st.columns(spec)
            for col, text in zip(head, ["Format", "Contents", ""]):
                col.markdown(f'<div class="ps-th">{text or "&nbsp;"}</div>', unsafe_allow_html=True)
            for fmt, contents, data, name, mime, key, kind in rows:
                c1, c2, c3 = st.columns(spec, vertical_alignment="center")
                c1.markdown(f'<div class="ps-td">{esc(fmt)}</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="ps-td">{esc(contents)}</div>', unsafe_allow_html=True)
                if data is None:
                    c3.markdown('<div class="ps-td ps-mute">unavailable</div>', unsafe_allow_html=True)
                else:
                    c3.download_button("Download", data, file_name=name, mime=mime, key=key, type=kind,
                                       on_click="ignore")
        counts = status_counts(verdicts)
        md('<p class="ps-mono ps-mute" style="margin-top:14px">'
           f'{len(verdicts)} claims · <span class="ps-pill ok">{counts["Supported"]} Supported</span> '
           f'<span class="ps-pill bad">{counts["Contradicted"]} Contradicted</span> '
           f'<span class="ps-pill warn">{counts["Unsupported"]} Unsupported</span> · '
           f'{counts["Greenwash"]} greenwash</p>')
