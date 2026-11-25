r"""Evaluate the Citation Auditor against the three answer-keyed datasets.

    set PYTHONPATH=<repo root>
    .venv\Scripts\python.exe scripts/evaluate.py            # deterministic (use_llm=False), writes docs/evaluation.md
    .venv\Scripts\python.exe scripts/evaluate.py --llm      # also runs with use_llm=True and reports the provider

Datasets: PS sample (sample_report_2025.txt + evidence_2024_2025.csv), country demo
(demo_country_report.txt, no evidence files, official data), company (Greenfield PDF + PDF tables + XLSX).
Claims are matched to the answer key by the sentence in the key's "source" column when it has the form
"<file> p.N | <sentence>", otherwise by claim_id.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditor.pipeline import run_audit  # noqa: E402

S = ROOT / "data" / "samples"
STATUSES = ["Supported", "Contradicted", "Unsupported"]
DATASETS = [
    dict(name="PS sample", report=S / "sample_report_2025.txt", evidence=[S / "evidence_2024_2025.csv"],
         key=S / "answer_key.csv", official=False),
    dict(name="Country demo", report=S / "demo_country_report.txt", evidence=[],
         key=S / "demo_answer_key.csv", official=True),
    dict(name="Company (Greenfield)", report=S / "company_report_2024.pdf",
         evidence=[S / "company_evidence_tables.pdf", S / "company_evidence.xlsx"],
         key=S / "company_answer_key.csv", official=False),
]


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t).strip().lower()


def load_key(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match(key: list[dict], verdicts) -> dict[str, object]:
    """claim_id (key) -> Verdict or None."""
    by_id = {v.claim.claim_id: v for v in verdicts}
    out = {}
    for row in key:
        src = row.get("source") or ""
        v = None
        if " | " in src:
            sent = _norm(src.split(" | ", 1)[1])
            v = next((x for x in verdicts if _norm(x.claim.text) == sent), None)
        else:
            v = by_id.get(row["claim_id"])
        out[row["claim_id"]] = v
    return out


def cited(v) -> bool:
    return bool(v.evidence) and all(f.source and f.locator and f.quote for f in v.evidence)


def evaluate(ds: dict, use_llm: bool) -> dict:
    key = load_key(ds["key"])
    t0 = time.perf_counter()
    res = run_audit(ds["report"], ds["evidence"], use_llm=use_llm, use_official=ds["official"])
    secs = time.perf_counter() - t0
    m = match(key, res.verdicts)
    found = sum(v is not None for v in m.values())
    conf = Counter()
    rows, errors = [], []
    flag_total = flag_ok = 0
    for row in key:
        v = m[row["claim_id"]]
        exp = row["expected_status"]
        got = v.status.value if v else "(not extracted)"
        conf[(exp, got)] += 1
        exp_flags = [f for f in (row.get("expected_flags") or "").split(";") if f]
        got_flags = list(v.flags) if v else []
        flags_ok = all(f in got_flags for f in exp_flags)
        if exp_flags:
            flag_total += 1
            flag_ok += flags_ok
        comp_ok = True
        if row.get("expected_computed") and v is not None:
            comp_ok = v.computed_value is not None and abs(v.computed_value - float(row["expected_computed"])) <= 0.05
        status_ok = got == exp
        rows.append(dict(id=row["claim_id"], exp=exp, got=got, flags_ok=flags_ok, comp_ok=comp_ok))
        if not (status_ok and flags_ok and comp_ok):
            errors.append(dict(
                id=row["claim_id"], text=v.claim.text if v else (row.get("source") or ""), exp=exp, got=got,
                exp_flags=exp_flags, got_flags=got_flags, exp_comp=row.get("expected_computed"),
                got_comp=v.computed_value if v else None, why=row.get("why", ""),
                calc=(v.calculation if v else "") or ""))
    checked = [v for v in res.verdicts if v.status.value in ("Supported", "Contradicted")]
    sc = [v.confidence for v in checked]
    unexpected_flags = sum(
        1 for row in key if m[row["claim_id"]] is not None
        for f in m[row["claim_id"]].flags
        if f in ("greenwash", "conflict") and f not in (row.get("expected_flags") or "").split(";"))
    per = {}
    for s in STATUSES:
        tp = conf[(s, s)]
        pred = sum(n for (e, g), n in conf.items() if g == s)
        act = sum(n for (e, g), n in conf.items() if e == s)
        per[s] = dict(p=tp / pred if pred else None, r=tp / act if act else None, n=act)
    return dict(
        name=ds["name"], n_key=len(key), n_found=found, n_extracted=len(res.verdicts),
        acc=sum(r["exp"] == r["got"] for r in rows) / len(rows), per=per, conf=conf,
        flags=(flag_ok, flag_total), unexpected_flags=unexpected_flags,
        cited=(sum(map(cited, checked)), len(checked)),
        guard_fail=sum("guard_failed" in v.flags for v in res.verdicts),
        mean_conf={s: (sum(v.confidence for v in checked if v.status.value == s) /
                       max(1, sum(v.status.value == s for v in checked))) for s in STATUSES[:2]},
        n_checked=len(sc), secs=secs, per_claim=secs / max(1, len(res.verdicts)), errors=errors,
        provider=res.llm_provider, pipeline_errors=res.errors)


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render(results: list[dict], title: str) -> str:
    L = [f"### {title}", "",
         "| Dataset | Claims in key | Extracted (recall) | Status accuracy | Flag accuracy | Cited (checked verdicts) "
         "| Guard failures | Mean conf. Supported / Contradicted | Time (s) | s/claim |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        fl = f"{r['flags'][0]}/{r['flags'][1]}" if r["flags"][1] else "n/a"
        L.append(f"| {r['name']} | {r['n_key']} | {r['n_found']}/{r['n_key']} ({pct(r['n_found'] / r['n_key'])}) "
                 f"| {round(r['acc'] * r['n_key'])}/{r['n_key']} ({pct(r['acc'])}) | {fl} "
                 f"| {r['cited'][0]}/{r['cited'][1]} | {r['guard_fail']} "
                 f"| {r['mean_conf']['Supported']:.2f} / {r['mean_conf']['Contradicted']:.2f} "
                 f"| {r['secs']:.2f} | {r['per_claim']:.3f} |")
    for r in results:
        L += ["", f"**{r['name']}** per-status precision / recall (support = expected count)", "",
              "| Status | Precision | Recall | Support |", "|---|---|---|---|"]
        for s in STATUSES:
            p = r["per"][s]
            L.append(f"| {s} | {pct(p['p'])} | {pct(p['r'])} | {p['n']} |")
        cols = STATUSES + ["(not extracted)"]
        L += ["", "Confusion matrix (rows = expected, columns = got):", "",
              "| expected \\ got | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for e in STATUSES:
            L.append(f"| {e} | " + " | ".join(str(r["conf"][(e, g)]) for g in cols) + " |")
        if r["unexpected_flags"]:
            L.append(f"\nUnexpected greenwash/conflict flags on other claims: {r['unexpected_flags']}")
        if r["pipeline_errors"]:
            L.append(f"\nPipeline errors: {r['pipeline_errors']}")
    return "\n".join(L)


def errors_md(results: list[dict]) -> str:
    L = []
    for r in results:
        for e in r["errors"]:
            L.append(f"- **{r['name']} {e['id']}**: \"{e['text']}\"  \n"
                     f"  expected {e['exp']} (flags {e['exp_flags'] or 'none'}, computed {e['exp_comp'] or 'n/a'}); "
                     f"got {e['got']} (flags {e['got_flags'] or 'none'}, computed {e['got_comp']}).  \n"
                     f"  Key reasoning: {e['why']}.  Pipeline said: {e['calc']}")
    return "\n".join(L) if L else "No mismatches: every expected status, flag and computed value was reproduced."


def official_mode() -> str:
    try:
        from evidence_layer.sources import is_offline
        return ("official data in offline/snapshot-only mode (AUDITOR_OFFLINE set)" if is_offline() else
                "official data used (live API when reachable, otherwise the curated snapshots; the pipeline does "
                "not record which one served each fact)")
    except Exception:
        return "official data used (source mode unknown)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="also run with use_llm=True and report the provider")
    ap.add_argument("--out", default=str(ROOT / "docs" / "evaluation.md"))
    ap.add_argument("--no-write", action="store_true", help="print only; do not write the Markdown file")
    args = ap.parse_args()

    det = [evaluate(d, use_llm=False) for d in DATASETS]
    llm = [evaluate(d, use_llm=True) for d in DATASETS] if args.llm else []

    cmd = "python scripts/evaluate.py" + (" --llm" if args.llm else "")
    total_k = sum(r["n_key"] for r in det)
    total_ok = sum(round(r["acc"] * r["n_key"]) for r in det)
    md = ["# Evaluation: PS 99 Environmental Report Citation Auditor", "",
          f"- Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
          f"- Command: `{cmd}` (run from the repo root with `PYTHONPATH` set to the repo root)",
          "- Mode: deterministic, `use_llm=False` (regex claim parser, template explanations); reproducible.",
          f"- Official data: {official_mode()}. Only the Country demo uses it; the PS sample and the company "
          "dataset run with `use_official=False` (their claims name no country).",
          f"- Overall: {total_ok}/{total_k} claims given the expected status.", "",
          "Datasets: **PS sample** (`sample_report_2025.txt` + `evidence_2024_2025.csv`, 6 claims), "
          "**Country demo** (`demo_country_report.txt`, no evidence files, 13 claims), "
          "**Company (Greenfield)** (fictional PDF report p.2-4 + ruled-table PDF + XLSX; built by "
          "`scripts/make_company_dataset.py`, 15 claims including a PDF/XLSX conflict and two greenwash sentences).",
          "", "Matching: claims are paired with the answer key by sentence (company) or claim id (others). A claim "
          "the parser did not extract counts as wrong. \"Cited\" = every evidence fact of a Supported/Contradicted "
          "verdict has source, locator and quote. Mean confidence is over checked verdicts only.", "",
          "## Results (deterministic)", "", render(det, "Summary"), ""]
    if llm:
        provs = sorted({str(r["provider"]) for r in llm})
        md += ["## Results (`--llm`, use_llm=True)", "",
               f"LLM provider that answered: {', '.join(provs)}."
               + (" No provider answered (no API keys / no local server on this machine), so this run falls back "
                  "to the same regex parser and templates and should match the deterministic run."
                  if provs == ["None"] else ""), "", render(llm, "Summary (LLM mode)"), ""]
    md += ["## Errors and analysis", "", "Deterministic run:", "", errors_md(det), ""]
    if llm:
        md += ["LLM run:", "", errors_md(llm), ""]
    md += ["## Limitations", "",
           "- Tiny datasets (6 / 13 / 15 claims): a single claim moves accuracy by 7-17 points; treat as a smoke "
           "benchmark, not a statistical estimate.",
           "- The company report was written by us to match the documented parser phrasings, so extraction recall "
           "is optimistic; real reports are more varied.",
           "- Expected verdicts come from the project's own rules (tolerance +/-1.0 pp, 2% relative for amounts), "
           "so the key tests implementation consistency, not ground truth.", ""]
    text = "\n".join(md)
    if not args.no_write:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")

    print(f"{'Dataset':24} {'extract':>9} {'status acc':>11} {'flags':>7} {'cited':>7} {'guard':>6} {'s/claim':>8}")
    for r in det:
        fl = f"{r['flags'][0]}/{r['flags'][1]}" if r["flags"][1] else "n/a"
        print(f"{r['name']:24} {r['n_found']:>4}/{r['n_key']:<4} {pct(r['acc']):>11} {fl:>7} "
              f"{r['cited'][0]}/{r['cited'][1]:<4} {r['guard_fail']:>6} {r['per_claim']:>8.3f}")
    if llm:
        print("LLM provider:", ", ".join(sorted({str(r['provider']) for r in llm})))
    print(f"mismatches: {sum(len(r['errors']) for r in det)}" + ("" if args.no_write else f"  -> {args.out}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
