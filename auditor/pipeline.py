"""End-to-end audit, wired exactly like docs/architecture.html. Shared file.

report  -> Claim Parser -> Evidence Resolver (ledger first, RAG fallback)
        -> Verifier -> narrate -> Citation Guard -> Confidence & Greenwash
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm_router.claims import parse_report
from trust_layer.explain import narrate
from trust_layer.guard import apply_guard
from evidence_layer.ingest import ingest_file, read_pages
from evidence_layer.ledger import Ledger
from llm_router import make_router
from evidence_layer.resolver import fill_years, resolve
from evidence_layer.retrieval import HybridIndex
from auditor.schemas import Status, Verdict
from trust_layer.scoring import score
from trust_layer.verifier import verify


@dataclass
class AuditResult:
    verdicts: list[Verdict]
    ledger: Ledger
    index: HybridIndex
    llm_provider: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    report_pages: dict[int, str] = field(default_factory=dict)  # for the UI's highlighted report view

    def counts(self) -> dict[str, int]:
        return {s.value: sum(v.status == s for v in self.verdicts) for s in Status}


def _official_pages() -> dict[str, dict[int, str]]:
    """Snapshot texts of official datasets, so the Citation Guard can re-find their quotes."""
    try:
        from evidence_layer.sources import snapshot_pages

        return snapshot_pages() or {}
    except Exception:  # sources package missing or offline: official facts then fail the guard, as they should
        return {}


def _parse(pages: dict[int, str], report_year: Optional[int], llm) -> list:
    try:
        from llm_router.claims import parse_pages  # page-aware parser (sets Claim.page)
    except ImportError:
        return parse_report("\n".join(pages.values()), report_year, llm)
    return parse_pages(pages, report_year, llm)


def run_audit(report_path: str | Path, evidence_paths: list[str | Path],
              report_year: Optional[int] = None, use_llm: bool = True,
              use_official: bool = True) -> AuditResult:
    llm = make_router() if use_llm else None
    ledger, index = Ledger(), HybridIndex()
    source_pages: dict[str, dict[int, str]] = {}
    errors: list[str] = []

    for path in evidence_paths:
        try:
            facts, chunks = ingest_file(path)
            ledger.add(facts)
            index.add(chunks)
            source_pages[Path(path).name] = read_pages(path)
        except Exception as exc:  # one bad file must not kill the audit
            errors.append(f"{Path(path).name}: {exc}")

    uploaded = set(source_pages)  # official snapshot text must never replace an uploaded file's text
    report_pages = read_pages(report_path)
    verdicts = []
    for claim in _parse(report_pages, report_year, llm):
        claim = fill_years(claim, ledger)
        evidence = resolve(claim, ledger, index, llm, use_official=use_official)
        verdict = verify(claim, evidence)
        verdict.route = evidence.route
        verdict.explanation = narrate(verdict, llm)
        if any(f.source not in uploaded for f in verdict.evidence):
            # Refresh every time: each live fetch adds lines to a dataset's snapshot text, so text
            # cached for an earlier claim would miss this claim's quotes (Italy after Germany, ...).
            for name, pages in _official_pages().items():
                if name not in uploaded:
                    source_pages[name] = pages
        verdict = apply_guard(verdict, source_pages)
        verdicts.append(score(verdict))

    return AuditResult(verdicts, ledger, index, llm.last_provider if llm else None, errors, report_pages)
