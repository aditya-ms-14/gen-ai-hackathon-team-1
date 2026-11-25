r"""Run the industry reports with the extract CSV, the full WDI bulk CSV and official data only; compare verdicts.

    .venv\Scripts\python.exe scripts/compare_industry_evidence.py <full WDI bulk csv>
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from auditor.pipeline import run_audit  # noqa: E402

D = ROOT / "data" / "samples" / "industry"
modes = {"extract": ([D / "worldbank_environment_extract.csv"], False), "full": ([Path(sys.argv[1])], False),
         "official": ([], True)}
for pdf in ("industry_report_1_india.pdf", "industry_report_2_europe.pdf"):
    res = {}
    for name, (ev, off) in modes.items():
        t = time.perf_counter()
        r = run_audit(D / pdf, ev, use_llm=False, use_official=off)
        res[name] = [(v.claim.text, v.status.value, v.computed_value, tuple(v.flags), v.route) for v in r.verdicts]
        print(f"{pdf} {name}: {time.perf_counter() - t:.1f}s", {s: sum(x[1] == s for x in res[name]) for s in ("Supported", "Contradicted", "Unsupported")})
    for name in ("full", "official"):
        for a, b in zip(res["extract"], res[name]):
            if a[1:4] != b[1:4]:
                print(f"  DIFF {name}: {a[0][:70]} | extract {a[1:4]} vs {name} {b[1:4]} route {b[4]}")
