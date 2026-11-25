"""Build the curated official-data snapshots in data/official/ for the offline demo.

    .venv\\Scripts\\python.exe scripts\\fetch_official_data.py

Pulls every mapped series of every source in auditor.sources for the demo
entities and years, writes data/official/<source>.csv (one canonical line per
series/entity/year: the same line that becomes a Fact quote) and
data/official/manifest.json. If a source is unreachable, its previous snapshot
rows are kept, so re-running offline never erases data.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditor.sources import CONNECTORS  # noqa: E402
from auditor.sources._common import OFFICIAL_DIR, snapshot_rows, write_snapshot  # noqa: E402

ENTITIES = ["DEU", "IND", "USA", "GBR", "FRA", "CHN", "BRA", "WLD", "EUU"]
YEARS = [1990, 2000, 2005, *range(2010, 2024)]   # common baselines + 2010-2023
MAX_BYTES = 2 * 1024 * 1024


def _pull(connector, series, entity):
    try:
        rows = [r for r in connector.live_rows(series, entity) if r.entity == entity and r.year in YEARS]
        return connector, series, entity, rows, None
    except Exception as exc:
        return connector, series, entity, None, f"{type(exc).__name__}: {str(exc)[:120]}"


def main() -> int:
    jobs = [(c, s, e) for c in CONNECTORS for s in sorted({s for ss in c.SERIES.values() for s in ss})
            for e in ENTITIES if c.supports(e)]
    print(f"Fetching {len(jobs)} series x entity pairs from {len(CONNECTORS)} sources ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: _pull(*job), jobs))

    manifest = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "entities": ENTITIES, "years": YEARS, "line_format": "series,entity,year,value,unit",
                "sources": []}
    for connector in CONNECTORS:
        mine = [r for r in results if r[0] is connector]
        old = snapshot_rows(connector.slug)
        rows, failures, empty = [], [], []
        for _, series, entity, got, err in mine:
            if got is None:
                kept = [r for r in old if r.series == series and r.entity == entity]
                rows += kept
                failures.append(f"{series}/{entity}: {err} (kept {len(kept)} old rows)")
            elif not got:
                empty.append(f"{series}/{entity}")
            else:
                rows += got
        path = write_snapshot(connector.slug, rows)
        manifest["sources"].append({
            "name": connector.name, "file": path.name, "endpoint": connector.endpoint, "home": connector.home,
            "indicators": {ind: [{"series": s, "unit": connector.UNITS.get(s, "")} for s in ss]
                           for ind, ss in connector.SERIES.items()},
            "rows": len({r.line() for r in rows}),
            "entities": sorted({r.entity for r in rows}),
            "no_data": empty, "failures": failures,
        })
        status = "OK" if not failures else f"{len(failures)} failed"
        print(f"  {connector.name:32s} {len(rows):5d} rows  entities={','.join(sorted({r.entity for r in rows}))}  {status}")
        for f in failures[:5]:
            print(f"      ! {f}")

    (OFFICIAL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    total = sum(p.stat().st_size for p in OFFICIAL_DIR.iterdir() if p.is_file())
    print(f"data/official: {total / 1024:.1f} KB")
    if total > MAX_BYTES:
        print("WARNING: data/official is over 2 MB")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
