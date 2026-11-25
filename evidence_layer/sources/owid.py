"""Our World in Data: owid/co2-data and owid/energy-data on GitHub. Owner: Aditya (official data sources).

The full CSVs (~15 MB and ~9 MB) are downloaded once into data/runtime/cache/owid/
and reduced to the columns below (years >= 1990).
"""
from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

from evidence_layer.sources._common import CACHE_DIR, Connector, Row, _fresh, download

DATASETS = {
    "co2": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
    "energy": "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
}
COLUMNS = {  # series (= OWID column) -> (dataset, unit, grapher chart slug)
    "co2": ("co2", "Mt CO2", "annual-co2-emissions-per-country"),
    "total_ghg_excluding_lucf": ("co2", "Mt CO2e", "total-ghg-emissions-excluding-lufc"),
    "renewables_share_energy": ("energy", "%", "renewable-share-energy"),
    "renewables_share_elec": ("energy", "%", "share-electricity-renewables"),
    "primary_energy_consumption": ("energy", "TWh", "primary-energy-cons"),
}
_AGG_NAMES = {"World": "WLD", "European Union (27)": "EUU"}
_GRAPHER_CODE = {"WLD": "OWID_WRL", "EUU": "OWID_EU27"}
_lock = threading.Lock()
_tables: dict[str, dict] = {}


def _reduce(csv_path: Path, out: Path, columns: list[str]) -> dict:
    table: dict[str, dict[str, dict[str, float]]] = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            entity = _AGG_NAMES.get(rec.get("country", "")) or rec.get("iso_code", "")
            if not entity or entity.startswith("OWID") or len(entity) != 3:
                continue
            try:
                year = int(rec["year"])
            except (KeyError, ValueError):
                continue
            if year < 1990:
                continue
            for col in columns:
                if rec.get(col) not in (None, ""):
                    table.setdefault(col, {}).setdefault(entity, {})[str(year)] = float(rec[col])
    out.write_text(json.dumps(table), encoding="utf-8")
    return table


def _table(dataset: str) -> dict:
    with _lock:
        reduced = CACHE_DIR / "owid" / f"{dataset}_reduced.json"
        if dataset in _tables and _fresh(reduced):
            return _tables[dataset]
        if _fresh(reduced):
            _tables[dataset] = json.loads(reduced.read_text(encoding="utf-8"))
            return _tables[dataset]
        raw = download("owid", DATASETS[dataset], f"owid-{dataset}-data.csv")
        cols = [c for c, spec in COLUMNS.items() if spec[0] == dataset]
        _tables[dataset] = _reduce(raw, reduced, cols)
        return _tables[dataset]


class OurWorldInData(Connector):
    name = "Our World in Data"
    slug = "our_world_in_data"
    home = "https://github.com/owid/co2-data"
    endpoint = "; ".join(DATASETS.values())
    SERIES = {
        "co2_emissions": ["co2"],
        "ghg_emissions": ["total_ghg_excluding_lucf"],
        "renewable_share": ["renewables_share_energy"],
        "renewable_electricity_share": ["renewables_share_elec"],
        "energy_consumption": ["primary_energy_consumption"],
    }
    UNITS = {col: spec[1] for col, spec in COLUMNS.items()}

    def live_rows(self, series: str, entity: str) -> list[Row]:
        dataset, unit, _ = COLUMNS[series]
        values = _table(dataset).get(series, {}).get(entity, {})
        rows = [Row.make(series, entity, year, value, unit) for year, value in values.items()]
        return [r for r in rows if r]

    def url(self, series: str, entity: str) -> str:
        chart = COLUMNS[series][2]
        return f"https://ourworldindata.org/grapher/{chart}?country={_GRAPHER_CODE.get(entity, entity)}"
