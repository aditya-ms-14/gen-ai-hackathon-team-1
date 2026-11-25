"""World Resources Institute: Climate Watch historical emissions API. Owner: Aditya (official data sources).

Endpoint: https://www.climatewatchdata.org/api/v1/data/historical_emissions
          ?regions[]={iso3}&source_ids[]={id}&per_page=200
One request per (data source, country) returns every gas x sector; rows are then
matched by gas and sector *name*, so a change of Climate Watch's internal ids
yields "no data" (snapshot fallback) instead of a wrong number.
Region codes (WORLD, EUU) expand to member countries in this API, so only
single countries are served.
"""
from __future__ import annotations

from urllib.parse import urlencode

from ._common import AGGREGATES, Connector, Row, get_json

API = "https://www.climatewatchdata.org/api/v1/data/historical_emissions"


class ClimateWatchBase(Connector):
    # series code -> (data source name, data source id, gas name, sector name)
    SPECS: dict[str, tuple[str, int, str, str]] = {}

    def supports(self, entity: str) -> bool:
        return entity not in AGGREGATES

    def _params(self, source_id: int, entity: str) -> dict:
        return {"regions[]": entity, "source_ids[]": source_id, "per_page": 200}

    def live_rows(self, series: str, entity: str) -> list[Row]:
        source_name, source_id, gas, sector = self.SPECS[series]
        data = get_json(self.slug, API, self._params(source_id, entity))
        rows = []
        for rec in (data or {}).get("data", []):
            if (rec.get("iso_code3") != entity or rec.get("data_source") != source_name
                    or rec.get("gas") != gas or rec.get("sector") != sector):
                continue
            unit = (rec.get("unit") or "MtCO2e").replace("₂", "2")
            for point in rec.get("emissions", []):
                if point.get("value") is not None:
                    row = Row.make(series, entity, point["year"], point["value"], unit)
                    if row:
                        rows.append(row)
        return rows

    def url(self, series: str, entity: str) -> str:
        source_id = self.SPECS[series][1]
        return f"{API}?{urlencode({'regions[]': entity, 'source_ids[]': source_id, 'per_page': 200})}"


class ClimateWatch(ClimateWatchBase):
    name = "WRI Climate Watch"
    slug = "wri_climate_watch"
    home = "https://www.climatewatchdata.org/ghg-emissions"
    endpoint = API
    SPECS = {
        "CW.ALL_GHG.TOTAL_EXCL_LULUCF": ("Climate Watch", 274, "All GHG", "Total excluding LULUCF"),
        "CW.CO2.TOTAL_EXCL_LULUCF": ("Climate Watch", 274, "CO2", "Total excluding LULUCF"),
    }
    SERIES = {
        "ghg_emissions": ["CW.ALL_GHG.TOTAL_EXCL_LULUCF"],
        "co2_emissions": ["CW.CO2.TOTAL_EXCL_LULUCF"],
    }
    UNITS = {code: "MtCO2e" for code in SPECS}
