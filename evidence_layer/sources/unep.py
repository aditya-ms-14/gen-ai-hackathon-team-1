"""UNEP environmental data via the UN SDG Global Database API. Owner: Aditya (official data sources).

Endpoint: https://unstats.un.org/sdgapi/v1/sdg/Series/Data?seriesCode={code}&areaCode={m49}
UNEP compiles the waste statistics (Waste and Environment Statistics, "UNEP / WESR")
behind SDG 11.6.1 and 12.4.2. Mapped series:
  EN_MWT_GENV  Municipal waste generated (tonnes)  -> waste
Areas use UN M49 numeric codes (Germany 276, World 1); the EU has no M49 code.
"""
from __future__ import annotations

from urllib.parse import urlencode

import pycountry

from evidence_layer.sources._common import Connector, Row, get_json

API = "https://unstats.un.org/sdgapi/v1/sdg/Series/Data"


def m49(entity: str) -> str | None:
    if entity == "WLD":
        return "1"
    country = pycountry.countries.get(alpha_3=entity)
    return str(int(country.numeric)) if country else None


class UNEP(Connector):
    name = "UNEP (UN SDG Global Database)"
    slug = "unep_sdg"
    home = "https://unstats.un.org/sdgs/dataportal"
    endpoint = API
    SERIES = {"waste": ["EN_MWT_GENV"]}
    UNITS = {"EN_MWT_GENV": "tonnes"}

    def supports(self, entity: str) -> bool:
        return m49(entity) is not None

    def live_rows(self, series: str, entity: str) -> list[Row]:
        area = m49(entity)
        if area is None:
            return []
        data = get_json(self.slug, API, {"seriesCode": series, "areaCode": area, "pageSize": 500})
        rows: dict[int, Row] = {}
        for rec in (data or {}).get("data", []):
            if str(rec.get("geoAreaCode")) != area:
                continue
            if (rec.get("dimensions") or {}).get("Reporting Type", "G") != "G":
                continue  # keep the globally harmonised figure, not national variants
            row = Row.make(series, entity, rec.get("timePeriodStart"), rec.get("value"), self.UNITS[series])
            if row:
                rows.setdefault(row.year, row)
        return list(rows.values())

    def url(self, series: str, entity: str) -> str:
        return f"{API}?{urlencode({'seriesCode': series, 'areaCode': m49(entity) or ''})}"
