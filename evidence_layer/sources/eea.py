"""European Environment Agency: "Europe's environment 2025" country profiles. Owner: Aditya (official data sources).

EEA's SQL API (discodata.eea.europa.eu) returned 502/403 when this was built, so we
read the chart data behind the country-profile pages through the EEA website's
Plone REST API (the `++api++` prefix returns the page as JSON, including the
Plotly `visualization.dataSources` columns):

  https://www.eea.europa.eu/++api++/en/europe-environment-2025/countries/{country}/{chart}

Charts used (EU-27 members + EEA members; the "EU_trend_*" column gives the EU):
  renewable-energy-sources        share of renewables in gross final energy consumption, %
  total-greenhouse-gas-emissions  GHG emissions without LULUCF, index 1990 = 100
  final-energy-consumption        final energy consumption, index 2005 = 100
  waste-generation                total waste generation, index 2010 = 100 (biennial)
Index series only support % change claims; their unit says so, so a level claim
in tonnes is never compared with an index.
"""
from __future__ import annotations

import pycountry

from evidence_layer.sources._common import Connector, Row, get_json

PAGE = "https://www.eea.europa.eu/en/europe-environment-2025/countries/{country}/{chart}"
API = "https://www.eea.europa.eu/++api++/en/europe-environment-2025/countries/{country}/{chart}"
CHARTS = {  # series code -> (chart slug, unit)
    "EEA-SOE2025.renewable-energy-sources": ("renewable-energy-sources", "%"),
    "EEA-SOE2025.total-greenhouse-gas-emissions": ("total-greenhouse-gas-emissions", "index 1990=100"),
    "EEA-SOE2025.final-energy-consumption": ("final-energy-consumption", "index 2005=100"),
    "EEA-SOE2025.waste-generation": ("waste-generation", "index 2010=100"),
}
MEMBERS = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN",
           "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP",
           "SWE", "ISL", "LIE", "NOR", "CHE", "TUR"]
_SLUG_FIX = {"CZE": "czechia", "TUR": "turkiye"}


def country_slug(entity: str) -> str | None:
    if entity == "EUU":
        return "germany"  # every country page also carries the EU-27 trend column
    if entity not in MEMBERS:
        return None
    if entity in _SLUG_FIX:
        return _SLUG_FIX[entity]
    country = pycountry.countries.get(alpha_3=entity)
    return country.name.lower().replace(" ", "-") if country else None


class EEA(Connector):
    name = "European Environment Agency"
    slug = "eea_country_profiles"
    home = "https://www.eea.europa.eu/en/europe-environment-2025/countries"
    endpoint = API
    SERIES = {
        "renewable_share": ["EEA-SOE2025.renewable-energy-sources"],
        "ghg_emissions": ["EEA-SOE2025.total-greenhouse-gas-emissions"],
        "energy_consumption": ["EEA-SOE2025.final-energy-consumption"],
        "waste": ["EEA-SOE2025.waste-generation"],
    }
    UNITS = {code: spec[1] for code, spec in CHARTS.items()}

    def supports(self, entity: str) -> bool:
        return country_slug(entity) is not None

    def live_rows(self, series: str, entity: str) -> list[Row]:
        country = country_slug(entity)
        if country is None:
            return []
        chart, unit = CHARTS[series]
        page = get_json(self.slug, API.format(country=country, chart=chart))
        columns = ((page or {}).get("visualization") or {}).get("dataSources") or {}
        years = columns.get("years") or []
        trend = [k for k in columns if "_trend_" in k]
        if entity == "EUU":
            keys = [k for k in trend if k.startswith("EU_trend_")]
        else:
            keys = [k for k in trend if not k.startswith("EU_trend_")]
        if len(keys) != 1:
            return []
        rows = []
        for year, value in zip(years, columns[keys[0]]):
            if value not in (None, ""):
                row = Row.make(series, entity, year, value, unit)
                if row:
                    rows.append(row)
        return rows

    def url(self, series: str, entity: str) -> str:
        return PAGE.format(country=country_slug(entity) or "", chart=CHARTS[series][0])
