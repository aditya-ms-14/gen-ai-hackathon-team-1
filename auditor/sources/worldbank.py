"""World Bank World Development Indicators (API v2). Owner: Aditya (official data sources).

Endpoint: https://api.worldbank.org/v2/country/{iso3}/indicator/{code}?format=json&date=1990:2024
Emissions use the 2024 EN.GHG.* series (EDGAR-based, AR5 GWPs, excluding LULUCF);
the old EN.ATM.CO2E.KT codes are archived.
"""
from __future__ import annotations

import pycountry

from ._common import Connector, Row, get_json

API = "https://api.worldbank.org/v2/country/{entity}/indicator/{series}"
_ISO2 = {"WLD": "1W", "EUU": "EU"}


class WorldBank(Connector):
    name = "World Bank WDI"
    slug = "world_bank_wdi"
    home = "https://data.worldbank.org/"
    endpoint = "https://api.worldbank.org/v2/country/{iso3}/indicator/{code}?format=json"
    SERIES = {
        "ghg_emissions": ["EN.GHG.ALL.MT.CE.AR5"],
        "co2_emissions": ["EN.GHG.CO2.MT.CE.AR5"],
        "renewable_share": ["EG.FEC.RNEW.ZS"],
        "renewable_electricity_share": ["EG.ELC.RNEW.ZS"],
        "water_withdrawal": ["ER.H2O.FWTL.K3"],
        "forest_area_share": ["AG.LND.FRST.ZS"],
    }
    UNITS = {
        "EN.GHG.ALL.MT.CE.AR5": "Mt CO2e",   # Total greenhouse gas emissions excluding LULUCF (Mt CO2e)
        "EN.GHG.CO2.MT.CE.AR5": "Mt CO2e",   # Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)
        "EG.FEC.RNEW.ZS": "%",               # Renewable energy consumption (% of total final energy consumption)
        "EG.ELC.RNEW.ZS": "%",               # Renewable electricity output (% of total electricity output)
        "ER.H2O.FWTL.K3": "billion m3",      # Annual freshwater withdrawals, total (billion cubic meters)
        "AG.LND.FRST.ZS": "%",               # Forest area (% of land area)
    }

    def live_rows(self, series: str, entity: str) -> list[Row]:
        data = get_json(self.slug, API.format(entity=entity, series=series),
                        {"format": "json", "date": "1990:2024", "per_page": 100})
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            return []
        rows = []
        for rec in data[1]:
            if rec.get("value") is None or rec.get("countryiso3code") != entity:
                continue
            row = Row.make(series, entity, rec["date"], rec["value"], self.UNITS[series])
            if row:
                rows.append(row)
        return rows

    def url(self, series: str, entity: str) -> str:
        iso2 = _ISO2.get(entity)
        if iso2 is None:
            country = pycountry.countries.get(alpha_3=entity)
            iso2 = country.alpha_2 if country else entity
        return f"https://data.worldbank.org/indicator/{series}?locations={iso2}"
