"""UNFCCC national GHG inventories, served by the Climate Watch API. Owner: Aditya (official data sources).

UNFCCC's own data interface (di.unfccc.int) has no stable public API, so we read
the inventories Climate Watch mirrors: data source UNFCCC_AI (Annex I parties,
annual CRF submissions) and UNFCCC_NAI (non-Annex I, sparse BUR/NC years).
Annex I is tried first, then non-Annex I.
"""
from __future__ import annotations

from .wri import API, ClimateWatchBase


class UNFCCC(ClimateWatchBase):
    name = "UNFCCC GHG inventories"
    slug = "unfccc_ghg_inventories"
    home = "https://di.unfccc.int/time_series"
    endpoint = API + " (source_ids 276 UNFCCC_AI, 277 UNFCCC_NAI)"
    SPECS = {
        "UNFCCC_AI.GHG.TOTAL_EXCL_LULUCF": ("UNFCCC_AI", 276, "Aggregate GHGs", "Total GHG emissions without LULUCF"),
        "UNFCCC_NAI.GHG.TOTAL_EXCL_LULUCF": ("UNFCCC_NAI", 277, "Aggregate GHGs", "Total GHG emissions excluding LULUCF/LUCF"),
        "UNFCCC_AI.CO2.TOTAL_EXCL_LULUCF": ("UNFCCC_AI", 276, "CO2", "Total GHG emissions without LULUCF"),
        "UNFCCC_NAI.CO2.TOTAL_EXCL_LULUCF": ("UNFCCC_NAI", 277, "CO2", "Total GHG emissions excluding LULUCF/LUCF"),
    }
    SERIES = {
        "ghg_emissions": ["UNFCCC_AI.GHG.TOTAL_EXCL_LULUCF", "UNFCCC_NAI.GHG.TOTAL_EXCL_LULUCF"],
        "co2_emissions": ["UNFCCC_AI.CO2.TOTAL_EXCL_LULUCF", "UNFCCC_NAI.CO2.TOTAL_EXCL_LULUCF"],
    }
    UNITS = {code: "MtCO2e" for code in SPECS}
