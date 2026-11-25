"""Canonical environmental indicators and the phrases that map to them.

Shared by ingestion (naming ledger facts) and the claim parser (naming claims),
so both sides land on the same indicator id. Add synonyms freely.
"""
from __future__ import annotations

import re
from typing import Optional

INDICATORS: dict[str, dict] = {
    "ghg_emissions": {
        "label": "GHG emissions",
        "share": False,
        "synonyms": ["carbon emissions", "ghg emissions", "greenhouse gas emissions", "greenhouse gas",
                     "co2e", "carbon footprint", "scope 1", "scope 2", "emissions"],
    },
    "co2_emissions": {
        "label": "CO2 emissions",
        "share": False,
        "synonyms": ["co2 emissions", "carbon dioxide emissions", "co₂ emissions", "fossil co2"],
    },
    "water_consumption": {
        "label": "Water consumption",
        "share": False,
        "synonyms": ["water consumption", "water use", "water usage", "water"],
    },
    "water_withdrawal": {
        "label": "Freshwater withdrawal",
        "share": False,
        "synonyms": ["freshwater withdrawals", "freshwater withdrawal", "water withdrawals", "water withdrawal"],
    },
    "renewable_share": {
        "label": "Renewable energy share",
        "share": True,  # claims about it are usually a level ("reached 75%")
        "synonyms": ["renewable energy usage", "renewable energy consumption", "renewable energy",
                     "renewables", "clean energy"],
    },
    "renewable_electricity_share": {
        "label": "Renewable electricity share",
        "share": True,
        "synonyms": ["renewable electricity", "renewable power", "clean electricity",
                     "electricity from renewable sources", "electricity from renewables"],
    },
    "forest_area_share": {
        "label": "Forest area (% of land)",
        "share": True,
        "synonyms": ["forest area", "forest cover", "forested land"],
    },
    "waste": {
        "label": "Waste generated",
        "share": False,
        "synonyms": ["industrial waste", "municipal waste", "municipal solid waste", "waste generation",
                     "waste generated", "waste"],
    },
    "energy_consumption": {
        "label": "Energy consumption",
        "share": False,
        "synonyms": ["energy consumption", "energy use", "electricity consumption", "electricity use"],
    },
    "biodiversity": {
        "label": "Biodiversity",
        "share": False,
        "synonyms": ["biodiversity", "land restored", "habitat restored"],
    },
}

# Longest phrase first so "renewable energy" wins over "energy", "industrial waste" over "waste".
_PHRASES = sorted(
    ((syn, ind) for ind, spec in INDICATORS.items() for syn in spec["synonyms"]),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def match_indicator(text: str) -> Optional[str]:
    """Return the canonical indicator id mentioned in text, or None."""
    lowered = text.lower()
    for phrase, indicator in _PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
            return indicator
    return None


def is_share(indicator: Optional[str]) -> bool:
    return bool(indicator and INDICATORS.get(indicator, {}).get("share"))


def label(indicator: Optional[str]) -> str:
    return INDICATORS.get(indicator or "", {}).get("label", indicator or "unknown")
