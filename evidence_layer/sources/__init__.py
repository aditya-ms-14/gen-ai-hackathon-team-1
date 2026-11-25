"""Official data sources for country / aggregate claims. Owner: Aditya (official data sources).

    fetch_official("co2_emissions", "DEU", [2010, 2022]) -> list[Fact]
    snapshot_pages() -> {source name: {1: page text}}   # for the Citation Guard

Priority (first source that covers every requested year wins; otherwise the
source covering the most years): World Bank WDI, Our World in Data,
WRI Climate Watch, UNFCCC GHG inventories, UNEP (UN SDG Global Database),
European Environment Agency; renewable_share puts EEA before OWID (same
definition as WDI). Facts always come from ONE source, so two
publishers' slightly different numbers never look like a conflict.

Each source: live API -> file cache (data/runtime/cache/) -> curated snapshot
(data/official/<slug>.csv, built by scripts/fetch_official_data.py).
`set_offline(True)` (or AUDITOR_OFFLINE=1) uses only the snapshots.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from auditor.entities import country_code
from auditor.schemas import Fact
from evidence_layer.sources._common import AGGREGATES, Connector, is_offline, page_text, set_offline
from evidence_layer.sources.eea import EEA
from evidence_layer.sources.owid import OurWorldInData
from evidence_layer.sources.unep import UNEP
from evidence_layer.sources.unfccc import UNFCCC
from evidence_layer.sources.worldbank import WorldBank
from evidence_layer.sources.wri import ClimateWatch

log = logging.getLogger("evidence_layer.sources")

CONNECTORS: list[Connector] = [WorldBank(), OurWorldInData(), ClimateWatch(), UNFCCC(), UNEP(), EEA()]
# Same-definition sources first: WB and EEA measure renewables as a share of FINAL energy
# consumption; OWID uses PRIMARY energy (substitution method), a different number.
PRIORITY_OVERRIDES: dict[str, list[str]] = {
    "renewable_share": ["World Bank WDI", "European Environment Agency", "Our World in Data"],
}


def _ordered(indicator: str) -> list[Connector]:
    names = PRIORITY_OVERRIDES.get(indicator)
    if not names:
        return CONNECTORS
    rank = {name: i for i, name in enumerate(names)}
    return sorted(CONNECTORS, key=lambda c: rank.get(c.name, len(rank)))

__all__ = ["CONNECTORS", "fetch_official", "snapshot_pages", "set_offline", "is_offline",
           "normalize_entity", "indicators_covered"]


def normalize_entity(entity: Optional[str]) -> Optional[str]:
    """'Germany' / 'DEU' / 'DE' -> 'DEU'; 'World' / 'WLD' -> 'WLD'; 'EU' / 'EUU' -> 'EUU'."""
    if not entity or not isinstance(entity, str):
        return None
    key = entity.strip()
    if key.upper() in AGGREGATES:
        return key.upper()
    return country_code(key)


def indicators_covered() -> dict[str, list[str]]:
    """indicator id -> source names that publish it, in priority order."""
    indicators = sorted({ind for c in CONNECTORS for ind in c.SERIES})
    return {ind: [c.name for c in _ordered(ind) if ind in c.SERIES] for ind in indicators}


def fetch_official(indicator: str, entity: str, years: Iterable[int]) -> list[Fact]:
    """Facts for indicator + entity + years from the first source covering all years. Never raises."""
    try:
        code = normalize_entity(entity)
        wanted = sorted({int(y) for y in years if y is not None})
        if not indicator or not code or not wanted:
            return []
        best: list[Fact] = []
        for connector in _ordered(indicator):
            if not connector.series_for(indicator) or not connector.supports(code):
                continue
            try:
                facts = connector.fetch(indicator, code, wanted)
            except Exception as exc:  # one broken source must not hide the others
                log.info("%s failed for %s/%s: %s", connector.name, indicator, code, exc)
                continue
            if {f.year for f in facts} >= set(wanted):
                return facts
            if len(facts) > len(best):
                best = facts
        return best
    except Exception as exc:
        log.warning("fetch_official(%r, %r, %r) failed: %s", indicator, entity, years, exc)
        return []


def snapshot_pages() -> dict[str, dict[int, str]]:
    """Page 1 text per source: every snapshot line plus every quote served in this process."""
    pages: dict[str, dict[int, str]] = {}
    for connector in CONNECTORS:
        try:
            pages[connector.name] = {1: page_text(connector.name, connector.slug)}
        except Exception as exc:
            log.warning("snapshot page for %s failed: %s", connector.name, exc)
    return pages
