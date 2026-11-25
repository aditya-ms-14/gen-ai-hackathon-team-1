"""Evidence Ledger. Owner: Aditya.

Structured facts table: verification reads numbers from here, never from
vector search. In-memory for now. Facts are keyed by (indicator, year, entity);
entity None is the reporting organisation, so a company figure never answers
a claim about Germany and the other way round.
TODO(Aditya, optional): persist to SQLite if we need it across reruns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .schemas import Fact

Key = tuple[str, int, Optional[str]]


class Ledger:
    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self._by_key: dict[Key, list[Fact]] = defaultdict(list)
        self._seen: set[tuple] = set()

    def add(self, facts: list[Fact]) -> None:
        for fact in facts:
            sig = (fact.fact_id, fact.source, fact.page, fact.value, fact.unit, fact.year, fact.entity, fact.quote)
            if sig in self._seen:  # the identical fact added twice (e.g. official data fetched per claim)
                continue
            self._seen.add(sig)
            self.facts.append(fact)
            self._by_key[(fact.indicator, fact.year, fact.entity)].append(fact)

    def lookup(self, indicator: str, year: int, entity: Optional[str] = None) -> list[Fact]:
        """Every fact for indicator + year + entity (more than one = possible conflict)."""
        return list(self._by_key.get((indicator, year, entity), []))

    def years(self, indicator: str, entity: Optional[str] = None) -> list[int]:
        return sorted({year for (ind, year, ent) in self._by_key if ind == indicator and ent == entity})

    def __len__(self) -> int:
        return len(self.facts)
