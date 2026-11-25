"""Who a claim or number is about. Shared contract (coordinator edits).

Countries use ISO3 codes ("DEU"); aggregates use World Bank codes: "WLD" world,
"EUU" European Union. None means the reporting organisation itself.
Facts and claims match only when their entity codes are equal.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import pycountry

ALIASES = {
    "united states": "USA", "united states of america": "USA", "america": "USA",
    "united kingdom": "GBR", "britain": "GBR", "great britain": "GBR",
    "russia": "RUS", "south korea": "KOR", "north korea": "PRK", "iran": "IRN", "vietnam": "VNM",
    "turkey": "TUR", "türkiye": "TUR", "czech republic": "CZE", "czechia": "CZE", "bolivia": "BOL",
    "venezuela": "VEN", "tanzania": "TZA", "syria": "SYR", "laos": "LAO", "moldova": "MDA",
    "taiwan": "TWN", "ivory coast": "CIV", "democratic republic of the congo": "COD", "dr congo": "COD",
    "european union": "EUU", "the eu": "EUU", "world": "WLD", "global": "WLD", "globally": "WLD",
}
# Short upper-case forms are matched case-sensitively so the pronoun "us" is not the USA.
_UPPER = {"US": "USA", "U.S.": "USA", "USA": "USA", "UK": "GBR", "U.K.": "GBR", "EU": "EUU"}


def country_code(name: str) -> Optional[str]:
    """'Germany' / 'germany' / 'DEU' / 'DE' / 'UK' -> 'DEU' / 'GBR'. None if not a country."""
    key = name.strip()
    if key in _UPPER:
        return _UPPER[key]
    if key.lower() in ALIASES:
        return ALIASES[key.lower()]
    try:
        return pycountry.countries.lookup(key).alpha_3
    except LookupError:
        return None


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern, re.Pattern, dict[str, str]]:
    names: dict[str, str] = dict(ALIASES)
    for c in pycountry.countries:
        for attr in ("name", "common_name", "official_name"):
            if value := getattr(c, attr, None):
                names[value.lower()] = c.alpha_3
    ordered = sorted(names, key=len, reverse=True)  # "Papua New Guinea" before "Guinea"
    ci = re.compile(r"\b(" + "|".join(re.escape(n) for n in ordered) + r")\b", re.IGNORECASE)
    cs = re.compile(r"(?<![\w.])(" + "|".join(re.escape(k) for k in sorted(_UPPER, key=len, reverse=True)) + r")(?![\w])")
    return ci, cs, names


def find_country(text: str) -> Optional[str]:
    """First country/aggregate mentioned in text, as a code. None if none found."""
    ci, cs, names = _patterns()
    hits = []
    if m := ci.search(text):
        hits.append((m.start(), names[m.group(1).lower()]))
    if m := cs.search(text):
        hits.append((m.start(), _UPPER[m.group(1)]))
    return min(hits)[1] if hits else None
