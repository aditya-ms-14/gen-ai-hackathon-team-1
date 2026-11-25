"""Unit normalisation so 50 ML and 50,000 kL compare correctly. Owner: Sibi.

Small parser + lookup table instead of pint: fast, predictable, easy to extend.
A unit is read as  [magnitude word] core [of] [CO2e suffix] [per year],
e.g. "Mt CO2e", "MtCO₂eq", "million tonnes of CO2 equivalent", "billion m3",
"kt CO2e/yr". Percent family: "%", "% of total", "% of land area", "percent";
"pp" / "percentage points" is its own unit (a difference of two % levels).

Base units: tCO2e, t, kL, MWh, %, pp. Unknown units come back unchanged.
"""
from __future__ import annotations

import re
from typing import Optional

# core unit (lower-case, no spaces) -> (base unit, factor to base)
_CORE: dict[str, tuple[str, float]] = {
    # mass -> t  (Gg = kt, Tg = Mt)
    "t": ("t", 1.0), "tonne": ("t", 1.0), "tonnes": ("t", 1.0), "ton": ("t", 1.0), "tons": ("t", 1.0),
    "metricton": ("t", 1.0), "metrictons": ("t", 1.0), "metrictonne": ("t", 1.0), "metrictonnes": ("t", 1.0),
    "kg": ("t", 0.001), "kt": ("t", 1e3), "gg": ("t", 1e3), "mt": ("t", 1e6), "tg": ("t", 1e6),
    "gt": ("t", 1e9), "pg": ("t", 1e9),
    "kilotonnes": ("t", 1e3), "kilotons": ("t", 1e3), "megatonnes": ("t", 1e6), "megatons": ("t", 1e6),
    "gigatonnes": ("t", 1e9), "gigatons": ("t", 1e9),
    # volume -> kL (1 m3 = 1 kL)
    "kl": ("kL", 1.0), "m3": ("kL", 1.0), "cubicmeter": ("kL", 1.0), "cubicmeters": ("kL", 1.0),
    "cubicmetre": ("kL", 1.0), "cubicmetres": ("kL", 1.0), "km3": ("kL", 1e9), "bcm": ("kL", 1e9),
    "l": ("kL", 0.001), "litre": ("kL", 0.001), "litres": ("kL", 0.001), "liter": ("kL", 0.001),
    "liters": ("kL", 0.001), "kilolitres": ("kL", 1.0), "kiloliters": ("kL", 1.0),
    "ml": ("kL", 1e3),  # megalitres (sustainability reports rarely mean millilitres)
    "megalitres": ("kL", 1e3), "megaliters": ("kL", 1e3), "gl": ("kL", 1e6), "gigalitres": ("kL", 1e6),
    # energy -> MWh
    "kwh": ("MWh", 1e-3), "mwh": ("MWh", 1.0), "gwh": ("MWh", 1e3), "twh": ("MWh", 1e6),
    "gj": ("MWh", 1 / 3.6), "tj": ("MWh", 1e3 / 3.6), "pj": ("MWh", 1e6 / 3.6), "ej": ("MWh", 1e9 / 3.6),
    "toe": ("MWh", 11.63), "ktoe": ("MWh", 11_630.0), "mtoe": ("MWh", 11_630_000.0),
}
_MAGNITUDE = {"thousand": 1e3, "million": 1e6, "mn": 1e6, "mio": 1e6, "billion": 1e9, "bn": 1e9,
              "trillion": 1e12}
_EMISSION_INDICATORS = {"ghg_emissions", "co2_emissions"}

_PP = re.compile(r"^(pp|p\.p\.?|ppt|ppts|%\s*pts?|%\s*points?|percentage\s*points?|percent(age)?\s*pts?)$")
_PERCENT = re.compile(r"^(%|percent\b|per\s*cent\b|percentage\b(?!\s*points?))")
_PER_YEAR = re.compile(r"\s*(/\s*(yr|year|a|annum)|per\s+(year|annum)|a-1|yr-1)\s*$")
_CO2E = re.compile(r"(\s*of)?\s*(co2|carbon\s*dioxide)(\s*-?\s*(e|eq|equiv|equivalents?)\.?)?\s*$")
_MAG = re.compile(r"^(" + "|".join(_MAGNITUDE) + r")\b\s*")


def _clean(unit: str) -> str:
    return (unit.strip().lower().replace("₂", "2").replace("³", "3").replace("–", "-")
            .replace(" ", " "))


def parse_unit(unit: Optional[str]) -> Optional[tuple[str, float]]:
    """'Mt CO2e' -> ('tCO2e', 1e6); '% of land area' -> ('%', 1); unknown -> None."""
    if not unit or not unit.strip():
        return None
    key = re.sub(r"\s+", " ", _clean(unit))
    if _PP.match(key):
        return "pp", 1.0
    if _PERCENT.match(key):
        return "%", 1.0
    key = _PER_YEAR.sub("", key)
    co2e = False
    if (m := _CO2E.search(key)) and m.start() > 0:  # bare "co2e" is not a unit by itself
        key, co2e = key[:m.start()], True
    factor = 1.0
    if m := _MAG.match(key):
        factor, key = _MAGNITUDE[m.group(1)], key[m.end():]
    key = key.replace(" ", "").rstrip(".")
    if key not in _CORE:
        return None
    base, core_factor = _CORE[key]
    if co2e and base != "t":  # "kL CO2e" makes no sense
        return None
    return ("tCO2e" if co2e else base), factor * core_factor


def normalize(value: float, unit: Optional[str], indicator: Optional[str] = None) -> tuple[float, Optional[str]]:
    """Convert to the base unit. Unknown units come back unchanged.

    For emissions indicators a bare mass ("700 Mt") is read as CO2e, so it
    compares with a dataset's "Mt CO2e".
    """
    parsed = parse_unit(unit)
    if parsed is None:
        return value, unit
    base, factor = parsed
    if base == "t" and indicator in _EMISSION_INDICATORS:
        base = "tCO2e"
    return value * factor, base


def base_unit(unit: Optional[str], indicator: Optional[str] = None) -> Optional[str]:
    return normalize(1.0, unit, indicator)[1]


def is_pp(unit: Optional[str]) -> bool:
    return base_unit(unit) == "pp"


def fmt_number(x: Optional[float]) -> str:
    """Readable, guard-safe number: no exponent; thousands separators from 10,000 up."""
    if x is None:
        return "n/a"
    if abs(x - round(x)) < 1e-9 * max(1.0, abs(x)):
        n = int(round(x))
        return f"{n:,}" if abs(n) >= 10_000 else str(n)
    decimals = 2 if abs(x) >= 100 else 4 if abs(x) >= 1 else 6
    s = f"{x:,.{decimals}f}" if abs(x) >= 10_000 else f"{x:.{decimals}f}"
    return s.rstrip("0").rstrip(".")
