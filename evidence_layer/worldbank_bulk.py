"""World Bank WDI bulk-download CSV (API_*_DS2_en_csv_*.csv) -> long evidence table. Owner: Aditya.

The bulk file has a 4-line preamble ("Data Source", blank, "Last Updated Date", blank), then
"Country Name","Country Code","Indicator Name","Indicator Code","1960".."2025".  We map series by
Indicator Code only (the series in evidence_layer.sources.worldbank) and ignore every other series,
because matching on the long "Indicator Name" text would send "Carbon dioxide (CO2) emissions ..."
to generic emissions.  Output columns: entity (World Bank country code, so aggregates such as
WLD / EUU survive), indicator, year, value, unit, source_type ("official"), series, country_name.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import pandas as pd

from evidence_layer.sources.worldbank import WorldBank

# indicator name written into the long table; auditor.indicators.match_indicator maps each to its id
INDICATOR_TEXT = {
    "ghg_emissions": "Total greenhouse gas emissions",
    "co2_emissions": "CO2 emissions",
    "renewable_share": "Renewable energy consumption",
    "renewable_electricity_share": "Renewable electricity output",
    "water_withdrawal": "Freshwater withdrawals",
    "forest_area_share": "Forest area",
}
# series code -> (indicator text, unit)
SERIES_MAP: dict[str, tuple[str, str]] = {
    code: (INDICATOR_TEXT[ind], WorldBank.UNITS[code])
    for ind, codes in WorldBank.SERIES.items() for code in codes
}
LONG_COLUMNS = ["entity", "indicator", "year", "value", "unit", "source_type", "series", "country_name"]


def _header_row(p: Path) -> Optional[int]:
    """Index of the 'Country Name','Country Code',... header line in the first lines of a CSV, or None."""
    try:
        with open(p, newline="", encoding="utf-8-sig") as f:
            for i, row in zip(range(12), csv.reader(f)):
                cells = [c.strip() for c in row]
                if "Country Code" in cells and "Indicator Code" in cells:
                    return i
                if i == 0 and cells and cells[0] != "Data Source":
                    return None
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return None


def is_worldbank_bulk(p: Path) -> bool:
    return p.suffix.lower() == ".csv" and _header_row(p) is not None


def read_worldbank_bulk(p: Path) -> pd.DataFrame:
    """Parse a WDI bulk CSV into the long layout (only the mapped series, only years with a value)."""
    skip = _header_row(p) or 0
    df = pd.read_csv(p, dtype=str, keep_default_na=False, skiprows=skip, encoding="utf-8-sig", index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["Indicator Code"].isin(SERIES_MAP)].reset_index(drop=True)
    year_cols = [c for c in df.columns if c.isdigit()]
    long = df.melt(id_vars=["Country Name", "Country Code", "Indicator Code"], value_vars=year_cols,
                   var_name="year", value_name="value", ignore_index=False)
    long = long[long["value"].str.strip() != ""]
    long["_row"] = long.index
    long = long.sort_values(["_row", "year"], kind="stable")
    codes = long["Indicator Code"]
    out = pd.DataFrame({
        "entity": long["Country Code"].str.strip().str.upper(),
        "indicator": codes.map(lambda c: SERIES_MAP[c][0]),
        "year": long["year"],
        "value": long["value"].str.strip(),
        "unit": codes.map(lambda c: SERIES_MAP[c][1]),
        "source_type": "official",
        "series": codes,
        "country_name": long["Country Name"].str.strip(),
    })
    return out.reset_index(drop=True)[LONG_COLUMNS]
