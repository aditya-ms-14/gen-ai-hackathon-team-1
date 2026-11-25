# Industry-style demo reports (fictional publishers, real World Bank data)

Both reports are invented documents for testing the PS 99 Citation Auditor. The publishers do not exist. Every checkable figure comes from the World Bank World Development Indicators (WDI) bulk download, and some claims are deliberately wrong or unsupported. Rebuild with `python scripts/make_industry_reports.py <WDI bulk .csv or .zip>`.

## Files

| File | What it is |
|---|---|
| `industry_report_1_india.pdf` | "Indian Industry Decarbonisation Review 2024" (3 pages, prose only) |
| `industry_report_1_answer_key.csv` | answer key for it (13 claims) |
| `industry_report_2_europe.pdf` | "European Manufacturing Climate Progress Report 2024" (3 pages, prose only) |
| `industry_report_2_answer_key.csv` | answer key for it (14 claims) |
| `worldbank_environment_extract.csv` | 42 rows of the WDI bulk CSV in the same format as the 22.9 MB download: 7 entities x 6 series. Upload it as the evidence file. |

## Which evidence to upload

Upload `worldbank_environment_extract.csv` with either report. The full WDI bulk CSV gives the same verdicts. With no evidence file the app falls back to its official-data connectors.

## Expected verdict counts

| Report | Supported | Contradicted | Unsupported (of which greenwash) |
|---|---|---|---|
| `industry_report_1_india.pdf` | 7 | 3 | 3 (1) |
| `industry_report_2_europe.pdf` | 8 | 3 | 3 (1) |

Tolerance is +/-1.0 percentage point for % changes and share levels. Unsupported claims name a year the data does not cover (renewable share ends in 2021), an indicator it lacks (industrial waste), or are vague (greenwash).
