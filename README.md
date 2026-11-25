# PS 99 – Environmental Report Citation Auditor

Checks every claim in a sustainability report against evidence and labels it **Supported**, **Contradicted** or **Unsupported**, with the exact source cited. The LLM only reads claims and writes explanations; **Python does every calculation**, and every cited quote is re-found word for word in its source.

**Team:** Aditya M S · Sibi Subhash · Fathima Rena · Kanishkar P S (B.Tech AI/ML)

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt      # requirements.txt = lean set (keyword-only retrieval)
copy .env.example .env                     # optional: GEMINI_API_KEY, GROQ_API_KEY, or start LM Studio's server
streamlit run ui/app.py --server.address 127.0.0.1
```

In the app: **Run problem-statement sample**, **Run country demo**, or upload a report (PDF/TXT) and evidence (CSV/XLSX/PDF, including the World Bank WDI bulk CSV). Without API keys it still runs: claims are read by the regex parser and explanations come from templates.

## Where to find what

| Question | Code | Docs |
|---|---|---|
| Which LLMs, fallback chain | `llm_router/router.py`, `providers.py`, `adapter.py`, `config.py`, `.env.example` | `llm_router/README.md` |
| Report parsing (PDF → claims) | `llm_router/claims.py` | |
| Evidence parsing & preprocessing | `evidence_layer/ingest.py`, `evidence_layer/worldbank_bulk.py` | |
| Normalisation | `auditor/indicators.py` (10 indicators, synonyms), `trust_layer/units.py` (units), `auditor/entities.py` (countries), `auditor/schemas.py` (Fact, Claim, Verdict) | |
| Verification calculations | `trust_layer/verifier.py`, `trust_layer/units.py` | |
| Citation Guard, confidence, greenwash | `trust_layer/guard.py`, `trust_layer/scoring.py`, `trust_layer/explain.py` | |
| RAG pipeline (end to end) | `auditor/pipeline.py`, `evidence_layer/resolver.py` | `docs/architecture.html` |
| Retrieval | `evidence_layer/ledger.py`, `evidence_layer/retrieval.py` | |
| Official data sources | `evidence_layer/sources/` (World Bank, OWID, WRI, UNFCCC, UNEP, EEA), `evidence_layer/scripts/fetch_official_data.py`, snapshots in `evidence_layer/data/official/` | |
| Evaluation | `scripts/evaluate.py`, `scripts/compare_llms.py`, answer keys in `data/samples/` | `docs/evaluation.md`, `docs/llm_comparison.md` |
| UI | `ui/app.py`, `ui/audit_view.py`, `ui/report.py` (PDF/CSV export) | `docs/ui-ux.html` (team design) |
| Review documents | | `docs/review2/` (report, slide deck, screenshots) |

## Repository layout

```
auditor/          shared contracts (schemas, indicators, entities, config) + pipeline     branch: main
evidence_layer/   ingestion, Evidence Ledger, hybrid retrieval, resolver, official data  branch: evidence-layer
llm_router/       LLM router (Gemini → Groq → LM Studio), claim parser, adapter           branch: llm-router
trust_layer/      verifier, units, Citation Guard, confidence & greenwash                 branch: trust-layer
ui/               Streamlit app, audit view, PDF/CSV export                               branch: ui
data/samples/     test reports, evidence files and answer keys (industry/ = World Bank)
scripts/          evaluate.py, compare_llms.py, dataset generators
tests/            end-to-end tests (each layer also has its own tests/)
docs/             architecture, UI design, evaluation results, review documents
```

| Branch | Folder | Owner |
|---|---|---|
| `main` | `auditor/`, `data/`, `scripts/`, `docs/` | Aditya M S |
| `evidence-layer` | `evidence_layer/` | Aditya M S |
| `llm-router` | `llm_router/` | Sibi Subhash |
| `trust-layer` | `trust_layer/` | Fathima Rena |
| `ui` | `ui/` | Kanishkar P S |

Each branch changes only its own folder, so they merge into `main` without conflicts.

## Datasets and answer keys

| Dataset | Files | Claims |
|---|---|---|
| Problem-statement sample | `data/samples/sample_report_2025.txt` + `evidence_2024_2025.csv`, key `answer_key.csv` | 6 |
| Country demo (official data, no upload) | `data/samples/demo_country_report.txt`, key `demo_answer_key.csv` | 13 |
| Company report (PDF evidence + Excel) | `data/samples/company_report_2024.pdf` + `company_evidence_tables.pdf` + `company_evidence.xlsx`, key `company_answer_key.csv` | 15 |
| Industry report: India | `data/samples/industry/industry_report_1_india.pdf` + World Bank WDI CSV (or `worldbank_environment_extract.csv`) | 13 |
| Industry report: Europe | `data/samples/industry/industry_report_2_europe.pdf` + the same World Bank file | 14 |

All reports and publishers in the samples are fictional; figures in the country and industry sets come from the World Bank WDI.

## Reproduce the numbers

```powershell
python -m pytest -q                         # all layers + end-to-end answer keys
python scripts/evaluate.py                  # deterministic evaluation → docs/evaluation.md
python scripts/compare_llms.py --modes "baseline,groq:openai/gpt-oss-120b,gemini:gemini-flash-lite-latest"
                                            # one model at a time on the same claims → docs/llm_comparison.md
```

## Deploy on Streamlit Community Cloud

share.streamlit.io → Create app → this repo, branch `main`, main file **`ui/app.py`**. Optional secrets: `GEMINI_API_KEY`, `GROQ_API_KEY`. Cloud installs `requirements.txt` (lean, no torch).
