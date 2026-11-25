# gen-ai-hackathon-team-1

**PS 99 – Environmental Report Citation Auditor.** Checks every claim in a sustainability report against evidence and labels it **Supported**, **Contradicted** or **Unsupported**, with the exact source cited. Numbers are computed in Python, never by the language model; every cited quote is re-found word for word in its source.

Architecture: open [docs/architecture.html](docs/architecture.html) in a browser. UI design: [docs/ui-ux.html](docs/ui-ux.html).

## Run locally (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt   # or requirements.txt for the lean set (BM25-only retrieval)
copy .env.example .env                  # optional: GEMINI_API_KEY / GROQ_API_KEY, or start LM Studio's server
streamlit run ui/app.py
```

In the app, click **Run problem-statement sample** or **Run country demo**, or upload your own report (PDF/TXT) and evidence (CSV/XLSX/PDF).
No API key? It still runs: claims fall back to a regex parser and explanations to templates.

```powershell
python -m pytest -q        # all layers + end-to-end answer keys
```

## Deploy on Streamlit Community Cloud

1. share.streamlit.io → **Create app** → repo `aditya-ms-14/gen-ai-hackathon-team-1`, branch `main`, main file path **`ui/app.py`**.
2. **Advanced settings → Secrets** (optional, enables LLM parsing/narration):
   ```toml
   GEMINI_API_KEY = "..."
   GROQ_API_KEY = "..."
   ```
3. Deploy. Cloud installs `requirements.txt` (lean: no torch, so retrieval is BM25-only). Official data (World Bank, OWID, WRI, UNFCCC, UNEP, EEA) is fetched live, with bundled snapshots as fallback.

## Repository layout (one folder per layer, one branch per layer)

| Folder | Layer | Branch / owner |
|---|---|---|
| `auditor/` | Shared contracts (`schemas`, `indicators`, `entities`, `config`) and the `pipeline` that wires the layers | `main` · Aditya |
| `evidence_layer/` | Ingestion (CSV/XLSX/PDF), Evidence Ledger, hybrid retrieval, resolver, official data sources + snapshots | `evidence-layer` · Aditya |
| `llm_router/` | Team LLM router (Gemini → Groq → LM Studio), claim parser, adapter used by the pipeline | `sibi-subhash` · Sibi |
| `trust_layer/` | Verifier & calculator, units, Citation Guard, confidence & greenwash | `trust-layer` · Rena |
| `ui/` | Streamlit app from the team design, audit view, PDF/CSV export | `ui` · Kanishkar |
| `data/samples/` | Sample reports, evidence and answer keys | `main` |

Pipeline: report → claim parser → evidence resolver (ledger → official data → RAG) → verifier → narration → Citation Guard → confidence & greenwash → dashboard / report.
