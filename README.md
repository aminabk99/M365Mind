# M365Mind

Local AI for Microsoft 365 governance — query your Conditional Access policies, Sensitivity Labels, and Named Locations in plain English. Everything runs on your machine.

---

## What you need to install

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **Ollama** — [ollama.com](https://ollama.com) — manages the local AI model

After installing Ollama, pull the generation model (one time, ~1 GB):

```bash
ollama pull qwen2.5:1.5b
```

The embedding model (`nomic-embed-text-v1.5`, ~270 MB) downloads automatically via sentence-transformers on first query — no manual step.

---

## Setup

```bash
git clone https://github.com/aminabk99/M365Mind
cd M365Mind
pip install -r requirements.txt
```

Copy the environment file (demo mode needs no changes):

```bash
cp .env.example .env
```

---

## Run it

Open **three terminals** from the `M365Mind` folder:

```bash
# Terminal 1 — start Ollama (skip if already running as a service)
ollama serve

# Terminal 2 — backend
uvicorn backend.main:app --reload

# Terminal 3 — frontend
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) and click **Launch Demo** — no Microsoft account needed.

---

## Connect your real tenant (optional)

Register an app in [Azure Portal → Entra ID → App registrations](https://portal.azure.com) with:

- API permissions: `Policy.Read.All`, `InformationProtectionPolicy.Read.All`
- Redirect URI: `http://localhost:8000/callback`

Then add to your `.env`:

```
AZURE_CLIENT_ID=...
AZURE_TENANT_ID=...
AZURE_CLIENT_SECRET=...
```

Click **Sign in with Microsoft** on the landing screen, then **Sync Policies** in the sidebar.

---

## Performance

The first query used to be slow because the embedding model, the cross-encoder
reranker, and the Ollama LLM each load lazily on first use (tens of seconds
cold). The app now **warms all three in a background thread at startup**, so the
cost is paid while the page loads, not on your first click. The LLM is also kept
resident in Ollama between requests (`keep_alive`), so an idle session doesn't
pay a reload.

Everything is tunable by environment variable — trade quality for speed without
touching code:

| Variable | Default | Faster setting |
|----------|---------|----------------|
| `M365_LLM_MODEL` | `qwen2.5:1.5b` | `qwen2.5:0.5b` — ~2x faster answers (run `ollama pull qwen2.5:0.5b` first) |
| `M365_MAX_TOKENS` | `256` | lower (e.g. `160`) for shorter, quicker answers |
| `M365_KEEP_ALIVE` | `30m` | keep the model resident longer/shorter |
| `M365_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | `sentence-transformers/all-MiniLM-L6-v2` — ~3x faster embedding* |
| `M365_RERANK_CANDIDATES` | `8` | fewer = faster rerank |

\* Changing the embedding model changes the vector dimension, so clear
`chroma_db/` and re-run **Launch Demo** once after switching.

The dominant per-query cost is LLM generation on CPU (a few seconds). For the
snappiest demo: `set M365_LLM_MODEL=qwen2.5:0.5b` before starting the backend.

---

## Tech

FastAPI · Streamlit · ChromaDB · BM25 (rank-bm25) · nomic-embed-text-v1.5 (sentence-transformers) · Qwen2.5-1.5B (Ollama) · MSAL · Microsoft Graph API
