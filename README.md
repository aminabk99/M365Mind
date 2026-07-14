# M365Mind

Local AI for Microsoft 365 governance — query your Conditional Access policies, Sensitivity Labels, and Named Locations in plain English. Everything runs on your machine.

---

## What you need to install

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **Ollama** — [ollama.com](https://ollama.com) — manages the local AI model

After installing Ollama, pull the model (one time, ~1 GB):

```bash
ollama pull qwen2.5:1.5b
```

---

## Setup

```bash
git clone https://github.com/aminabk99/m365mind
cd docmind
pip install -r requirements.txt
```

Copy the environment file (demo mode needs no changes):

```bash
cp .env.example .env
```

---

## Run it

Open **two terminals** from the `docmind` folder:

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

## Tech

FastAPI · Streamlit · ChromaDB · BM25 · nomic-embed-text · Qwen2.5-1.5B via Ollama · MSAL · Microsoft Graph API
