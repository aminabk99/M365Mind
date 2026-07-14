# DocMind

A RAG-powered document intelligence app that runs **fully locally** — no API keys, no cloud, no cost. Upload PDFs, ask natural-language questions, get cited answers with confidence scores.

Uses a **hybrid BM25 + vector search pipeline** with cross-encoder reranking and enforced source citations.

---

## Features

- Upload PDFs and ask questions in plain English
- **Hybrid retrieval** — BM25 sparse search + ChromaDB vector search, fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — MiniLM rescores candidates against the query for higher precision
- **Cited answers** — every answer includes `[filename, page N]` citations; hallucinated citations are stripped automatically
- **Confidence score** — based on normalized rerank scores (🟢 ≥0.80 · 🟡 0.50–0.79 · 🔴 <0.50)
- Manage documents — list and delete uploaded PDFs from the sidebar
- Docker support for one-command startup

---

## Architecture

```
Streamlit UI  ──HTTP──►  FastAPI Backend
                              │
              ┌───────────────┼───────────────┐
          POST /upload    POST /query     GET /documents
              │                │
          LlamaIndex       ① Vector search   (ChromaDB, top-20)
          chunk + embed    ② BM25 search     (rank-bm25, top-20)
          → ChromaDB       ③ RRF fusion      (k=60, top-15)
          → BM25 index     ④ Cross-encoder rerank  (MiniLM, top-k)
                           ⑤ Ollama tinyllama  (cited-answer prompt)
                           ⑥ Citation enforcement
                              │
                    ChromaDB + BM25 index (./chroma_db)
                    Ollama (localhost:11434)
```

---

## Setup

**Requirements:** Python 3.11+ · [Ollama](https://ollama.com)

**1. Pull models & start Ollama**
```bash
ollama pull tinyllama
ollama pull nomic-embed-text
ollama serve
```

**2. Clone & install**
```bash
git clone https://github.com/aminabk99/docmind.git
cd docmind
pip install -r requirements.txt
```

**3. Configure**
```bash
cp .env.example .env
# No API keys needed — defaults work out of the box
```

**4. Run**
```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload
# API at http://localhost:8000

# Terminal 2 — frontend
streamlit run frontend/app.py
# UI at http://localhost:8501
```

**Or with Docker**
```bash
cp .env.example .env
# Set: OLLAMA_BASE_URL=http://host.docker.internal:11434
docker-compose up --build
```

> If you have an existing `chroma_db/` from a previous version, delete it first — embedding dimensions changed and ChromaDB will reject the mismatch.

---

## Demo

<img width="561" height="427" alt="DocMind Demo" src="https://github.com/user-attachments/assets/3853cdee-17ce-4b77-beb4-639db9bfce68" />

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + uvicorn |
| Chunking | LlamaIndex SentenceSplitter (512 tokens, 50 overlap) |
| Embeddings | Ollama nomic-embed-text |
| Vector DB | ChromaDB |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Reranking | sentence-transformers MiniLM cross-encoder |
| LLM | Ollama tinyllama |
| Containerization | Docker + docker-compose |

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>
