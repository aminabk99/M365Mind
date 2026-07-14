"""
Cross-encoder reranker using sentence-transformers.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22 MB, runs on CPU, ~50 ms per (query, passage) pair
  - Returns a relevance logit; higher = more relevant

The model is downloaded on first use and cached in ~/.cache/huggingface.
No Ollama or GPU required.

Usage
-----
    from backend.reranker import rerank

    chunks = [
        {"chunk_id": "a", "text": "...", "metadata": {...}, "vector_score": 0.9},
        ...
    ]
    reranked = rerank(query="what is the policy?", chunks=chunks, top_k=5)
    # Each chunk gets a new key: "rerank_score" (float)
    # List is sorted by rerank_score descending, trimmed to top_k
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load the cross-encoder model (cached after first call)."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANK_MODEL)
        logger.info("Cross-encoder loaded: %s", RERANK_MODEL)
        return model
    except Exception as exc:
        logger.warning("Cross-encoder unavailable (%s). Falling back to vector scores.", exc)
        return None


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks using the cross-encoder. Each chunk dict must have a "text" key.

    Parameters
    ----------
    query  : the user's question
    chunks : list of chunk dicts (must contain "text")
    top_k  : how many to return after reranking

    Returns
    -------
    List of the same dicts, each augmented with "rerank_score",
    sorted descending by rerank_score, trimmed to top_k.

    Falls back to sorting by "rrf_score" if the model is unavailable.
    """
    if not chunks:
        return []

    model = _get_cross_encoder()

    if model is None:
        # Graceful degradation: keep RRF/vector order
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("rrf_score", chunk.get("vector_score", 0.0))
        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores: list[float] = model.predict(pairs).tolist()

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 6)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    return reranked
