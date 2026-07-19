"""
Embedding module for M365Mind.

Local embeddings via sentence-transformers. Defaults to all-MiniLM-L6-v2
for fast CPU inference; override with M365_EMBED_MODEL.

Model downloads automatically on first use (~270 MB, cached in
~/.cache/huggingface after that).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Default is all-MiniLM-L6-v2: ~80 MB, loads in a few seconds, embeds fast on
# CPU. It replaced nomic-embed-text-v1.5 (~270 MB, ~50 s to load) because the
# cold load was the single biggest source of "the app is slow" — for 17 short
# policies the accuracy difference is negligible. Env-configurable; to go back
# to the larger model set M365_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5.
# Changing this changes the vector dimension, so clear chroma_db/ once and
# re-load after switching.
_MODEL_NAME = os.getenv("M365_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", _MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
    logger.info("Embedding model ready.")
    return model


def embed(text: str) -> list[float]:
    """Embed a single string. Returns a normalised float list."""
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings efficiently. Returns list of float lists."""
    return _get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()
