"""
BM25 sparse index that mirrors the ChromaDB collection.

Rebuilt from ChromaDB on demand, persisted as a pickle alongside chroma_db.
Thread-safe for concurrent reads; writes acquire a lock.

Usage
-----
    from backend.bm25_store import get_bm25_store
    store = get_bm25_store()
    store.rebuild()                    # call after every ingest / delete
    hits = store.query("what is the policy", top_k=20)
    # -> [{"chunk_id": "uuid_0", "score": 4.32, "rank": 1}, ...]
"""

from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config import CHROMA_DB_PATH

_BM25_PICKLE = Path(CHROMA_DB_PATH) / "bm25_index.pkl"


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Store:
    """
    BM25Okapi index over every chunk in the ChromaDB collection.

    Persisted fields (in pickle)
    ----------------------------
    _ids  : list[str]  — chunk IDs in index order
    _bm25 : BM25Okapi  — trained index
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not _BM25_PICKLE.exists():
            return
        try:
            with open(_BM25_PICKLE, "rb") as fh:
                data = pickle.load(fh)
            self._ids = data["ids"]
            self._bm25 = data["bm25"]
        except Exception:
            self._ids = []
            self._bm25 = None

    def _save(self) -> None:
        _BM25_PICKLE.parent.mkdir(parents=True, exist_ok=True)
        with open(_BM25_PICKLE, "wb") as fh:
            pickle.dump({"ids": self._ids, "bm25": self._bm25}, fh)

    # ------------------------------------------------------------------ #
    # Rebuild                                                              #
    # ------------------------------------------------------------------ #

    def rebuild(self) -> int:
        """
        Pull every chunk from ChromaDB, re-train BM25, persist to disk.
        Returns the number of documents indexed.
        """
        # Import here to avoid circular imports at module load time
        from backend.ingest import get_chroma_collection

        collection = get_chroma_collection()
        results = collection.get(include=["documents"], limit=100_000)

        ids: list[str] = results["ids"]
        docs: list[str] = results["documents"]

        if not docs:
            with self._lock:
                self._ids = []
                self._bm25 = None
                self._save()
            return 0

        tokenised = [_tokenise(d) for d in docs]

        with self._lock:
            self._ids = ids
            self._bm25 = BM25Okapi(tokenised)
            self._save()

        return len(docs)

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def query(self, text: str, top_k: int = 20) -> list[dict]:
        """
        Return up to top_k hits as:
            [{"chunk_id": str, "score": float, "rank": int}, ...]
        Sorted by score descending. Zero-score hits are excluded.
        """
        with self._lock:
            if self._bm25 is None or not self._ids:
                return []
            scores: list[float] = self._bm25.get_scores(_tokenise(text)).tolist()
            ids = list(self._ids)

        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {"chunk_id": cid, "score": round(sc, 6), "rank": i + 1}
            for i, (cid, sc) in enumerate(ranked)
            if sc > 0
        ]


# --------------------------------------------------------------------------- #
# Module-level singleton                                                        #
# --------------------------------------------------------------------------- #

_store: BM25Store | None = None
_store_lock = threading.Lock()


def get_bm25_store() -> BM25Store:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = BM25Store()
    return _store
