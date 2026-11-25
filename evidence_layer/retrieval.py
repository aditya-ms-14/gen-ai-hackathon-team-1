"""Hybrid Index + Hybrid Retriever. Owner: Aditya.

Pipeline: metadata filter (indicator, year) -> BM25Plus + vector -> reciprocal
rank fusion (k=60) -> cross-encoder rerank of the top 20 -> top k.

Vectors: sentence-transformers "all-MiniLM-L6-v2", normalised embeddings kept
in a numpy matrix per index (exact cosine = dot product). Chroma isn't used:
the metadata filter already runs in Python, corpora are a few hundred chunks,
and an in-memory matrix has no per-index collection to clean up in a
long-running Streamlit process.
Rerank: "cross-encoder/ms-marco-MiniLM-L-6-v2".
Both models load lazily, once per process. If either can't load (offline,
download fails, no torch), search silently degrades: no embedder -> BM25 only;
no cross-encoder -> fused order. Set AUDITOR_BM25_ONLY=1 to skip models.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

from auditor.schemas import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
RERANK_TOP = 20
MIN_COSINE = 0.25  # a chunk with no shared word must at least be this similar to enter the fusion

_lock = threading.Lock()
_models: dict[str, object] = {}  # name -> model, or None once loading failed


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _load(name: str):
    """Model by name, loaded once; None if it can't be loaded (never retried)."""
    if os.getenv("AUDITOR_BM25_ONLY"):
        return None
    with _lock:
        if name not in _models:
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer

                _models[name] = CrossEncoder(name) if name == RERANK_MODEL else SentenceTransformer(name)
            except Exception:  # offline, HF down, missing torch... BM25 still works
                _models[name] = None
        return _models[name]


def embedder():
    return _load(EMBED_MODEL)


def reranker():
    return _load(RERANK_MODEL)


def models_available() -> bool:
    return embedder() is not None and reranker() is not None


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum 1 / (k + rank), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


class HybridIndex:
    def __init__(self, dense: bool = True) -> None:
        self.chunks: list[Chunk] = []
        self.dense = dense            # False = BM25 only, never loads a model
        self.last_mode = "bm25"       # what the last search actually used (shown in debug / tests)
        self._vectors: dict[str, object] = {}  # chunk_id -> normalised embedding

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, indicator: Optional[str] = None,
               years: Optional[list[int]] = None, k: int = 5) -> list[Chunk]:
        pool = [
            c for c in self.chunks
            if (not indicator or not c.indicators or indicator in c.indicators)
            and (not years or not c.years or set(years) & set(c.years))
        ]
        if not pool:
            return []
        self.last_mode = "bm25"
        rankings = [self._bm25(query, pool)]
        dense = self._dense(query, pool) if self.dense else None
        if dense is not None:
            rankings.append(dense)
            self.last_mode = "hybrid"
        fused = rrf(rankings) if len(rankings) > 1 else rankings[0]
        if not fused:
            return []
        top = [pool[i] for i in fused[:RERANK_TOP]]
        if dense is not None and len(top) > 1:
            top = self._rerank(query, top)
        return top[:k]

    # ---- rankers (each returns pool indices, best first) ---------------
    @staticmethod
    def _bm25(query: str, pool: list[Chunk]) -> list[int]:
        # BM25Plus, not Okapi: Okapi gives idf 0 to a term in half the docs, so with
        # only 2-3 evidence PDFs it would rank real matches at zero.
        from rank_bm25 import BM25Plus

        docs = [_tokens(c.text) for c in pool]
        q = _tokens(query)
        if not q:
            return []
        scores = BM25Plus(docs).get_scores(q)
        ranked = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        return [i for i in ranked if set(q) & set(docs[i])]

    def _dense(self, query: str, pool: list[Chunk]) -> Optional[list[int]]:
        model = embedder()
        if model is None:
            return None
        try:
            import numpy as np

            todo = [c for c in pool if c.chunk_id not in self._vectors]
            if todo:
                vecs = model.encode([c.text for c in todo], normalize_embeddings=True, show_progress_bar=False)
                self._vectors.update({c.chunk_id: v for c, v in zip(todo, vecs)})
            q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            sims = np.stack([self._vectors[c.chunk_id] for c in pool]) @ q
        except Exception:
            return None
        ranked = sorted(range(len(pool)), key=lambda i: (-float(sims[i]), i))
        return [i for i in ranked if float(sims[i]) >= MIN_COSINE]

    @staticmethod
    def _rerank(query: str, top: list[Chunk]) -> list[Chunk]:
        model = reranker()
        if model is None:
            return top
        try:
            scores = model.predict([(query, c.text) for c in top], show_progress_bar=False)
        except Exception:
            return top
        order = sorted(range(len(top)), key=lambda i: (-float(scores[i]), i))
        return [top[i] for i in order]

    def __len__(self) -> int:
        return len(self.chunks)
