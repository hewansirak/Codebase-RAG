"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion, then reranked."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, tokenize
from .rerank import Reranker
from .store import Index

RRF_K = 60


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in docs]
        self.lengths = np.array([len(doc) for doc in docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                freq = counts.get(term)
                if freq:
                    norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1)
                    out[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        return out


@dataclass
class Filters:
    """Metadata filters; an empty field means "no restriction"."""

    repos: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    kinds: set[str] = field(default_factory=set)
    path_prefix: str = ""

    def allows(self, chunk: Chunk) -> bool:
        return (
            (not self.repos or chunk.repo in self.repos)
            and (not self.languages or chunk.language in self.languages)
            and (not self.kinds or chunk.kind in self.kinds)
            and chunk.path.startswith(self.path_prefix)
        )


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


def _ranks(scores: np.ndarray, allowed: np.ndarray, limit: int) -> dict[int, int]:
    """Map chunk position -> 1-based rank for the top `limit` positive-scoring allowed chunks."""
    masked = np.where(allowed, scores, -np.inf)
    order = np.argsort(-masked)[:limit]
    return {int(i): r for r, i in enumerate((i for i in order if masked[i] > 0), start=1)}


class HybridRetriever:
    """Searches one or more indexes built with the same embedder."""

    def __init__(self, indexes: list[Index], embedder: Embedder, reranker: Reranker | None = None):
        if not indexes:
            raise ValueError("At least one index is required")
        if any(ix.embedder != embedder.name for ix in indexes):
            raise ValueError("Index was built with a different embedding model than the one supplied")
        self.embedder = embedder
        self.reranker = reranker
        self.chunks = [c for ix in indexes for c in ix.chunks]
        self.vectors = np.vstack([ix.vectors for ix in indexes])
        self.bm25 = BM25([tokenize(c.embedding_text()) for c in self.chunks])

    def search(self, query: str, k: int = 6, filters: Filters | None = None, pool: int = 30) -> list[Hit]:
        filters = filters or Filters()
        allowed = np.array([filters.allows(c) for c in self.chunks], dtype=bool)
        if not allowed.any():
            return []
        q_vec = self.embedder.encode([query])[0]
        vec_ranks = _ranks(self.vectors @ q_vec, allowed, pool)
        bm_ranks = _ranks(self.bm25.scores(tokenize(query)), allowed, pool)

        fused: dict[int, float] = {}
        for ranks in (vec_ranks, bm_ranks):
            for pos, rank in ranks.items():
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused, key=fused.get, reverse=True)[:pool]
        hits = [Hit(self.chunks[p], fused[p], vec_ranks.get(p), bm_ranks.get(p)) for p in ordered]
        if self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]
