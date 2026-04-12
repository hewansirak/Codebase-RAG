"""Second-stage rerankers applied to the fused candidate pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .embeddings import tokenize

if TYPE_CHECKING:
    from .retrieval import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class HeuristicReranker:
    """Cheap reranker: keeps the fused score and boosts query terms found in the symbol or path."""

    def __init__(self, symbol_boost: float = 0.02, path_boost: float = 0.01):
        self.symbol_boost, self.path_boost = symbol_boost, path_boost

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        terms = set(tokenize(query))
        if not terms:
            return hits
        for hit in hits:
            symbol_terms = set(tokenize(hit.chunk.symbol))
            path_terms = set(tokenize(hit.chunk.path))
            hit.score += self.symbol_boost * len(terms & symbol_terms) / len(terms)
            hit.score += self.path_boost * len(terms & path_terms) / len(terms)
        return sorted(hits, key=lambda h: h.score, reverse=True)


class CrossEncoderReranker:
    """Cross-encoder reranker from sentence-transformers (install the `ml` extra)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self._model.predict([(query, h.chunk.embedding_text()[:2000]) for h in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.score = float(score)
        return sorted(hits, key=lambda h: h.score, reverse=True)
