"""Embedding backends and the code-aware tokenizer shared with BM25."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase identifiers, splitting snake_case and camelCase: `getUserName` -> get, user, name."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(word)]
        tokens.extend(parts if len(parts) > 1 else [word.lower()])
    return tokens


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free bag-of-subwords embedder (feature hashing). Good for tests and offline demos."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode()).digest()
                col = int.from_bytes(digest[:4], "little") % self.dim
                out[row, col] += 1.0 if digest[4] & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    """Semantic embeddings from sentence-transformers (install the `ml` extra)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(name: str) -> Embedder:
    """`hashing` / `hashing-<dim>` -> HashingEmbedder, anything else -> a sentence-transformers model."""
    if name.startswith("hashing"):
        dim = int(name.split("-")[1]) if "-" in name else 512
        return HashingEmbedder(dim)
    return SentenceTransformerEmbedder(name)
