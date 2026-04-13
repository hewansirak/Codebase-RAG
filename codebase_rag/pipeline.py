"""End-to-end indexing: clone -> chunk -> embed -> save."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_file
from .config import Settings
from .embeddings import Embedder
from .ingest import clone_repository, iter_source_files, parse_github_url
from .store import Index, save_index

ProgressFn = Callable[[float, str], None]
BATCH = 64


def chunk_directory(root: Path, repo: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rel_path, text in iter_source_files(root):
        chunks += chunk_file(repo, rel_path, text)
    return chunks


def build_index(root: Path, repo: str, embedder: Embedder, progress: ProgressFn | None = None) -> Index:
    """Chunk and embed every supported file under `root`."""
    report = progress or (lambda _f, _m: None)
    report(0.05, "Chunking source files")
    chunks = chunk_directory(root, repo)
    if not chunks:
        raise ValueError(f"No supported source files found in {repo}")
    batches = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        batches.append(embedder.encode([c.embedding_text() for c in batch]))
        done = min(start + BATCH, len(chunks))
        report(0.1 + 0.85 * done / len(chunks), f"Embedded {done}/{len(chunks)} chunks")
    return Index(repo, embedder.name, chunks, np.vstack(batches))


def index_github_repo(url: str, embedder: Embedder, settings: Settings, progress: ProgressFn | None = None) -> Index:
    """Clone (or update) a GitHub repo, index it and persist the index."""
    report = progress or (lambda _f, _m: None)
    owner, name = parse_github_url(url)
    report(0.0, f"Cloning {owner}/{name}")
    root = clone_repository(url, settings.repos_dir)
    index = build_index(root, f"{owner}/{name}", embedder, report)
    save_index(index, settings.indexes_dir)
    report(1.0, f"Indexed {len(index.chunks)} chunks")
    return index
