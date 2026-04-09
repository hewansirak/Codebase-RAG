"""On-disk vector index: one directory per repository (chunks.jsonl + vectors.npy + meta.json)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class Index:
    repo: str  # "owner/name"
    embedder: str
    chunks: list[Chunk]
    vectors: np.ndarray  # (n_chunks, dim), L2-normalised


def index_slug(repo: str) -> str:
    return repo.replace("/", "__")


def save_index(index: Index, indexes_dir: Path) -> Path:
    target = indexes_dir / index_slug(index.repo)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in index.chunks:
            fh.write(json.dumps(chunk.to_dict()) + "\n")
    np.save(target / "vectors.npy", index.vectors)
    (target / "meta.json").write_text(
        json.dumps({"repo": index.repo, "embedder": index.embedder, "chunks": len(index.chunks)}),
        encoding="utf-8",
    )
    return target


def load_index(repo: str, indexes_dir: Path) -> Index:
    target = indexes_dir / index_slug(repo)
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    with (target / "chunks.jsonl").open(encoding="utf-8") as fh:
        chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
    return Index(meta["repo"], meta["embedder"], chunks, np.load(target / "vectors.npy"))


def list_indexes(indexes_dir: Path) -> list[dict]:
    """Metadata of every saved index, sorted by repo name."""
    if not indexes_dir.exists():
        return []
    metas = [
        json.loads(p.read_text(encoding="utf-8")) for p in indexes_dir.glob("*/meta.json")
    ]
    return sorted(metas, key=lambda m: m["repo"])


def delete_index(repo: str, indexes_dir: Path) -> None:
    shutil.rmtree(indexes_dir / index_slug(repo), ignore_errors=True)
