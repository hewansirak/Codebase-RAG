"""Retrieval-quality evaluation: hit rate, recall and MRR over a labelled question set.

Dataset format (JSONL), one object per line:
    {"question": "How are Python files chunked?", "expected_files": ["codebase_rag/chunking.py"]}
A question counts as a hit when any expected file appears in the top-k results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .retrieval import HybridRetriever


@dataclass
class EvalReport:
    questions: int
    hit_rate: float
    recall: float
    mrr: float
    k: int

    def __str__(self) -> str:
        return (
            f"questions={self.questions}  k={self.k}\n"
            f"hit_rate@{self.k}={self.hit_rate:.3f}  recall@{self.k}={self.recall:.3f}  MRR={self.mrr:.3f}"
        )


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(retriever: HybridRetriever, dataset: list[dict], k: int = 5) -> EvalReport:
    hits_total = recall_total = rr_total = 0.0
    for row in dataset:
        expected = set(row["expected_files"])
        ranked_files: list[str] = []
        for hit in retriever.search(row["question"], k=k):
            if hit.chunk.path not in ranked_files:
                ranked_files.append(hit.chunk.path)
        found = expected & set(ranked_files)
        hits_total += bool(found)
        recall_total += len(found) / len(expected)
        first = next((i for i, f in enumerate(ranked_files, start=1) if f in expected), None)
        rr_total += 1 / first if first else 0.0
    n = len(dataset) or 1
    return EvalReport(len(dataset), hits_total / n, recall_total / n, rr_total / n, k)
