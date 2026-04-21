"""Command line interface: index, ask, eval, webhook."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .embeddings import get_embedder
from .evaluate import evaluate, load_dataset
from .generation import ChatMemory, Generator, cited_hits
from .pipeline import build_index, index_github_repo
from .rerank import HeuristicReranker
from .retrieval import HybridRetriever
from .store import list_indexes, load_index
from .webhook import serve


def _retriever(settings: Settings, repos: list[str], embedder_name: str) -> HybridRetriever:
    indexes = [load_index(r, settings.indexes_dir) for r in repos]
    return HybridRetriever(indexes, get_embedder(embedder_name), HeuristicReranker())


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="codebase-rag", description=__doc__)
    parser.add_argument(
        "--embedder",
        default=settings.embedding_model,
        help="'hashing' or a sentence-transformers model",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="clone and index a GitHub repository")
    p.add_argument("url")

    p = sub.add_parser("ask", help="ask a question about indexed repositories")
    p.add_argument("question")
    p.add_argument("--repo", action="append", help="owner/name (repeatable); default: all indexed")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--retrieval-only", action="store_true", help="print the retrieved sources, skip the LLM")

    p = sub.add_parser("eval", help="measure retrieval quality on a JSONL dataset")
    p.add_argument("dataset", type=Path)
    p.add_argument("--path", type=Path, default=Path("."), help="local directory to index for the evaluation")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("webhook", help="re-index repositories on GitHub push events")
    p.add_argument("--port", type=int, default=8765)

    sub.add_parser("list", help="list indexed repositories")

    args = parser.parse_args(argv)

    if args.command == "index":
        index_github_repo(args.url, get_embedder(args.embedder), settings, lambda f, m: print(f"[{f:4.0%}] {m}"))
    elif args.command == "list":
        for meta in list_indexes(settings.indexes_dir):
            print(f"{meta['repo']}  ({meta['chunks']} chunks, {meta['embedder']})")
    elif args.command == "ask":
        repos = args.repo or [m["repo"] for m in list_indexes(settings.indexes_dir)]
        if not repos:
            raise SystemExit("No indexes found. Run `python -m codebase_rag index <github-url>` first.")
        hits = _retriever(settings, repos, args.embedder).search(args.question, k=args.k)
        if args.retrieval_only or not settings.groq_api_key:
            for i, hit in enumerate(hits, start=1):
                print(f"[{i}] {hit.chunk.repo}/{hit.chunk.citation}  {hit.chunk.symbol}  score={hit.score:.4f}")
            return
        generator = Generator(settings.groq_api_key, settings.llm_model, settings.llm_base_url)
        answer = "".join(generator.stream(args.question, hits, ChatMemory()))
        print(answer)
        print("\nSources:")
        for hit in cited_hits(answer, hits):
            print(f"  - {hit.chunk.repo}/{hit.chunk.citation}")
    elif args.command == "eval":
        index = build_index(args.path.resolve(), "local/eval", get_embedder(args.embedder))
        retriever = HybridRetriever([index], get_embedder(args.embedder), HeuristicReranker())
        print(evaluate(retriever, load_dataset(args.dataset), k=args.k))
    elif args.command == "webhook":
        serve(settings, args.embedder, args.port)
