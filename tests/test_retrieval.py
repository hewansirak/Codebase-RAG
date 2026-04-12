import numpy as np

from codebase_rag.embeddings import HashingEmbedder, tokenize
from codebase_rag.pipeline import build_index
from codebase_rag.rerank import HeuristicReranker
from codebase_rag.retrieval import BM25, Filters, HybridRetriever


def test_tokenize_splits_identifiers():
    assert tokenize("getUserName parse_http_response") == ["get", "user", "name", "parse", "http", "response"]


def test_bm25_prefers_documents_with_rare_terms():
    docs = [["common", "word"], ["common", "unicorn"], ["common", "common", "word"]]
    scores = BM25(docs).scores(["unicorn"])
    assert scores.argmax() == 1 and scores[0] == 0


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder()
    a, b = emb.encode(["parse json file"]), emb.encode(["parse json file"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5


def make_repo(tmp_path):
    (tmp_path / "auth.py").write_text("def verify_token(token):\n    return token.startswith('sk-')\n")
    (tmp_path / "billing.py").write_text("def charge_card(amount):\n    return amount * 100\n")
    (tmp_path / "web.ts").write_text("export function renderPage() {\n  return '<html/>'\n}\n")
    return tmp_path


def test_hybrid_search_finds_relevant_chunk_with_line_numbers(tmp_path):
    emb = HashingEmbedder()
    index = build_index(make_repo(tmp_path), "o/r", emb)
    retriever = HybridRetriever([index], emb, HeuristicReranker())
    top = retriever.search("where is the token verified?", k=1)[0]
    assert top.chunk.path == "auth.py" and top.chunk.citation == "auth.py:1-2"


def test_metadata_filters(tmp_path):
    emb = HashingEmbedder()
    retriever = HybridRetriever([build_index(make_repo(tmp_path), "o/r", emb)], emb)
    hits = retriever.search("function", k=10, filters=Filters(languages={"typescript"}))
    assert hits and {h.chunk.language for h in hits} == {"typescript"}
    assert retriever.search("function", filters=Filters(path_prefix="nope/")) == []


def test_multiple_repositories_are_searched_together(tmp_path):
    emb = HashingEmbedder()
    a = tmp_path / "a"
    a.mkdir()
    (a / "x.py").write_text("def alpha_feature():\n    pass\n")
    b = tmp_path / "b"
    b.mkdir()
    (b / "y.py").write_text("def beta_feature():\n    pass\n")
    retriever = HybridRetriever([build_index(a, "o/a", emb), build_index(b, "o/b", emb)], emb)
    assert retriever.search("beta_feature", k=1)[0].chunk.repo == "o/b"
    assert retriever.search("alpha_feature", k=1, filters=Filters(repos={"o/b"}))[0].chunk.repo == "o/b"
