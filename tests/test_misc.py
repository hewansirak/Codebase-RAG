import hashlib
import hmac
import json

import pytest

from codebase_rag.chunking import Chunk
from codebase_rag.embeddings import HashingEmbedder
from codebase_rag.evaluate import evaluate
from codebase_rag.generation import ChatMemory, build_messages, cited_hits
from codebase_rag.ingest import iter_source_files, parse_github_url
from codebase_rag.pipeline import build_index
from codebase_rag.repomap import map_data
from codebase_rag.retrieval import Hit, HybridRetriever
from codebase_rag.store import load_index, save_index
from codebase_rag.webhook import repo_url_from_push, verify_signature


def hit(n):
    return Hit(Chunk("o/r", f"f{n}.py", 1, 5, "python", "function", f"fn{n}", "pass"), 1.0)


def test_parse_github_url():
    assert parse_github_url("https://github.com/hewansirak/Codebase-RAG.git") == ("hewansirak", "Codebase-RAG")
    with pytest.raises(ValueError):
        parse_github_url("https://gitlab.com/a/b")


def test_iter_source_files_skips_ignored_dirs_and_flattens_notebooks(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("var a = 1;")
    (tmp_path / "keep.py").write_text("x = 1")
    nb = {"cells": [{"cell_type": "code", "source": ["y = 2"]}, {"cell_type": "markdown", "source": ["# hi"]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    assert dict(iter_source_files(tmp_path)) == {"keep.py": "x = 1", "n.ipynb": "y = 2"}


def test_index_roundtrip(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    index = build_index(tmp_path, "o/r", HashingEmbedder())
    save_index(index, tmp_path / "idx")
    loaded = load_index("o/r", tmp_path / "idx")
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert loaded.vectors.shape == index.vectors.shape


def test_citations_and_memory():
    hits = [hit(1), hit(2), hit(3)]
    assert cited_hits("See [3] and [1], also [9].", hits) == [hits[2], hits[0]]
    assert cited_hits("no citations", hits) == hits
    memory = ChatMemory(max_turns=2)
    for i in range(4):
        memory.add(f"q{i}", f"a{i}")
    assert [q for q, _ in memory.turns] == ["q2", "q3"]
    messages = build_messages("next?", hits, memory)
    assert messages[0]["role"] == "system" and len(messages) == 6
    assert "[1] o/r/f1.py:1-5" in messages[-1]["content"]


def test_webhook_signature_and_ref_filtering():
    body = b'{"x":1}'
    sig = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
    assert verify_signature("s3cret", body, sig)
    assert not verify_signature("wrong", body, sig)
    assert not verify_signature("", body, sig)
    payload = {"ref": "refs/heads/main", "repository": {"default_branch": "main", "html_url": "https://github.com/o/r"}}
    assert repo_url_from_push(payload) == "https://github.com/o/r"
    assert repo_url_from_push({**payload, "ref": "refs/heads/dev"}) is None


def test_repo_map_highlights_retrieved_files(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def a():\n    pass\n")
    (tmp_path / "b.py").write_text("def b():\n    pass\n")
    index = build_index(tmp_path, "o/r", HashingEmbedder())
    data = map_data(index, [Hit(index.chunks[0], 1.0)])
    colors = dict(zip(data["ids"], data["colors"], strict=True))
    assert colors["o/r/pkg/a.py"] != colors["o/r/b.py"]
    assert "o/r/pkg" in data["ids"]


def test_evaluate_reports_perfect_scores_on_easy_dataset(tmp_path):
    (tmp_path / "auth.py").write_text("def verify_token(t):\n    return t\n")
    (tmp_path / "pay.py").write_text("def charge_card(c):\n    return c\n")
    emb = HashingEmbedder()
    retriever = HybridRetriever([build_index(tmp_path, "o/r", emb)], emb)
    report = evaluate(
        retriever,
        [
            {"question": "verify token", "expected_files": ["auth.py"]},
            {"question": "charge card", "expected_files": ["pay.py"]},
        ],
        k=1,
    )
    assert (report.hit_rate, report.mrr) == (1.0, 1.0)
