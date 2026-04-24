"""Streamlit UI: index a GitHub repo, chat with it, inspect citations and the retrieval map."""

from __future__ import annotations

import streamlit as st

from codebase_rag.config import Settings
from codebase_rag.embeddings import get_embedder
from codebase_rag.generation import ChatMemory, Generator, cited_hits
from codebase_rag.ingest import parse_github_url
from codebase_rag.pipeline import index_github_repo
from codebase_rag.repomap import build_repo_map
from codebase_rag.rerank import HeuristicReranker
from codebase_rag.retrieval import Filters, HybridRetriever
from codebase_rag.store import delete_index, list_indexes, load_index

st.set_page_config(page_title="Codebase RAG", page_icon="🧭", layout="wide")
settings = Settings()

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()
    st.session_state.messages = []  # {"role", "content", "hits"}


@st.cache_resource(show_spinner="Loading embedding model...")
def embedder_for(name: str):
    return get_embedder(name)


@st.cache_resource(show_spinner=False)
def retriever_for(repos: tuple[str, ...], embedder_name: str, mtimes: tuple):
    """`mtimes` is part of the cache key so re-indexing a repo invalidates the cached retriever."""
    indexes = [load_index(r, settings.indexes_dir) for r in repos]
    return HybridRetriever(indexes, embedder_for(embedder_name), HeuristicReranker())


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧭 Codebase RAG")
    embedder_name = st.selectbox(
        "Embedding model",
        [settings.embedding_model, "hashing"],
        help="'hashing' is a dependency-free baseline; the sentence-transformers model is semantic.",
    )

    st.subheader("1 · Index a repository")
    url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    if st.button("Index", type="primary", disabled=not url):
        try:
            parse_github_url(url)
            bar = st.progress(0.0, text="Starting")
            index_github_repo(url, embedder_for(embedder_name), settings, lambda f, m: bar.progress(f, text=m))
            st.success("Indexed")
            st.rerun()
        except Exception as exc:  # surface clone/validation errors in the UI
            st.error(str(exc))

    indexed = list_indexes(settings.indexes_dir)
    st.subheader("2 · Choose codebases")
    selected = st.multiselect(
        "Repositories",
        [m["repo"] for m in indexed],
        default=[m["repo"] for m in indexed][:1],
        help="Select several to chat with multiple codebases at once.",
    )
    if selected and st.button("Delete selected indexes"):
        for repo in selected:
            delete_index(repo, settings.indexes_dir)
        st.rerun()

    st.subheader("3 · Retrieval settings")
    top_k = st.slider("Sources per answer", 2, 12, 6)
    language_options = ["python", "javascript", "typescript", "java", "go", "rust", "markdown"]
    languages = st.multiselect("Language filter", language_options)
    path_prefix = st.text_input("Path prefix filter", placeholder="src/")

    api_key = settings.groq_api_key or st.text_input("Groq API key", type="password", help="Kept in this session only.")
    if st.button("Clear conversation"):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- main
chat_tab, map_tab = st.tabs(["💬 Chat", "🗺️ Repo map"])


def render_sources(hits):
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        with st.expander(f"[{n}] {c.repo}/{c.citation} · {c.kind} `{c.symbol}`"):
            st.code(c.text, language=c.language if c.language != "text" else None, line_numbers=True)


with chat_tab:
    if not indexed:
        st.info("Paste a GitHub URL in the sidebar and press **Index** to get started.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("hits"):
                render_sources(msg["hits"])

    question = st.chat_input("Ask about the selected codebase(s)", disabled=not selected)
    if question and selected:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        mtimes = tuple((settings.indexes_dir / r.replace("/", "__") / "meta.json").stat().st_mtime for r in selected)
        retriever = retriever_for(tuple(selected), embedder_name, mtimes)
        filters = Filters(languages=set(languages), path_prefix=path_prefix)
        with st.chat_message("assistant"):
            if api_key:
                generator = Generator(api_key, settings.llm_model, settings.llm_base_url)
                search_query = generator.condense(question, st.session_state.memory)
                hits = retriever.search(search_query, k=top_k, filters=filters)
                answer = st.write_stream(generator.stream(question, hits, st.session_state.memory))
                shown = cited_hits(answer, hits)
                st.session_state.memory.add(question, answer)
            else:
                hits = retriever.search(question, k=top_k, filters=filters)
                answer = "No Groq API key set, so here are the most relevant code excerpts (retrieval only)."
                st.markdown(answer)
                shown = hits
            render_sources(shown)
        st.session_state.messages.append({"role": "assistant", "content": answer, "hits": shown})
        st.session_state.last_hits = hits

with map_tab:
    if not selected:
        st.info("Select a repository to see its map.")
    else:
        repo = st.selectbox("Repository", selected)
        hits = st.session_state.get("last_hits", [])
        st.caption("Orange files were retrieved for the latest answer; tile size = lines of code.")
        st.plotly_chart(build_repo_map(load_index(repo, settings.indexes_dir), hits), use_container_width=True)
