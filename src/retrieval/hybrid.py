"""
Hybrid retrieval for the RAG pipeline.

Dense embeddings excel at paraphrases and conceptual overlap; BM25 excels at exact
tokens (IDs, names, acronyms). Running both and merging ranks (rather than picking one)
covers queries that either method alone would miss, without normalizing incompatible
score scales between vector distance and BM25.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def _debug_enabled() -> bool:
    return os.getenv("ENABLE_PRINT_DEBUG", "False").lower() == "true"


def _debug_print(*parts: object) -> None:
    if _debug_enabled():
        print("[DEBUG][retrieval]", *parts)


def _doc_label(doc: Document, *, max_chars: int = 80) -> str:
    """Compact label for logs: filename, page, and a short text preview."""
    meta = doc.metadata or {}
    source = meta.get("source", "")
    name = Path(source).name if source else "?"
    raw_page = meta.get("page")
    page = f":p{(raw_page + 1)}" if isinstance(raw_page, int) else ""
    preview = doc.page_content.replace("\n", " ").strip()[:max_chars]
    if len(doc.page_content) > max_chars:
        preview += "…"
    return f"{name}{page} «{preview}»"

RetrievalMode = Literal["semantic", "lexical", "hybrid"]

# Standard RRF constant from the literature; dampens the influence of very deep ranks
# so a chunk that barely appears in one list cannot dominate a chunk ranked highly in both.
RRF_RANK_CONSTANT = 60
VALID_MODES = frozenset({"semantic", "lexical", "hybrid"})


@dataclass
class RetrieverBundle:
    """
    Everything needed at query time, built once at index time.

    Prompt injection and source attribution must use the same retrieval path; passing
    one bundle avoids drift where the model sees chunks the UI never cites.
    """

    vectorstore: InMemoryVectorStore
    bm25: Any  # BM25Retriever from langchain_community
    mode: RetrievalMode
    k: int
    fetch_k: int  # candidates per branch before hybrid RRF (typically 2 * k)
    lexical_weight: float


def _parse_retrieval_mode() -> RetrievalMode:
    """
    Read retrieval mode from the environment.

    Defaults to hybrid so new deployments get both signals without extra config.
    Invalid values fall back to hybrid rather than failing startup—retrieval should
    degrade gracefully when misconfigured.
    """
    raw = os.getenv("RAG_RETRIEVAL_MODE", "hybrid").strip().lower()
    # .env files often carry trailing comments; strip them so "hybrid # default" works.
    raw = raw.split("#", 1)[0].strip()
    if raw not in VALID_MODES:
        _debug_print(f"invalid RAG_RETRIEVAL_MODE={raw!r}, falling back to hybrid")
        return "hybrid"
    return raw  # type: ignore[return-value]


def get_retrieval_config(
    k: int = 4,
    mode: Optional[RetrievalMode] = None,
    lexical_weight: Optional[float] = None,
) -> tuple[RetrievalMode, int, float]:
    """
    Resolve retrieval knobs from explicit args, then env, then safe defaults.

    Centralizing here keeps rag.py and prompt middleware aligned on the same k and
    mode without each caller re-reading os.environ.

    Args:
        k: int - The number of chunks to retrieve.
        mode: Optional[RetrievalMode] - The retrieval mode to use.
        lexical_weight: Optional[float] - The weight of the lexical (BM25) branch in hybrid RRF merge.

    Returns:
        tuple[RetrievalMode, int, float] - The resolved retrieval mode, number of chunks, and lexical weight.
    """
    resolved_mode = mode if mode is not None else _parse_retrieval_mode()
    resolved_k = k
    if lexical_weight is not None:
        resolved_weight = lexical_weight
    else:
        try:
            # lexical_weight is how strongly the BM25 (keyword) branch counts when hybrid mode merges semantic and lexical results with RRF.
            # semantic search is always the baseline; lexical_weight only scales the BM25 side.
            resolved_weight = float(os.getenv("RAG_LEXICAL_WEIGHT", "0.5"))
        except ValueError:
            # A non-numeric env should not take down the app; 0.5 keeps branches balanced.
            resolved_weight = 0.5
    return resolved_mode, resolved_k, resolved_weight


def _branch_fetch_k(k: int, *, num_chunks: int) -> int:
    """Candidates to pull from each branch in hybrid mode (capped by corpus size)."""
    return min(2 * k, num_chunks) if num_chunks else k


def build_bm25_retriever(chunks: List[Document], k: int) -> Any:
    """
    Build an in-memory BM25 index over the same chunks as the vector store.

    Both indexes must index identical chunk objects so metadata (source, page) stays
    consistent when results are merged or deduplicated.
    """
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def build_retriever_bundle(
    vectorstore: InMemoryVectorStore,
    chunks: List[Document],
    *,
    k: int = 4,
    mode: Optional[RetrievalMode] = None,
    lexical_weight: Optional[float] = None,
) -> RetrieverBundle:
    """
    Assemble semantic and lexical retrievers after indexing.

    BM25 is built even when mode is semantic-only so switching RAG_RETRIEVAL_MODE
    only requires a restart, not a re-upload. The extra memory is acceptable for
    this app's in-memory, upload-sized corpora.

    A Lexical Weight is set in order to bias the retrieval towards the lexical (BM25) branch,
    but giving more weight to the semantic branch.
    """
    resolved_mode, resolved_k, resolved_weight = get_retrieval_config(
        k=k, mode=mode, lexical_weight=lexical_weight
    )
    num_chunks = len(chunks)
    fetch_k = _branch_fetch_k(resolved_k, num_chunks=num_chunks)
    bm25 = build_bm25_retriever(chunks, resolved_k)
    _debug_print(
        f"index ready mode={resolved_mode} k={resolved_k} fetch_k={fetch_k} "
        f"lexical_weight={resolved_weight} chunks={num_chunks}"
    )
    return RetrieverBundle(
        vectorstore=vectorstore,
        bm25=bm25,
        mode=resolved_mode,
        k=resolved_k,
        fetch_k=fetch_k,
        lexical_weight=resolved_weight,
    )


def _chunk_key(doc: Document) -> str:
    """
    Stable identity for a chunk when merging ranked lists.

    Source + page + start_index distinguish chunks from the same file; content hash
    breaks ties when metadata is sparse. Without a stable key, the same passage could
    appear twice in merged results under slightly different metadata.
    """
    meta = doc.metadata or {}
    source = meta.get("source", "")
    page = meta.get("page", "")
    start = meta.get("start_index", "")
    return f"{source}|{page}|{start}|{hash(doc.page_content)}"


def _rrf_contribution(rank: int, weight: float = 1.0) -> float:
    """Score contribution for a document at zero-based *rank* in one ranked list."""
    return weight / (RRF_RANK_CONSTANT + rank + 1)


def _merge_rrf(
    ranked_lists: List[List[Document]],
    *,
    weights: List[float],
    k: int,
) -> List[Document]:
    """
    Fuse multiple ranked lists with Reciprocal Rank Fusion.

    RRF only needs ranks, not raw scores—so we never have to calibrate vector cosine
    distance against BM25 term frequency. Chunks that rank well in both lists accumulate
    mass; chunks strong in only one list still have a path in via the other branch.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for doc_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(doc_list):
            key = _chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + _rrf_contribution(rank, weight)
            docs_by_key[key] = doc

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    merged = [docs_by_key[key] for key in ordered_keys[:k]]

    if _debug_enabled():
        _debug_print(f"RRF merge (k={k}, weights={weights}) -> {len(merged)} chunks")
        for i, key in enumerate(ordered_keys[:k]):
            _debug_print(f"  #{i + 1} score={scores[key]:.6f} {_doc_label(docs_by_key[key])}")

    return merged


def retrieve_documents(query: str, bundle: RetrieverBundle) -> List[Document]:
    """
    Return up to *k* chunks for *query* according to the bundle's retrieval mode.

    Used by both prompt middleware and source attribution so the model and the UI
    reason over the same evidence.
    """
    if not query.strip():
        return []

    k = bundle.k
    query_preview = query.strip().replace("\n", " ")
    if len(query_preview) > 120:
        query_preview = query_preview[:120] + "…"

    fetch_k = bundle.fetch_k
    _debug_print(
        f"query={query_preview!r} mode={bundle.mode} k={k} fetch_k={fetch_k} "
        f"lexical_weight={bundle.lexical_weight}"
    )

    try:
        if bundle.mode == "semantic":
            docs = bundle.vectorstore.similarity_search(query, k=k)
            if _debug_enabled():
                _debug_print(f"semantic-only -> {len(docs)} hits")
                for i, doc in enumerate(docs):
                    _debug_print(f"  sem #{i + 1} {_doc_label(doc)}")
            return docs

        if bundle.mode == "lexical":
            docs = bundle.bm25.invoke(query)
            if _debug_enabled():
                _debug_print(f"lexical-only -> {len(docs)} hits")
                for i, doc in enumerate(docs):
                    _debug_print(f"  lex #{i + 1} {_doc_label(doc)}")
            return docs

        semantic = bundle.vectorstore.similarity_search(query, k=fetch_k)
        prev_bm25_k = bundle.bm25.k
        # BM25’s k is fixed on the retriever object,
        # so we need to set it to the fetch_k for the lexical branch.
        # If we run on lexical only mode, fetch_k is set to the number of sources.
        try:
            bundle.bm25.k = fetch_k
            lexical = bundle.bm25.invoke(query)
        finally:
            bundle.bm25.k = prev_bm25_k

        if _debug_enabled():
            _debug_print(f"semantic branch (fetch_k={fetch_k}) -> {len(semantic)} hits")
            for i, doc in enumerate(semantic):
                _debug_print(f"  sem #{i + 1} {_doc_label(doc)}")
            _debug_print(f"lexical branch (fetch_k={fetch_k}) -> {len(lexical)} hits")
            for i, doc in enumerate(lexical):
                _debug_print(f"  lex #{i + 1} {_doc_label(doc)}")

        # Semantic branch at weight 1.0; lexical scaled so operators can bias toward keywords.
        return _merge_rrf(
            [semantic, lexical],
            weights=[1.0, bundle.lexical_weight],
            k=k,
        )
    except Exception as exc:
        _debug_print(f"retrieval failed: {type(exc).__name__}: {exc}")
        # Retrieval failure should not block generation; the agent can still answer without context.
        return []


def documents_to_sources(docs: List[Document]) -> List[dict]:
    """
    Collapse retrieved chunks into citation rows for the API.

    Users care which file (and page) grounded the answer, not which overlapping chunk
    won the merge—so we dedupe by (path, page) rather than by chunk key.
    """
    seen: set = set()
    sources: List[dict] = []
    for doc in docs:
        meta = doc.metadata or {}
        source_path = meta.get("source", "")
        raw_page = meta.get("page")
        # PyPDFLoader uses 0-based pages; the UI expects human-readable 1-based numbering.
        page = (raw_page + 1) if isinstance(raw_page, int) else None
        key = (source_path, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": Path(source_path).name if source_path else "Unknown",
                "path": source_path,
                "page": page,
            })

    if _debug_enabled():
        _debug_print(f"citations -> {len(sources)} unique source(s) from {len(docs)} chunk(s)")
        for i, src in enumerate(sources):
            page = f" p.{src['page']}" if src.get("page") else ""
            _debug_print(f"  cite #{i + 1} {src['file']}{page}")

    return sources
