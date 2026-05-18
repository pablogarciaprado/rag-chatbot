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
from typing import Any, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

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
    bm25 = build_bm25_retriever(chunks, resolved_k)
    return RetrieverBundle(
        vectorstore=vectorstore,
        bm25=bm25,
        mode=resolved_mode,
        k=resolved_k,
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
    return [docs_by_key[key] for key in ordered_keys[:k]]


def retrieve_documents(query: str, bundle: RetrieverBundle) -> List[Document]:
    """
    Return up to *k* chunks for *query* according to the bundle's retrieval mode.

    Used by both prompt middleware and source attribution so the model and the UI
    reason over the same evidence.
    """
    if not query.strip():
        return []

    k = bundle.k

    try:
        if bundle.mode == "semantic":
            return bundle.vectorstore.similarity_search(query, k=k)

        if bundle.mode == "lexical":
            return bundle.bm25.invoke(query)

        semantic = bundle.vectorstore.similarity_search(query, k=k)
        lexical = bundle.bm25.invoke(query)
        # Semantic branch at weight 1.0; lexical scaled so operators can bias toward keywords.
        return _merge_rrf(
            [semantic, lexical],
            weights=[1.0, bundle.lexical_weight],
            k=k,
        )
    except Exception:
        # Retrieval failure should not block generation; the agent can still answer without context.
        return []


def documents_to_sources(docs: List[Document]) -> List[dict]:
    """
    Collapse retrieved chunks into citation rows for the API.

    Users care which file (and page) grounded the answer, not which overlapping chunk
    won the merge—so we dedupe by (path, page) rather than by chunk key.
    """
    from pathlib import Path

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
    return sources
