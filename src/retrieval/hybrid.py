"""
Hybrid retrieval: semantic (vector) + lexical (BM25), merged with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

RetrievalMode = Literal["semantic", "lexical", "hybrid"]

RRF_RANK_CONSTANT = 60
VALID_MODES = frozenset({"semantic", "lexical", "hybrid"})


@dataclass
class RetrieverBundle:
    vectorstore: InMemoryVectorStore
    bm25: Any  # BM25Retriever from langchain_community
    mode: RetrievalMode
    k: int
    lexical_weight: float


def _parse_retrieval_mode() -> RetrievalMode:
    raw = os.getenv("RAG_RETRIEVAL_MODE", "hybrid").strip().lower()
    # Allow inline comments in .env values (e.g. "hybrid   # default").
    raw = raw.split("#", 1)[0].strip()
    if raw not in VALID_MODES:
        return "hybrid"
    return raw  # type: ignore[return-value]


def get_retrieval_config(
    k: int = 4,
    mode: Optional[RetrievalMode] = None,
    lexical_weight: Optional[float] = None,
) -> tuple[RetrievalMode, int, float]:
    resolved_mode = mode if mode is not None else _parse_retrieval_mode()
    resolved_k = k
    if lexical_weight is not None:
        resolved_weight = lexical_weight
    else:
        try:
            resolved_weight = float(os.getenv("RAG_LEXICAL_WEIGHT", "0.5"))
        except ValueError:
            resolved_weight = 0.5
    return resolved_mode, resolved_k, resolved_weight


def build_bm25_retriever(chunks: List[Document], k: int) -> Any:
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
    meta = doc.metadata or {}
    source = meta.get("source", "")
    page = meta.get("page", "")
    start = meta.get("start_index", "")
    return f"{source}|{page}|{start}|{hash(doc.page_content)}"


def _rrf_contribution(rank: int, weight: float = 1.0) -> float:
    return weight / (RRF_RANK_CONSTANT + rank + 1)


def _merge_rrf(
    ranked_lists: List[List[Document]],
    *,
    weights: List[float],
    k: int,
) -> List[Document]:
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
    """Retrieve chunks for a query using the configured retrieval mode."""
    if not query.strip():
        return []

    k = bundle.k

    try:
        if bundle.mode == "semantic":
            return bundle.vectorstore.similarity_search(query, k=k)

        if bundle.mode == "lexical":
            return bundle.bm25.invoke(query)

        # Hybrid RRF merge
        semantic = bundle.vectorstore.similarity_search(query, k=k)
        lexical = bundle.bm25.invoke(query)
        return _merge_rrf(
            [semantic, lexical],
            weights=[1.0, bundle.lexical_weight],
            k=k,
        )
    except Exception:
        return []


def documents_to_sources(docs: List[Document]) -> List[dict]:
    """Map retrieved documents to deduplicated source metadata for the API."""
    from pathlib import Path

    seen: set = set()
    sources: List[dict] = []
    for doc in docs:
        meta = doc.metadata or {}
        source_path = meta.get("source", "")
        raw_page = meta.get("page")
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
