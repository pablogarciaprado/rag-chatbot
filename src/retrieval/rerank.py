"""
Re-rank retrieved chunks with the Google Cloud Discovery Engine Ranking API.

The ranking service is stateless: send a query plus candidate record texts and
receive relevance scores. This runs after first-stage retrieval (vector / BM25 /
RRF) to reorder the candidate pool before chunks reach the LLM prompt.

The ranking API uses the term record to indicate a document. 
A record is made up of an ID, a title, and the content of a document. 
Unlike the documents that are contained in Agent Search data stores, 
the records input to the ranking API are in JSON format and have not been indexed by Agent Search.

The maximum supported tokens per record depends on the model version being used. 
For example, models up to version 003 support 512 tokens, while version 004 supports 1024 tokens. 
If the combined length of the title and content exceeds the model's token limit, 
the extra content is truncated. You can include up to 200 records per request.

To read more about the Discovery Engine Ranking API, see the following link:
- https://docs.cloud.google.com/generative-ai-app-builder/docs/ranking
- https://docs.cloud.google.com/generative-ai-app-builder/docs/ranking#models
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

# Conservative per-record cap; semantic-ranker models use a ~1024-token window.
MAX_RECORD_CONTENT_CHARS = 4000
DISCOVERY_ENGINE_MAX_RECORDS = 200 # You can include up to 200 records per request, according to the Discovery Engine Ranking API documentation.


def _debug_enabled() -> bool:
    return os.getenv("ENABLE_PRINT_DEBUG", "False").lower() == "true"


def _debug_print(*parts: object) -> None:
    if _debug_enabled():
        print("[DEBUG][rerank]", *parts)


def is_rerank_enabled() -> bool:
    raw = os.getenv("RAG_RERANK_ENABLED", "false").strip().lower()
    raw = raw.split("#", 1)[0].strip()
    return raw in {"1", "true", "yes", "on"}


def _parse_rerank_candidates() -> int:
    raw = os.getenv("RAG_RERANK_CANDIDATES", "0").strip()
    raw = raw.split("#", 1)[0].strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _parse_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip().split("#", 1)[0].strip()
    try:
        return float(raw)
    except ValueError:
        return default


def get_rerank_filter_config() -> tuple[float, float]:
    """
    Resolve post-rerank score filtering from the environment.

    Returns:
        (min_score, gap_ratio) — both in [0, 1]. Documents below *min_score*
        are dropped; scanning stops when the next score falls below
        *gap_ratio* × the previous kept score.
    """
    min_score = max(0.0, min(1.0, _parse_env_float("RAG_RERANK_MIN_SCORE", 0.12)))
    gap_ratio = max(0.0, min(1.0, _parse_env_float("RAG_RERANK_GAP_RATIO", 0.45)))
    return min_score, gap_ratio


def get_rerank_config() -> tuple[bool, str, str, str, int]:
    """
    Resolve re-ranking settings from the environment.

    Returns:
        (enabled, project_id, location, model, explicit_candidate_pool)
    """
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
        or os.getenv("GCP_PROJECT")
        or ""
    ).strip()
    location = os.getenv("RAG_RERANK_LOCATION", "global").strip().split("#", 1)[0].strip()
    model = (
        os.getenv("RAG_RERANK_MODEL", "semantic-ranker-default@latest")
        .strip()
        .split("#", 1)[0]
        .strip()
    )
    return is_rerank_enabled(), project, location, model, _parse_rerank_candidates()


def resolve_rerank_pool_k(
    k: int,
    *,
    num_chunks: int,
    explicit_pool: int = 0,
) -> int:
    """
    How many first-stage hits to keep before Discovery Engine re-ranking.

    Args:
        k: The number of chunks to retrieve.
        num_chunks: The number of chunks to re-rank.
        explicit_pool: The number of chunks to keep before re-ranking.

    Returns:
        The number of chunks to keep before re-ranking.
    """
    if explicit_pool > 0:
        pool = explicit_pool
    else:
        pool = max(2 * k, k + 4)
    pool = min(pool, DISCOVERY_ENGINE_MAX_RECORDS)
    if num_chunks:
        pool = min(pool, num_chunks)

    resolved_pool_k = max(pool, k) if num_chunks >= k else num_chunks
    _debug_print(f"resolved_pool_k={resolved_pool_k}")

    return resolved_pool_k


def _doc_record_id(doc: Document) -> str:
    """Stable id for a chunk; must match hybrid._chunk_key for dedupe consistency."""
    meta = doc.metadata or {}
    source = meta.get("source", "")
    page = meta.get("page", "")
    start = meta.get("start_index", "")
    return f"{source}|{page}|{start}|{hash(doc.page_content)}"


def _record_title(doc: Document) -> str:
    meta = doc.metadata or {}
    source = meta.get("source", "")
    name = Path(source).name if source else "chunk"
    raw_page = meta.get("page")
    if isinstance(raw_page, int):
        return f"{name} (page {raw_page + 1})"
    return name


def filter_reranked_documents(
    docs: List[Document],
    *,
    min_score: float,
    gap_ratio: float,
) -> List[Document]:
    """
    Drop reranked chunks below *min_score* or after a score cliff.

    The top chunk must meet *min_score*; otherwise nothing is kept. Each
    subsequent chunk must also meet *min_score* and stay at or above
    *gap_ratio* × the previous kept score.
    """
    if not docs:
        return []

    filtered: List[Document] = []
    for doc in docs:
        score = (doc.metadata or {}).get("rerank_score")
        if not isinstance(score, (int, float)):
            # Rerank was skipped or failed for this chunk; without a score we
            # can't apply thresholds, so keep first-stage order as a fallback.
            filtered.append(doc)
            continue

        if not filtered:
            # The best available match sets the bar: if even #1 is below
            # min_score, nothing in the pool is trustworthy enough to ground
            # an answer, so return [] and let the caller say "no relevant docs".
            if score < min_score:
                _debug_print(
                    f"filter: top score {score:.4f} < min {min_score:.4f}; keeping 0"
                )
                return []
            filtered.append(doc)
            continue

        prev_score = (filtered[-1].metadata or {}).get("rerank_score")
        if not isinstance(prev_score, (int, float)):
            # Mixed scored/unscored list (shouldn't happen in normal flow);
            # keep going rather than drop evidence we can't compare.
            filtered.append(doc)
            continue

        if score < min_score:
            # Scores are descending; once we hit the floor, the rest will too.
            _debug_print(
                f"[filter] score {score:.4f} < min {min_score:.4f} for {doc.metadata.get("title")}; "
                f"keeping the top {len(filtered)}"
            )
            break
        # if gap_ratio > 0 and score < prev_score * gap_ratio:
        #     # A sharp drop (e.g. 28% → 7%) marks the boundary between useful
        #     # context and the long tail of weak first-stage candidates.
        #     _debug_print(
        #         f"filter: score {score:.4f} < {gap_ratio:.0%} of prev "
        #         f"{prev_score:.4f}; keeping {len(filtered)}"
        #     )
        #     break
        filtered.append(doc)

    if _debug_enabled() and len(filtered) != len(docs):
        _debug_print(
            f"[filter] From {len(docs)} to {len(filtered)} "
            f"(min_score={min_score:.4f}, gap_ratio={gap_ratio:.2f})"
        )

    return filtered

def rerank_documents(
    query: str,
    docs: List[Document],
    *,
    top_n: int,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Document]:
    """
    Re-rank *docs* for *query* using Discovery Engine.

    Falls back to the incoming order when re-ranking is disabled, misconfigured,
    or the API call fails so retrieval never blocks generation.

    There's a setting (`"ignoreRecordDetailsInResponse": true,`) that specifies whether you want just the ID of the record returned by the API 
    or if you want the record title and content returned as well. By default, the full record is returned. 
    In case we want to reduce the size of the response payload, this might be interesting.

    Args:
        query: The query to re-rank the documents for.
        docs: The documents to re-rank.
        top_n: The maximum number of records that you want the ranking API to return. By default, all records are returned; however, you can use thetopN field to return fewer records. All records are ranked regardless of what value is set.
        project_id: The Google Cloud project ID for the Discovery Engine Ranking API.
        location: The location for the Discovery Engine Ranking API.
        model: This specifies the model to be used for ranking the documents. If no model is specified, then `semantic-ranker-default@latest` is used

    Returns:
        The re-ranked documents.
    """
    if not docs:
        return []
    if len(docs) == 1:
        return docs

    enabled, default_project, default_location, default_model, _ = get_rerank_config()
    if not enabled:
        return docs[:top_n]

    project = (project_id or default_project).strip()
    if not project:
        _debug_print(
            "RAG_RERANK_ENABLED but GOOGLE_CLOUD_PROJECT is unset; keeping retrieval order"
        )
        return docs[:top_n]

    loc = location or default_location
    rank_model = model or default_model
    candidates = docs[:DISCOVERY_ENGINE_MAX_RECORDS]

    try:
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = discoveryengine.RankServiceClient()
        ranking_config = client.ranking_config_path(
            project=project,
            location=loc,
            ranking_config="default_ranking_config",
        )

        docs_by_id = {_doc_record_id(doc): doc for doc in candidates}
        records = [
            discoveryengine.RankingRecord(
                id=record_id,
                title=_record_title(doc),
                content=doc.page_content[:MAX_RECORD_CONTENT_CHARS], # If the combined length of the title and content exceeds the model's token limit (starting from version 004, this limit increases to 1024 tokens), the extra content is truncated.
            )
            for record_id, doc in docs_by_id.items()
        ]

        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model=rank_model,
            top_n=min(top_n, len(records)),
            query=query,
            records=records,
        )
        response = client.rank(request=request)
    except Exception as exc:
        _debug_print(f"Discovery Engine rerank failed: {type(exc).__name__}: {exc}")
        return docs[:top_n]

    ranked: List[Document] = []
    # The ranking API returns a ranked list of records with following outputs:
    # - Score: a float value between 0 and 1 that indicates relevance of the record.
    # - ID: the unique ID of the record.
    # - If requested, the full object: the ID, title, and content.
    for record in response.records:
        doc = docs_by_id.get(record.id)
        if doc is None:
            continue
        meta = dict(doc.metadata or {})
        # Add rerank score to the metadata of the document.
        meta["rerank_score"] = record.score
        ranked.append(Document(page_content=doc.page_content, metadata=meta))

    if _debug_enabled():
        _debug_print(
            f"reranked {len(candidates)} -> {len(ranked)} "
            f"(model={rank_model!r}, top_n={top_n})"
        )
        for i, doc in enumerate(ranked):
            score = (doc.metadata or {}).get("rerank_score")
            score_text = f" score={score:.4f}" if isinstance(score, (int, float)) else ""
            _debug_print(f"  #{i + 1}{score_text} {_record_title(doc)}")

    if not ranked:
        _debug_print("Warning: reranked 0 documents; keeping retrieval order")

    result = ranked if ranked else docs[:top_n]
    if result and any(
        isinstance((doc.metadata or {}).get("rerank_score"), (int, float))
        for doc in result
    ):
        min_score, gap_ratio = get_rerank_filter_config()
        result = filter_reranked_documents(
            result,
            min_score=min_score,
            gap_ratio=gap_ratio,
        )

    (_debug_print(f"Final set of reranked documents: \n\t{doc.metadata.get("title")} ({doc.metadata.get("rerank_score") * 100}%)") 
        for doc in result if _debug_enabled())

    return result
