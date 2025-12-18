from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def _metadata_matches(doc: Document, filters: Dict[str, Any]) -> bool:
    md = doc.metadata or {}
    for k, v in filters.items():
        if md.get(k) != v:
            return False
    return True


def similarity_search(
    vectorstore: FAISS,
    query: str,
    *,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Basic similarity search. If filters are provided, we:
      1) retrieve more candidates (top_k * 4)
      2) filter by exact-match metadata
      3) return top_k filtered results

    This is a simple, reliable approach for FAISS (since FAISS doesn't natively filter).
    """
    if not query.strip():
        return []

    filters = filters or {}
    fetch_k = max(top_k * 4, top_k)

    # Grab candidates
    candidates = vectorstore.similarity_search(query, k=fetch_k)

    if not filters:
        return candidates[:top_k]

    filtered: List[Document] = []
    for d in candidates:
        if _metadata_matches(d, filters):
            filtered.append(d)
        if len(filtered) >= top_k:
            break

    return filtered


def similarity_search_with_scores(
    vectorstore: FAISS,
    query: str,
    *,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """
    Same as similarity_search, but includes distances/scores from FAISS.

    Note: LangChain FAISS returns (doc, score) where score meaning can vary
    (often smaller distance is better). Treat as a relative ranking signal.
    """
    if not query.strip():
        return []

    filters = filters or {}
    fetch_k = max(top_k * 4, top_k)

    candidates = vectorstore.similarity_search_with_score(query, k=fetch_k)

    if not filters:
        return candidates[:top_k]

    filtered: List[Tuple[Document, float]] = []
    for d, score in candidates:
        if _metadata_matches(d, filters):
            filtered.append((d, score))
        if len(filtered) >= top_k:
            break

    return filtered
