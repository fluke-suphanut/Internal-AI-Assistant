from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ingestion.loader import load_all_corpora, flatten_documents
from app.ingestion.splitter import split_documents
from app.ingestion.embeddings import get_embeddings


logger = setup_logging()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_manifest(
    *,
    total_files: int,
    total_docs: int,
    total_chunks: int,
    embedding_model: str,
    chat_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    return {
        "built_at_utc": _utc_now_iso(),
        "total_files": total_files,
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "embedding_model": embedding_model,
        "chat_model": chat_model,
        "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
    }


def _count_unique_files(docs: List[Document]) -> int:
    files = set()
    for d in docs:
        fp = (d.metadata or {}).get("file_path") or (d.metadata or {}).get("file_name")
        if fp:
            files.add(fp)
    return len(files)


def build_faiss_index(
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    index_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Build and persist a FAISS index from:
      - data/ai_test_bug_report.*
      - data/ai_test_user_feedback.*

    Persists:
      - FAISS index folder to STORAGE_DIR/faiss_index/
      - manifest.json to STORAGE_DIR/manifest.json

    Returns manifest dict.
    """
    s = get_settings()
    index_dir = index_dir or s.FAISS_INDEX_DIR
    manifest_path = manifest_path or (s.STORAGE_DIR / "manifest.json")

    logger.info("Loading corpora from: %s", s.DATA_DIR.resolve())
    corpora = load_all_corpora(s.DATA_DIR)
    raw_docs = flatten_documents(corpora)
    logger.info("Loaded documents: %d", len(raw_docs))

    # Split into chunks
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Split into chunks: %d", len(chunks))

    # Create vector store
    embeddings = get_embeddings()
    logger.info("Building FAISS index with embedding model: %s", s.OPENAI_EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist index
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving FAISS index to: %s", index_dir.resolve())
    vectorstore.save_local(str(index_dir))

    # Write manifest for traceability
    manifest = _build_manifest(
        total_files=_count_unique_files(raw_docs),
        total_docs=len(raw_docs),
        total_chunks=len(chunks),
        embedding_model=s.OPENAI_EMBEDDING_MODEL,
        chat_model=s.OPENAI_CHAT_MODEL,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote manifest: %s", manifest_path.resolve())

    return manifest


if __name__ == "__main__":
    # Allows: python -m app.ingestion.build_index
    build_faiss_index()
