from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_core.documents import Document

# Minimal loader dependencies
# pip install langchain-community
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)


@dataclass(frozen=True)
class LoadedCorpus:
    name: str
    documents: List[Document]


def _pick_loader(path: Path):
    """
    Pick a loader based on file extension.
    Supported: .txt, .md, .pdf, .docx
    """
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return UnstructuredWordDocumentLoader(str(path))
    raise ValueError(f"Unsupported file type: {ext} for {path.name}")


def _discover_files(data_dir: Path, stem_prefix: str) -> List[Path]:
    """
    Finds files in data_dir that match a prefix like:
      ai_test_bug_report.*
      ai_test_user_feedback.*
    Returns sorted paths for deterministic indexing.
    """
    candidates = sorted(data_dir.glob(f"{stem_prefix}.*"))
    return [p for p in candidates if p.is_file()]


def load_corpus(
    data_dir: Path,
    stem_prefix: str,
    *,
    corpus_name: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> LoadedCorpus:
    """
    Load a corpus from files that match stem_prefix.* inside data_dir.

    Each loaded Document gets metadata:
      - source: corpus_name (or stem_prefix)
      - file_name
      - file_path
      - page (if available)
      - plus extra_metadata (if provided)
    """
    files = _discover_files(data_dir, stem_prefix)
    if not files:
        raise FileNotFoundError(
            f"No files found for prefix '{stem_prefix}.*' in {data_dir.resolve()}"
        )

    name = corpus_name or stem_prefix
    md_extra = extra_metadata or {}

    all_docs: List[Document] = []
    for fp in files:
        loader = _pick_loader(fp)
        docs = loader.load()

        # Normalize metadata
        for d in docs:
            d.metadata = dict(d.metadata or {})
            d.metadata.update(
                {
                    "source": name,
                    "file_name": fp.name,
                    "file_path": str(fp.resolve()),
                }
            )
            d.metadata.update(md_extra)
        all_docs.extend(docs)

    return LoadedCorpus(name=name, documents=all_docs)


def load_all_corpora(data_dir: Path) -> List[LoadedCorpus]:
    """
    Convenience helper for this test:
      - ai_test_bug_report.*
      - ai_test_user_feedback.*
    """
    bug = load_corpus(data_dir, "ai_test_bug_report", corpus_name="ai_test_bug_report")
    feedback = load_corpus(
        data_dir, "ai_test_user_feedback", corpus_name="ai_test_user_feedback"
    )
    return [bug, feedback]


def flatten_documents(corpora: Sequence[LoadedCorpus]) -> List[Document]:
    """
    Merge multiple corpora into a single list of Documents.
    """
    out: List[Document] = []
    for c in corpora:
        out.extend(c.documents)
    return out
