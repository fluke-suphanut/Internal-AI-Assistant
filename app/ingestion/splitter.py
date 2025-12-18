from __future__ import annotations

from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_splitter(
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter tuned for product/bug docs.

    Defaults:
      - chunk_size ~ 900 chars (safe for many embedding models; adjust as needed)
      - overlap ~ 150 chars to preserve context between chunks
    """
    seps = separators or ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=seps,
        length_function=len,
        add_start_index=True,
    )


def split_documents(
    documents: List[Document],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into chunks and add chunk metadata.

    Adds/ensures:
      - chunk_id: stable-ish id based on order within input list
      - parent_source/file_name/page metadata kept from original
    """
    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for i, d in enumerate(chunks):
        d.metadata = dict(d.metadata or {})
        d.metadata["chunk_id"] = d.metadata.get("chunk_id") or f"chunk_{i}"
    return chunks
