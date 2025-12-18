from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from app.core.config import get_settings


class FAISSStore:
    """
    Thin wrapper around a persisted FAISS index on disk.
    Expects the index to be built via app/ingestion/build_index.py (save_local()).
    """

    def __init__(self, index_dir: Optional[Path] = None):
        s = get_settings()
        self.index_dir: Path = index_dir or s.FAISS_INDEX_DIR

    def exists(self) -> bool:
        """
        FAISS save_local produces files like:
          - index.faiss
          - index.pkl
        """
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def load(self, embeddings: Embeddings) -> FAISS:
        """
        Load the FAISS index from disk.
        NOTE: allow_dangerous_deserialization=True is required due to pickle usage in FAISS docstore.
        Only use with trusted local files you created.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_dir.resolve()}. "
                f"Build it first: python -m app.ingestion.build_index"
            )

        return FAISS.load_local(
            folder_path=str(self.index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
