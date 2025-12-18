from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingestion.build_index import build_faiss_index
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from internal documents")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Chunk size for text splitting (default: 900)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for text splitting (default: 150)",
    )
    args = parser.parse_args()

    settings = get_settings()

    logger.info("Starting ingestion pipeline")
    logger.info("Data directory: %s", settings.DATA_DIR.resolve())
    logger.info("FAISS index directory: %s", settings.FAISS_INDEX_DIR.resolve())

    manifest = build_faiss_index(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info("Ingestion completed successfully")
    logger.info("Manifest summary:\n%s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
