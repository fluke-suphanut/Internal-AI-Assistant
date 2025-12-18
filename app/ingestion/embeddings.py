from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns an OpenAI embeddings client (text-embedding-3-small by default).
    Requires OPENAI_API_KEY in environment/.env.
    """
    s = get_settings()
    if not s.is_openai_configured:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or environment variables."
        )

    return OpenAIEmbeddings(
        model=s.OPENAI_EMBEDDING_MODEL,
        api_key=s.OPENAI_API_KEY,
    )
