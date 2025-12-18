from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.core.config import get_settings
from app.ingestion.embeddings import get_embeddings
from app.retriever.faiss_store import FAISSStore
from app.retriever.search import similarity_search
from app.schemas.responses import InternalQAOutput, Citation


def _build_context(docs: List[Document]) -> str:
    """
    Build a context string from retrieved documents.
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        text = d.page_content.strip()
        blocks.append(f"[{i}] ({source}:{chunk_id})\n{text}")
    return "\n\n".join(blocks)


def internal_qa_tool(query: str, top_k: int = 5) -> InternalQAOutput:
    """
    Perform:
      1) FAISS similarity search
      2) LLM answer grounded in retrieved context
      3) Return structured output
    """
    s = get_settings()

    # Load FAISS
    embeddings = get_embeddings()
    store = FAISSStore()
    vectorstore = store.load(embeddings)

    # Retrieve documents
    docs = similarity_search(vectorstore, query, top_k=top_k)

    if not docs:
        return InternalQAOutput(
            answer="No relevant information was found in the internal documents.",
            citations=[],
            confidence="low",
        )

    context = _build_context(docs)

    llm = ChatOpenAI(
        model=s.OPENAI_CHAT_MODEL,
        api_key=s.OPENAI_API_KEY,
        temperature=0,
    )

    prompt = f"""
You are an internal AI assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{query}

Answer concisely and clearly.
"""

    answer = llm.invoke(prompt).content.strip()

    citations = []
    for d in docs:
        md = d.metadata or {}
        citations.append(
            Citation(
                source=md.get("source", "unknown"),
                doc_id=md.get("file_name"),
                chunk_id=md.get("chunk_id"),
                snippet=d.page_content[:200],
            )
        )

    confidence = "high" if len(docs) >= 3 else "medium"

    return InternalQAOutput(
        answer=answer,
        citations=citations,
        confidence=confidence,
    )
