from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request schema for internal document Q&A.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question for document-based Q&A",
        examples=["What are the issues reported on email notification?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from FAISS vector search",
        examples=[5],
    )


class SummarizeRequest(BaseModel):
    """
    Request schema for issue summarization.
    """
    issue_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw issue or bug description text to summarize",
        examples=[
            "Users report that email notifications are delayed by several hours "
            "and sometimes not sent at all."
        ],
    )
