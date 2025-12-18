from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, List, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Pydantic Schemas
ToolName = Literal["internal_qa", "issue_summary"]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question to the internal AI assistant")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve from vector search (internal_qa only)")


class SummarizeRequest(BaseModel):
    issue_text: str = Field(..., min_length=1, description="Raw issue text to summarize")


class Citation(BaseModel):
    source: str = Field(..., description="Document source name, e.g., ai_test_bug_report")
    doc_id: Optional[str] = Field(None, description="Document identifier (if available)")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier (if available)")
    snippet: Optional[str] = Field(None, description="Short excerpt from retrieved text")


class ToolOutputInternalQA(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class ToolOutputIssueSummary(BaseModel):
    reported_issues: List[str] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    severity: Literal["Low", "Medium", "High", "Critical", "Unknown"] = "Unknown"
    notes: Optional[str] = None


class AgentResponse(BaseModel):
    request_id: str
    timestamp: str
    tool_selected: ToolName
    reasoning: str
    tool_output: Dict[str, Any]


# App
app = FastAPI(
    title="Internal AI Assistant API",
    description=(
        "API-first internal assistant for product/engineering insights. "
        "Provides document Q&A (vector search + LLM) and issue summarization."
    ),
    version="1.0.0",
)


# Dependency placeholders
def route_tool(query: str) -> tuple[ToolName, str]:
    """
    Minimal router (placeholder).
    Replace with LLM router later (gpt-4o-mini) that outputs JSON:
    { tool_selected: ..., reasoning: ... }
    """
    q = query.lower()
    if any(k in q for k in ["summarize", "summary", "สรุป", "สรุปให้", "issue text"]):
        return "issue_summary", "Detected summarization intent based on keywords."
    return "internal_qa", "Defaulted to document Q&A intent for informational query."


def internal_qa_tool(query: str, top_k: int) -> ToolOutputInternalQA:
    """
    Placeholder for: FAISS vector search + LLM answer.
    Replace with your retriever + gpt-4o-mini answerer.
    """
    # Example placeholder answer
    return ToolOutputInternalQA(
        answer=f"(stub) Answer for: {query}",
        citations=[
            Citation(
                source="ai_test_bug_report",
                doc_id="bug_report_001",
                chunk_id="c12",
                snippet="(stub) relevant chunk snippet..."
            )
        ],
        confidence="medium",
    )


def issue_summary_tool(issue_text: str) -> ToolOutputIssueSummary:
    """
    Placeholder for: LLM summarization (gpt-4o-mini) to structured JSON.
    """
    return ToolOutputIssueSummary(
        reported_issues=["(stub) Issue 1", "(stub) Issue 2"],
        affected_components=["(stub) Component A"],
        severity="Unknown",
        notes="Replace this stub with LLM summarization output.",
    )


# Routes
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AgentResponse)
def ask(payload: AskRequest) -> AgentResponse:
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    try:
        tool_selected, reasoning = route_tool(payload.query)
        if tool_selected == "internal_qa":
            tool_out = internal_qa_tool(payload.query, payload.top_k).model_dump()
        else:
            # If user hits /ask with a summarization query, we still handle gracefully.
            tool_out = issue_summary_tool(payload.query).model_dump()

        return AgentResponse(
            request_id=request_id,
            timestamp=ts,
            tool_selected=tool_selected,
            reasoning=reasoning,
            tool_output=tool_out,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


@app.post("/summarize", response_model=AgentResponse)
def summarize(payload: SummarizeRequest) -> AgentResponse:
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    try:
        tool_selected: ToolName = "issue_summary"
        reasoning = "Explicit summarize endpoint."
        tool_out = issue_summary_tool(payload.issue_text).model_dump()

        return AgentResponse(
            request_id=request_id,
            timestamp=ts,
            tool_selected=tool_selected,
            reasoning=reasoning,
            tool_output=tool_out,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")
