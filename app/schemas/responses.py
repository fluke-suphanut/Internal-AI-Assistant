from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


ToolName = Literal["internal_qa", "issue_summary"]


class Citation(BaseModel):
    """
    Reference to a retrieved document chunk.
    """
    source: str = Field(
        ...,
        description="Source document name, e.g. ai_test_bug_report or ai_test_user_feedback",
        examples=["ai_test_bug_report"],
    )
    doc_id: Optional[str] = Field(
        None,
        description="Document identifier (if available)",
        examples=["bug_report_001"],
    )
    chunk_id: Optional[str] = Field(
        None,
        description="Chunk identifier inside the document",
        examples=["chunk_12"],
    )
    snippet: Optional[str] = Field(
        None,
        description="Short excerpt from the retrieved text chunk",
        examples=["Email notifications are delayed during peak hours..."],
    )


class InternalQAOutput(BaseModel):
    """
    Structured output for Internal Q&A tool.
    """
    answer: str = Field(
        ...,
        description="LLM-generated answer based on retrieved documents",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of document chunks used to generate the answer",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Estimated confidence level of the answer",
    )


class IssueSummaryOutput(BaseModel):
    """
    Structured output for Issue Summary tool.
    """
    reported_issues: List[str] = Field(
        default_factory=list,
        description="List of reported issues extracted from the text",
        examples=[["Delayed email notifications", "Missing notifications"]],
    )
    affected_components: List[str] = Field(
        default_factory=list,
        description="Impacted features or system components",
        examples=[["Email Service", "Notification Queue"]],
    )
    severity: Literal["Low", "Medium", "High", "Critical", "Unknown"] = Field(
        default="Unknown",
        description="Estimated severity of the issue",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional context or observations from the summary",
    )


class AgentResponse(BaseModel):
    """
    Unified response returned by the AI Agent.
    """
    request_id: str = Field(
        ...,
        description="Unique identifier for the request",
    )
    timestamp: str = Field(
        ...,
        description="UTC timestamp when the response was generated",
    )
    tool_selected: ToolName = Field(
        ...,
        description="Tool chosen by the agent to handle the request",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the agent selected this tool",
    )
    tool_output: Dict[str, Any] = Field(
        ...,
        description="Structured output returned by the selected tool",
    )
