from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.schemas.responses import IssueSummaryOutput


def issue_summary_tool(issue_text: str) -> IssueSummaryOutput:
    """
    Use LLM to summarize an issue into structured fields.
    """
    s = get_settings()

    llm = ChatOpenAI(
        model=s.OPENAI_CHAT_MODEL,
        api_key=s.OPENAI_API_KEY,
        temperature=0,
    )

    prompt = f"""
You are an AI assistant for product and engineering teams.

Summarize the issue text into the following JSON format ONLY:

{{
  "reported_issues": [string],
  "affected_components": [string],
  "severity": "Low | Medium | High | Critical | Unknown",
  "notes": string
}}

Issue text:
{issue_text}
"""

    response = llm.invoke(prompt).content.strip()

    # Minimal safe parse
    try:
        import json
        data = json.loads(response)
    except Exception:
        # Fallback if LLM response is malformed
        return IssueSummaryOutput(
            reported_issues=[],
            affected_components=[],
            severity="Unknown",
            notes="Failed to parse structured summary from LLM.",
        )

    return IssueSummaryOutput(
        reported_issues=data.get("reported_issues", []),
        affected_components=data.get("affected_components", []),
        severity=data.get("severity", "Unknown"),
        notes=data.get("notes"),
    )
