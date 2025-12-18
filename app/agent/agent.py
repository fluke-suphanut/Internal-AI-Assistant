from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.agent.router import route_tool
from app.core.config import get_settings
from app.schemas.responses import AgentResponse
from app.tools.internal_qa_tool import internal_qa_tool
from app.tools.issue_summary_tool import issue_summary_tool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AIAgent:
    """
    Orchestrates:
      1) route user input -> tool
      2) call selected tool
      3) return a unified AgentResponse (structured)
    """

    def __init__(self):
        self.settings = get_settings()

    def run(
        self,
        *,
        user_text: str,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AgentResponse:
        rid = request_id or str(uuid.uuid4())
        ts = _utc_now_iso()

        tool_selected, reasoning = route_tool(user_text)

        # Run tool
        if tool_selected == "internal_qa":
            k = top_k or self.settings.DEFAULT_TOP_K
            tool_out = internal_qa_tool(user_text, top_k=k).model_dump()
        else:
            tool_out = issue_summary_tool(user_text).model_dump()

        return AgentResponse(
            request_id=rid,
            timestamp=ts,
            tool_selected=tool_selected,  # type: ignore
            reasoning=reasoning,
            tool_output=tool_out,
        )

    def run_issue_summary(
        self,
        *,
        issue_text: str,
        request_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Convenience method for explicit summarize endpoint.
        """
        rid = request_id or str(uuid.uuid4())
        ts = _utc_now_iso()

        tool_out = issue_summary_tool(issue_text).model_dump()

        return AgentResponse(
            request_id=rid,
            timestamp=ts,
            tool_selected="issue_summary",  # type: ignore
            reasoning="Explicit summarize endpoint.",
            tool_output=tool_out,
        )
