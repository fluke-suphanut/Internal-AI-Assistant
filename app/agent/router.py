from __future__ import annotations

import json
from typing import Tuple

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.agent.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT_TEMPLATE
from app.utils.json_guard import parse_json_object


def route_tool(user_text: str) -> Tuple[str, str]:
    """
    LLM-based router.
    Returns (tool_selected, reasoning).

    tool_selected: "internal_qa" | "issue_summary"
    """
    s = get_settings()
    if not s.is_openai_configured:
        raise RuntimeError("OPENAI_API_KEY is not set. Cannot run router.")

    llm = ChatOpenAI(
        model=s.OPENAI_CHAT_MODEL,  # e.g., gpt-4o-mini
        api_key=s.OPENAI_API_KEY,
        temperature=0,
    )

    user_prompt = ROUTER_USER_PROMPT_TEMPLATE.format(user_text=user_text.strip())
    msg = llm.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    ).content

    data = parse_json_object(msg)

    tool = data.get("tool_selected", "internal_qa")
    reasoning = data.get("reasoning", "").strip() or "No reasoning provided."

    if tool not in ("internal_qa", "issue_summary"):
        tool = "internal_qa"
        reasoning = "Router produced an invalid tool name; defaulted to internal_qa."

    return tool, reasoning
