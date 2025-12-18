from __future__ import annotations

import json
import re
from typing import Any, Dict


class JSONGuardError(ValueError):
    pass


def _extract_json_object(text: str) -> str:
    """
    Extract the first JSON object from arbitrary text.
    Handles cases where the model returns extra text around JSON.
    """
    if not text or not text.strip():
        raise JSONGuardError("Empty response; cannot parse JSON.")

    s = text.strip()

    # If it's already valid JSON, return it
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # Try to find the first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JSONGuardError("No JSON object boundaries found in response.")

    candidate = s[start : end + 1].strip()

    # Remove common markdown fences if present
    candidate = re.sub(r"^\s*```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```\s*$", "", candidate)

    return candidate


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from model output robustly:
      - Attempts direct JSON parse
      - Extracts JSON block if surrounded by extra text/fences
      - Raises JSONGuardError on failure

    Returns a dict (must be a JSON object).
    """
    candidate = _extract_json_object(text)

    try:
        data = json.loads(candidate)
    except Exception as e:
        raise JSONGuardError(f"Failed to parse JSON: {e}")

    if not isinstance(data, dict):
        raise JSONGuardError("Parsed JSON is not an object (dict).")

    return data
