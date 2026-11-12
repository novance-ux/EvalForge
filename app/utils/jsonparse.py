"""Helpers for parsing JSON produced by LLMs."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_json_output(text: str) -> tuple[bool, Any]:
    """Parse model output as JSON.

    Surrounding whitespace and a single Markdown code fence (```json ... ```)
    are tolerated because many models add them; any other surrounding prose
    makes the output invalid. Returns ``(ok, value)``.
    """
    if text is None:
        return False, None
    candidate = text.strip()
    m = _FENCE.match(candidate)
    if m:
        candidate = m.group(1)
    try:
        return True, json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return False, None


def lookup_dotted(data: dict[str, Any], dotted: str) -> Any:
    """Resolve ``expected.category`` style references against a dict."""
    node: Any = data
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(f"'{dotted}' not found")
    return node
