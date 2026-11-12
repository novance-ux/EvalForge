"""A small, dependency-free JSONPath subset.

Supported syntax (enough for assertions on model output):

    $                 the whole document
    $.a.b             object keys
    $['a b']          quoted keys
    $.items[0]        list index (negative indices allowed)
    $.items[*].name   wildcard over a list or object values
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(
    r"""
    \.(?P<key>[A-Za-z_][\w\-]*)        # .key
  | \[(?P<index>-?\d+)\]               # [0]
  | \[(?P<star>\*)\]|\.(?P<dotstar>\*) # [*] or .*
  | \[['"](?P<qkey>[^'"]+)['"]\]       # ['key']
    """,
    re.VERBOSE,
)

_MISSING = object()


class JSONPathError(ValueError):
    pass


def parse_path(path: str) -> list[tuple[str, Any]]:
    if not path or not path.startswith("$"):
        raise JSONPathError(f"JSONPath must start with '$': {path!r}")
    tokens: list[tuple[str, Any]] = []
    pos = 1
    while pos < len(path):
        m = _TOKEN.match(path, pos)
        if not m:
            raise JSONPathError(f"Invalid JSONPath {path!r} near position {pos}: {path[pos:]!r}")
        if m.group("key") is not None:
            tokens.append(("key", m.group("key")))
        elif m.group("qkey") is not None:
            tokens.append(("key", m.group("qkey")))
        elif m.group("index") is not None:
            tokens.append(("index", int(m.group("index"))))
        else:
            tokens.append(("star", None))
        pos = m.end()
    return tokens


def find_all(data: Any, path: str) -> list[Any]:
    """Return every value matched by ``path`` (empty list when nothing matches)."""
    current = [data]
    for kind, arg in parse_path(path):
        nxt: list[Any] = []
        for node in current:
            if kind == "key" and isinstance(node, dict) and arg in node:
                nxt.append(node[arg])
            elif kind == "index" and isinstance(node, list) and -len(node) <= arg < len(node):
                nxt.append(node[arg])
            elif kind == "star":
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
        current = nxt
    return current


def extract(data: Any, path: str, default: Any = _MISSING) -> Any:
    """Return the first match of ``path``; raise KeyError if missing and no default given."""
    matches = find_all(data, path)
    if matches:
        return matches[0]
    if default is _MISSING:
        raise KeyError(f"JSONPath {path} matched nothing")
    return default
