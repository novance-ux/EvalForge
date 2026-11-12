"""Small shared helpers: hashing, git metadata, time."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from typing import Any


def stable_hash(obj: Any, length: int | None = None) -> str:
    """SHA-256 of a JSON-serialisable object with sorted keys."""
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def git_commit() -> str | None:
    """Current git commit hash, or None outside a repository / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for providers that omit usage."""
    return max(1, (len(text) + 3) // 4) if text else 0
