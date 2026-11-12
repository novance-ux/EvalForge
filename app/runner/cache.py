"""Response cache backed by the results database.

The cache key is a SHA-256 over everything that can change a response:
provider, model, the exact rendered prompt, generation parameters, the input
variables and the sample index (so repeated samples at temperature > 0 are
cached independently instead of all returning the first sample).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError

from app.models.results import CacheStats, LLMResponse
from app.storage.db import CacheEntry, Database
from app.utils.misc import stable_hash


def cache_key(
    provider: str,
    model: str,
    prompt: str,
    parameters: dict[str, Any],
    input_vars: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> str:
    return stable_hash(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "parameters": parameters,
            "input": input_vars or {},
            "sample": sample_index,
        }
    )


class ResponseCache:
    def __init__(self, db: Database, enabled: bool = True):
        self.db = db
        self.enabled = enabled
        self.stats = CacheStats()

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        with self.db.session() as s:
            entry = s.get(CacheEntry, key)
            if entry is None:
                self.stats.misses += 1
                return None
            entry.hit_count += 1
            s.commit()
            self.stats.hits += 1
            self.stats.cost_saved += entry.cost
            resp = LLMResponse.model_validate(entry.response)
            resp.cached = True
            return resp

    def put(self, key: str, response: LLMResponse) -> None:
        if not self.enabled:
            return
        with self.db.session() as s:
            s.merge(
                CacheEntry(
                    key=key,
                    provider=response.provider,
                    model=response.model,
                    response=response.model_dump(exclude={"cached", "raw"}),
                    cost=response.cost,
                )
            )
            try:
                s.commit()
            except IntegrityError:  # concurrent writer stored it first; that's fine
                s.rollback()

    def clear(self) -> int:
        with self.db.session() as s:
            n = s.query(CacheEntry).delete()
            s.commit()
            return n
