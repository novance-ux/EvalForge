"""Cost tracking driven by the pricing file (config/pricing.yaml).

Prices are never hard-coded in code. A model that is not in the pricing file
is costed at $0 and reported as *unpriced* so the user notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m_tokens: float
    output_per_1m_tokens: float


class PricingTable:
    def __init__(self, prices: dict[str, ModelPrice] | None = None):
        self.prices = prices or {}
        self.unpriced: set[str] = set()

    @classmethod
    def from_file(cls, path: str | Path) -> "PricingTable":
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        prices = {
            name: ModelPrice(
                float(cfg.get("input_per_1m_tokens", 0)), float(cfg.get("output_per_1m_tokens", 0))
            )
            for name, cfg in (data.get("models") or {}).items()
        }
        return cls(prices)

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        return self.prices.get(f"{provider}/{model}") or self.prices.get(model)

    def cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.lookup(provider, model)
        if price is None:
            self.unpriced.add(f"{provider}/{model}")
            return 0.0
        return (
            input_tokens * price.input_per_1m_tokens + output_tokens * price.output_per_1m_tokens
        ) / 1_000_000

    def breakdown(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> dict:
        price = self.lookup(provider, model) or ModelPrice(0, 0)
        inp = input_tokens * price.input_per_1m_tokens / 1_000_000
        out = output_tokens * price.output_per_1m_tokens / 1_000_000
        return {"input_cost": inp, "output_cost": out, "total_cost": inp + out}
