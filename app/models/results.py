"""Plain data objects passed between the runner, scorers and storage layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Normalised response returned by every provider adapter."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    cached: bool = False
    raw: dict[str, Any] | None = Field(default=None, exclude=True)


class ScoreResult(BaseModel):
    """Result of a single scorer. `score` is normalised to [0, 1]."""

    scorer_type: str
    score: float
    passed: bool
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class TestOutcome(BaseModel):
    """Everything recorded for one (test case, model, repeat) execution."""

    __test__ = False  # not a pytest test class

    test_id: str
    model: str
    provider: str
    repeat_index: int = 0
    vars: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    rendered_prompt: str
    output: str = ""
    error: str | None = None
    score: float = 0.0
    passed: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cached: bool = False
    json_valid: bool = False
    predicted_label: str | None = None
    expected_label: str | None = None
    scorer_results: list[ScoreResult] = Field(default_factory=list)


class CacheStats(BaseModel):
    hits: int = 0
    misses: int = 0
    cost_saved: float = 0.0
