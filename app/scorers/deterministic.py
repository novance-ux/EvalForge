"""Deterministic (rule-based) scorers. No LLM calls, fully reproducible."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from app.config.suite_loader import schema_key
from app.models.results import ScoreResult
from app.scorers.base import Scorer, ScorerInputError, ScoringContext, register_scorer


def normalize(value: Any, case_sensitive: bool = False) -> str:
    """Whitespace-collapsed, optionally lower-cased, quote/period-trimmed text."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = " ".join(text.split()).strip().strip("\"'").rstrip(".").strip()
    return text if case_sensitive else text.lower()


@register_scorer("exact_match")
class ExactMatchScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        actual = normalize(self.target(ctx), self.spec.case_sensitive)
        expected = normalize(self.expected_value(ctx), self.spec.case_sensitive)
        ok = actual == expected
        return self.result(
            1.0 if ok else 0.0, ok,
            "Exact match" if ok else f"Expected '{expected}', got '{actual[:80]}'",
            expected=expected, actual=actual[:500],
        )


@register_scorer("equals")
class JsonPathEqualsScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        actual = self.target(ctx)
        expected = self.expected_value(ctx)
        if isinstance(actual, str) and isinstance(expected, str):
            ok = normalize(actual, self.spec.case_sensitive) == normalize(expected, self.spec.case_sensitive)
        else:
            ok = actual == expected
        where = self.spec.path or "output"
        return self.result(
            1.0 if ok else 0.0, ok,
            f"{where} == {expected!r}" if ok else f"{where} is {actual!r}, expected {expected!r}",
            expected=expected, actual=actual,
        )


@register_scorer("contains")
class ContainsScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        haystack = normalize(self.target(ctx), self.spec.case_sensitive)
        needle = normalize(self.expected_value(ctx), self.spec.case_sensitive)
        ok = needle in haystack
        return self.result(
            1.0 if ok else 0.0, ok,
            f"Output contains '{needle}'" if ok else f"'{needle}' not found in output",
        )


@register_scorer("regex")
class RegexScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        flags = 0
        for f in self.spec.flags:
            flags |= getattr(re, f)
        try:
            pattern = re.compile(self.spec.pattern, flags)
        except re.error as exc:
            raise ScorerInputError(f"invalid regex {self.spec.pattern!r}: {exc}") from exc
        target = self.target(ctx)
        text = target if isinstance(target, str) else json.dumps(target)
        m = pattern.search(text)
        return self.result(
            1.0 if m else 0.0, bool(m),
            f"Matched /{self.spec.pattern}/: '{m.group(0)[:60]}'" if m else f"No match for /{self.spec.pattern}/",
        )


@register_scorer("json_schema")
class JsonSchemaScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.parsed_ok and not self.spec.path:
            return self.result(0.0, False, "Output is not valid JSON")
        schema = self._schema(ctx)
        instance = self.target(ctx) if self.spec.path else ctx.parsed
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        if not errors:
            return self.result(1.0, True, "Output matches JSON schema")
        first = errors[0]
        loc = "/".join(map(str, first.path)) or "<root>"
        return self.result(
            0.0, False, f"Schema violation at {loc}: {first.message}",
            errors=[e.message for e in errors[:5]],
        )

    def _schema(self, ctx: ScoringContext) -> dict:
        if isinstance(self.spec.schema_, dict):
            return self.spec.schema_
        if ctx.loaded is not None and schema_key(self.spec.schema_) in ctx.loaded.schemas:
            return ctx.loaded.schemas[schema_key(self.spec.schema_)]
        from pathlib import Path

        return json.loads(Path(self.spec.schema_).read_text(encoding="utf-8"))


@register_scorer("numeric_tolerance")
class NumericToleranceScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        raw = self.target(ctx)
        actual = _to_number(raw)
        if actual is None:
            raise ScorerInputError(f"could not read a number from {str(raw)[:60]!r}")
        expected = float(self.expected_value(ctx, "expected"))
        diff = abs(actual - expected)
        ok = diff <= self.spec.tolerance
        # Partial credit decays linearly to 0 at 2x tolerance.
        span = max(self.spec.tolerance, 1e-9) * 2
        partial = 1.0 if ok else max(0.0, 1 - diff / span)
        return self.result(
            partial, ok,
            f"|{actual} - {expected}| = {diff:g} {'<=' if ok else '>'} tolerance {self.spec.tolerance:g}",
            actual=actual, expected=expected,
        )


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if m:
            return float(m.group(0).replace(",", ""))
    return None


@register_scorer("latency_under_ms")
class LatencyScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        if ctx.response is None:
            return self.result(0.0, False, "No response recorded")
        latency = ctx.response.latency_ms
        ok = latency <= self.spec.value
        return self.result(
            1.0 if ok else 0.0, ok,
            f"Latency {latency:.0f} ms {'<=' if ok else '>'} limit {self.spec.value:.0f} ms"
            + (" (cached response: original latency)" if ctx.response.cached else ""),
            latency_ms=latency,
        )
