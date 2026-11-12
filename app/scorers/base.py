"""Pluggable scorer architecture.

A scorer receives a ``ScoringContext`` and returns a ``ScoreResult`` with a
normalised score in [0, 1], a pass/fail verdict and a human-readable reason.

Add a scorer by subclassing ``Scorer`` and decorating it:

    @register_scorer("my_type")
    class MyScorer(Scorer):
        async def score(self, ctx): ...

and adding a matching assertion model in ``app/models/suite.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.models.results import LLMResponse, ScoreResult
from app.models.suite import TestCase
from app.utils.jsonparse import lookup_dotted
from app.utils.jsonpath import JSONPathError, find_all

if TYPE_CHECKING:
    from app.config.suite_loader import LoadedSuite
    from app.runner.generation import Generator
    from app.config.settings import Settings


@dataclass
class ScoringContext:
    output: str
    parsed_ok: bool
    parsed: Any
    test: TestCase
    rendered_prompt: str
    response: LLMResponse | None = None
    loaded: "LoadedSuite | None" = None
    generator: "Generator | None" = None
    settings: "Settings | None" = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_ref(self, dotted: str) -> Any:
        """Resolve 'expected.category' / 'vars.ticket' against the test case."""
        return lookup_dotted({"expected": self.test.expected, "vars": self.test.vars}, dotted)


class ScorerInputError(Exception):
    """Raised when a scorer cannot evaluate the output (e.g. path not found)."""


class Scorer(ABC):
    type: str = "base"

    def __init__(self, spec: Any):
        self.spec = spec

    @abstractmethod
    async def score(self, ctx: ScoringContext) -> ScoreResult: ...

    # ---------------------------------------------------------- helpers
    def result(self, score: float, passed: bool, reason: str, **details: Any) -> ScoreResult:
        details.setdefault("name", getattr(self.spec, "name", None) or self.type)
        details.setdefault("weight", getattr(self.spec, "weight", 1.0))
        return ScoreResult(
            scorer_type=self.type,
            score=float(max(0.0, min(1.0, score))),
            passed=passed,
            reason=reason,
            details=details,
        )

    def target(self, ctx: ScoringContext) -> Any:
        """The value to score: the raw output, or the JSONPath selection from it."""
        path = getattr(self.spec, "path", None)
        if not path:
            return ctx.output
        if not ctx.parsed_ok:
            raise ScorerInputError(f"output is not valid JSON, cannot evaluate {path}")
        try:
            matches = find_all(ctx.parsed, path)
        except JSONPathError as exc:
            raise ScorerInputError(str(exc)) from exc
        if not matches:
            raise ScorerInputError(f"{path} not found in output")
        return matches[0]

    def expected_value(self, ctx: ScoringContext, literal_attr: str = "value") -> Any:
        ref = getattr(self.spec, "value_from", None)
        if ref:
            try:
                return ctx.resolve_ref(ref)
            except KeyError as exc:
                raise ScorerInputError(f"value_from '{ref}' not found in test case") from exc
        return getattr(self.spec, literal_attr)


_REGISTRY: dict[str, type[Scorer]] = {}


def register_scorer(type_name: str):
    def deco(cls: type[Scorer]) -> type[Scorer]:
        cls.type = type_name
        _REGISTRY[type_name] = cls
        return cls

    return deco


def build_scorer(spec: Any) -> Scorer:
    # Import side-effect registers the built-in scorers.
    from app.scorers import deterministic, judge, semantic  # noqa: F401

    try:
        return _REGISTRY[spec.type](spec)
    except KeyError as exc:
        raise ValueError(f"No scorer registered for type '{spec.type}'") from exc


def registered_scorers() -> list[str]:
    from app.scorers import deterministic, judge, semantic  # noqa: F401

    return sorted(_REGISTRY)


async def run_scorer(scorer: Scorer, ctx: ScoringContext) -> ScoreResult:
    """Run a scorer, turning input errors into a failed result instead of a crash."""
    try:
        return await scorer.score(ctx)
    except ScorerInputError as exc:
        return scorer.result(0.0, False, str(exc))
