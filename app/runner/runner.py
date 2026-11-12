"""Evaluation runner: executes a suite against every configured model.

The runner is provider-agnostic. For each (model, test, repeat) it:
  1. renders the prompt
  2. calls ``Generator.generate`` (cache -> provider -> cost)
  3. runs every scorer
  4. aggregates into a ``TestOutcome``
and finally persists everything to the database.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from app.config.settings import Settings, get_settings
from app.config.suite_loader import LoadedSuite
from app.models.results import CacheStats, TestOutcome
from app.models.suite import ModelSpec, TestCase
from app.providers.base import ProviderError
from app.providers.registry import ProviderRegistry
from app.runner.cache import ResponseCache
from app.runner.cost import PricingTable
from app.runner.generation import Generator
from app.scorers.base import ScoringContext, build_scorer, run_scorer
from app.storage.db import Database
from app.storage.repository import Repository
from app.utils.jsonparse import parse_json_output
from app.utils.jsonpath import extract
from app.utils.misc import git_commit, stable_hash


@dataclass
class RunOptions:
    repeats: int = 1
    use_cache: bool = True
    max_concurrency: int | None = None
    temperature: float | None = None
    label: str | None = None


@dataclass
class RunResult:
    run_id: int
    outcomes: list[TestOutcome]
    cache: CacheStats
    unpriced_models: set[str] = field(default_factory=set)


class EvaluationRunner:
    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
        registry: ProviderRegistry | None = None,
        pricing: PricingTable | None = None,
    ):
        self.settings = settings or get_settings()
        self.db = db
        self.repo = Repository(db)
        self.registry = registry or ProviderRegistry(self.settings)
        self.pricing = pricing or PricingTable.from_file(self.settings.pricing_path)

    async def run(
        self,
        loaded: LoadedSuite,
        options: RunOptions | None = None,
        progress: Callable[[TestOutcome], None] | None = None,
    ) -> RunResult:
        opts = options or RunOptions()
        suite = loaded.suite
        cache = ResponseCache(self.db, enabled=opts.use_cache)
        generator = Generator(
            self.registry, self.pricing, cache, opts.max_concurrency or self.settings.max_concurrency
        )
        temperature = opts.temperature if opts.temperature is not None else suite.parameters.temperature
        params = {"temperature": temperature, "max_tokens": suite.parameters.max_tokens}

        model_specs = [
            ModelSpec(
                provider=m.provider,
                model=self.registry.resolve_model(m.provider, m.model),
                label=m.label,
                parameters=m.parameters,
            )
            for m in suite.models
        ]
        run = self.repo.create_run(
            suite_name=suite.name,
            suite_path=str(loaded.path),
            suite_hash=stable_hash(loaded.raw),
            suite_config=loaded.raw,
            prompt_version=loaded.prompt_version,
            prompt_path=str(loaded.template.path) if loaded.template.path else None,
            prompt_hash=stable_hash(loaded.template.source, 16),
            prompt_template=loaded.template.source,
            models=[m.key() for m in model_specs],
            parameters=params,
            repeats=opts.repeats,
            label=opts.label,
            git_commit=git_commit(),
            status="running",
        )

        async def one(spec: ModelSpec, test: TestCase, repeat: int) -> TestOutcome:
            outcome = await self._evaluate(loaded, generator, spec, test, repeat, temperature)
            if progress:
                progress(outcome)
            return outcome

        jobs = [one(m, t, r) for m in model_specs for t in suite.tests for r in range(opts.repeats)]
        try:
            outcomes = list(await asyncio.gather(*jobs))
            self.repo.save_outcomes(run.id, outcomes)
            self.repo.finish_run(run.id, "completed", cache.stats)
        except BaseException as exc:
            self.repo.finish_run(run.id, "failed", cache.stats, error=repr(exc))
            raise
        finally:
            await self.registry.aclose()
        return RunResult(run.id, outcomes, cache.stats, set(self.pricing.unpriced))

    async def _evaluate(
        self,
        loaded: LoadedSuite,
        generator: Generator,
        spec: ModelSpec,
        test: TestCase,
        repeat: int,
        temperature: float,
    ) -> TestOutcome:
        suite = loaded.suite
        prompt = loaded.template.render(test.vars)
        cls = suite.classification
        expected_label = str(test.expected.get(cls.expected_field)) if cls and cls.expected_field in test.expected else None
        outcome = TestOutcome(
            test_id=test.id,
            model=spec.key(),
            provider=spec.provider,
            repeat_index=repeat,
            vars=test.vars,
            expected=test.expected,
            rendered_prompt=prompt,
            expected_label=expected_label,
        )
        try:
            response = await generator.generate(
                spec,
                prompt,
                temperature=spec.parameters.get("temperature", temperature),
                max_tokens=spec.parameters.get("max_tokens", suite.parameters.max_tokens),
                input_vars=test.vars,
                sample_index=repeat,
            )
        except ProviderError as exc:
            outcome.error = str(exc)
            return outcome

        parsed_ok, parsed = parse_json_output(response.text)
        outcome.output = response.text
        outcome.latency_ms = response.latency_ms
        outcome.input_tokens = response.input_tokens
        outcome.output_tokens = response.output_tokens
        outcome.cost = response.cost
        outcome.cached = response.cached
        outcome.json_valid = parsed_ok
        if cls and parsed_ok:
            label = extract(parsed, cls.path, default=None)
            outcome.predicted_label = None if label is None else str(label).strip().lower()

        ctx = ScoringContext(
            output=response.text,
            parsed_ok=parsed_ok,
            parsed=parsed,
            test=test,
            rendered_prompt=prompt,
            response=response,
            loaded=loaded,
            generator=generator,
            settings=self.settings,
            extra={"model_key": spec.key()},
        )
        results = [await run_scorer(build_scorer(a), ctx) for a in suite.assertions_for(test)]
        outcome.scorer_results = results
        if results:
            total_w = sum(r.details.get("weight", 1.0) for r in results) or 1.0
            outcome.score = sum(r.score * r.details.get("weight", 1.0) for r in results) / total_w
            outcome.passed = all(r.passed for r in results)
        else:
            outcome.score, outcome.passed = 1.0, True
        return outcome
