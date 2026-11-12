"""High-level operations shared by the CLI, the dashboard and experiment scripts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config.settings import Settings, get_settings
from app.config.suite_loader import load_suite
from app.evaluation.metrics import summarize, summarize_by_model
from app.models.results import TestOutcome
from app.models.suite import RegressionThresholds
from app.providers.registry import ProviderRegistry
from app.regression.engine import ComparisonReport, compare_runs
from app.runner.cost import PricingTable
from app.runner.runner import EvaluationRunner, RunOptions, RunResult
from app.storage.db import Database, Run
from app.storage.repository import Repository


def get_db(settings: Settings | None = None) -> Database:
    return Database((settings or get_settings()).db_path)


def labels_for(run: Run) -> list[str] | None:
    cls = (run.suite_config or {}).get("classification") or {}
    return cls.get("labels") or None


@dataclass
class SuiteRun:
    run: Run
    result: RunResult
    summary: dict[str, Any]
    per_model: dict[str, dict[str, Any]]
    comparison: ComparisonReport | None


def run_suite(
    suite_path: str | Path,
    prompt: str | None = None,
    models: list[str] | None = None,
    options: RunOptions | None = None,
    compare_to: int | None = None,
    auto_compare: bool = True,
    db: Database | None = None,
    settings: Settings | None = None,
    registry: ProviderRegistry | None = None,
    progress: Callable[[TestOutcome], None] | None = None,
) -> SuiteRun:
    settings = settings or get_settings()
    db = db or get_db(settings)
    repo = Repository(db)
    loaded = load_suite(suite_path, prompt_override=prompt, models_override=models)
    pricing = PricingTable.from_file(settings.pricing_path)
    runner = EvaluationRunner(db, settings, registry=registry, pricing=pricing)
    result = asyncio.run(runner.run(loaded, options, progress))
    run = repo.get_run(result.run_id)

    labels = loaded.suite.classification.labels if loaded.suite.classification else None
    summary = summarize(run.results, labels, pricing)
    per_model = summarize_by_model(run.results, labels, pricing)

    comparison = None
    baseline_id = compare_to
    if baseline_id is None and auto_compare:
        b = repo.get_baseline(loaded.suite.name)
        baseline_id = b.run_id if b else None
    if baseline_id is not None and baseline_id != run.id:
        base_run = repo.get_run(baseline_id)
        if base_run is not None:
            try:
                comparison = compare_runs(base_run, run, loaded.suite.regression)
            except ValueError:
                comparison = None
    return SuiteRun(run, result, summary, per_model, comparison)


def compare(
    baseline_id: int,
    current_id: int,
    db: Database | None = None,
    thresholds: RegressionThresholds | None = None,
    baseline_model: str | None = None,
    current_model: str | None = None,
) -> ComparisonReport:
    repo = Repository(db or get_db())
    base, cur = repo.get_run(baseline_id), repo.get_run(current_id)
    if base is None:
        raise KeyError(f"Run {baseline_id} not found")
    if cur is None:
        raise KeyError(f"Run {current_id} not found")
    return compare_runs(base, cur, thresholds, baseline_model, current_model)


def run_summary(run: Run, settings: Settings | None = None) -> dict[str, Any]:
    pricing = PricingTable.from_file((settings or get_settings()).pricing_path)
    return summarize(run.results, labels_for(run), pricing)
