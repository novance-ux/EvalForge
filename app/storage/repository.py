"""Data-access functions. The rest of the app never writes SQL directly."""

from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.results import CacheStats, TestOutcome
from app.storage.db import Baseline, Database, HumanLabel, Run, ScorerResult, TestResult
from app.utils.misc import utcnow


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ---------------------------------------------------------------- runs
    def create_run(self, **fields: Any) -> Run:
        with self.db.session() as s:
            run = Run(**fields)
            s.add(run)
            s.commit()
            return run

    def save_outcomes(self, run_id: int, outcomes: Iterable[TestOutcome]) -> None:
        with self.db.session() as s:
            for o in outcomes:
                tr = TestResult(
                    run_id=run_id,
                    test_id=o.test_id,
                    model=o.model,
                    provider=o.provider,
                    repeat_index=o.repeat_index,
                    input=o.rendered_prompt,
                    vars=o.vars,
                    expected=o.expected,
                    output=o.output,
                    error=o.error,
                    score=o.score,
                    passed=o.passed,
                    latency_ms=o.latency_ms,
                    input_tokens=o.input_tokens,
                    output_tokens=o.output_tokens,
                    cost=o.cost,
                    cached=o.cached,
                    json_valid=o.json_valid,
                    predicted_label=o.predicted_label,
                    expected_label=o.expected_label,
                )
                tr.scorer_results = [
                    ScorerResult(
                        scorer_type=r.scorer_type,
                        name=r.details.get("name"),
                        score=r.score,
                        passed=r.passed,
                        reason=r.reason,
                        weight=r.details.get("weight", 1.0),
                        details=r.details,
                    )
                    for r in o.scorer_results
                ]
                s.add(tr)
            s.commit()

    def finish_run(self, run_id: int, status: str, cache: CacheStats | None = None, error: str | None = None) -> None:
        with self.db.session() as s:
            run = s.get(Run, run_id)
            run.status = status
            run.finished_at = utcnow()
            run.error = error
            if cache:
                run.cache_hits, run.cache_misses, run.cost_saved = cache.hits, cache.misses, cache.cost_saved
            s.commit()

    def get_run(self, run_id: int) -> Run | None:
        with self.db.session() as s:
            stmt = (
                select(Run)
                .where(Run.id == run_id)
                .options(selectinload(Run.results).selectinload(TestResult.scorer_results))
            )
            return s.scalars(stmt).first()

    def list_runs(self, suite_name: str | None = None, limit: int = 100) -> list[Run]:
        with self.db.session() as s:
            stmt = select(Run).order_by(Run.id.desc()).limit(limit)
            if suite_name:
                stmt = stmt.where(Run.suite_name == suite_name)
            return list(s.scalars(stmt))

    def list_runs_with_results(self, limit: int = 100) -> list[Run]:
        with self.db.session() as s:
            stmt = (
                select(Run)
                .order_by(Run.id.desc())
                .limit(limit)
                .options(selectinload(Run.results))
            )
            return list(s.scalars(stmt))

    def count_test_results(self) -> int:
        with self.db.session() as s:
            return s.scalar(select(func.count(TestResult.id))) or 0

    # ----------------------------------------------------------- baselines
    def set_baseline(self, run_id: int, note: str | None = None) -> Baseline:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Run {run_id} not found")
        if run.status != "completed":
            raise ValueError(f"Run {run_id} has status '{run.status}'; only completed runs can be baselines")
        with self.db.session() as s:
            b = Baseline(suite_name=run.suite_name, run_id=run_id, note=note)
            s.add(b)
            s.commit()
            return b

    def get_baseline(self, suite_name: str) -> Baseline | None:
        with self.db.session() as s:
            stmt = (
                select(Baseline)
                .where(Baseline.suite_name == suite_name)
                .order_by(Baseline.created_at.desc(), Baseline.id.desc())
            )
            return s.scalars(stmt).first()

    def list_baselines(self) -> list[Baseline]:
        with self.db.session() as s:
            return list(s.scalars(select(Baseline).order_by(Baseline.id.desc())))

    def baseline_run_ids(self) -> set[int]:
        with self.db.session() as s:
            return set(s.scalars(select(Baseline.run_id)))

    # -------------------------------------------------------- human labels
    def import_human_labels(
        self, suite_name: str, rows: Iterable[tuple[str, float]], annotator: str = "default", source: str | None = None
    ) -> int:
        count = 0
        with self.db.session() as s:
            for test_id, score in rows:
                existing = s.scalars(
                    select(HumanLabel).where(
                        HumanLabel.suite_name == suite_name,
                        HumanLabel.test_id == test_id,
                        HumanLabel.annotator == annotator,
                    )
                ).first()
                if existing:
                    existing.human_score, existing.source = score, source
                else:
                    s.add(HumanLabel(suite_name=suite_name, test_id=test_id, human_score=score, annotator=annotator, source=source))
                count += 1
            s.commit()
        return count

    def get_human_labels(self, suite_name: str, annotator: str | None = None) -> dict[str, float]:
        with self.db.session() as s:
            stmt = select(HumanLabel).where(HumanLabel.suite_name == suite_name)
            if annotator:
                stmt = stmt.where(HumanLabel.annotator == annotator)
            return {h.test_id: h.human_score for h in s.scalars(stmt)}
