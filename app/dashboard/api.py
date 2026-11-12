"""FastAPI dashboard backend.

Read-only JSON API over the results database plus a single static HTML page.
The CLI never depends on this module.

Run with:  llm-harness serve   (or: uvicorn app.dashboard.api:app --reload)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from app.config.settings import get_settings
from app.regression.engine import compare_runs
from app.runner.cost import PricingTable
from app.services import get_db, labels_for
from app.evaluation.metrics import summarize, summarize_by_model
from app.storage.db import Run
from app.storage.repository import Repository

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="LLM Regression Harness API",
    description="Read-only access to evaluation runs, comparisons and test drill-downs.",
    version="0.1.0",
)


def repo() -> Repository:
    return Repository(get_db())


def pricing() -> PricingTable:
    return PricingTable.from_file(get_settings().pricing_path)


def _get_run(r: Repository, run_id: int) -> Run:
    run = r.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return run


def _regression_status(r: Repository, run: Run) -> str | None:
    """Status of this run vs the suite baseline at query time (None if no baseline)."""
    b = r.get_baseline(run.suite_name)
    if b is None or b.run_id == run.id or run.status != "completed" or not run.results:
        return None
    base = r.get_run(b.run_id)
    try:
        return compare_runs(base, run).overall.value
    except ValueError:
        return None


def _run_row(r: Repository, run: Run, baseline_ids: set[int], with_status: bool = True) -> dict[str, Any]:
    s = summarize(run.results, labels_for(run)) if run.results else None
    return {
        "id": run.id,
        "timestamp": run.timestamp.isoformat(),
        "suite_name": run.suite_name,
        "prompt_version": run.prompt_version,
        "models": run.models,
        "status": run.status,
        "label": run.label,
        "git_commit": run.git_commit,
        "is_baseline": run.id in baseline_ids,
        "tests": s["n"] if s else 0,
        "accuracy": s["pass_rate"] if s else None,
        "json_validity": s["json_validity"] if s else None,
        "cost": s["total_cost"] if s else None,
        "latency_mean_ms": s["latency"]["mean"] if s else None,
        "latency_p95_ms": s["latency"]["p95"] if s else None,
        "regression_status": _regression_status(r, run) if with_status else None,
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    r = repo()
    runs = [run for run in r.list_runs_with_results(limit=500) if run.status == "completed"]
    results = [res for run in runs for res in run.results]
    regressions = sum(1 for run in runs if _regression_status(r, r.get_run(run.id)) == "REGRESSED")
    n = len(results)
    ok = [x.latency_ms for x in results if not x.error]
    return {
        "total_runs": len(runs),
        "total_tests": n,
        "pass_rate": sum(x.passed for x in results) / n if n else None,
        "regressions": regressions,
        "average_cost_per_run": sum(x.cost for x in results) / len(runs) if runs else None,
        "average_latency_ms": sum(ok) / len(ok) if ok else None,
        "suites": sorted({run.suite_name for run in runs}),
        "baselines": {b.suite_name: b.run_id for b in reversed(r.list_baselines())},
    }


@app.get("/api/runs")
def list_runs(suite: str | None = None, limit: int = Query(50, le=500)) -> list[dict[str, Any]]:
    r = repo()
    ids = r.baseline_run_ids()
    return [_run_row(r, r.get_run(run.id), ids) for run in r.list_runs(suite, limit)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: int) -> dict[str, Any]:
    r = repo()
    run = _get_run(r, run_id)
    p = pricing()
    labels = labels_for(run)
    return {
        **_run_row(r, run, r.baseline_run_ids()),
        "parameters": run.parameters,
        "prompt_template": run.prompt_template,
        "prompt_hash": run.prompt_hash,
        "cache": {"hits": run.cache_hits, "misses": run.cache_misses, "cost_saved": run.cost_saved},
        "summary": summarize(run.results, labels, p),
        "by_model": summarize_by_model(run.results, labels, p),
    }


@app.get("/api/runs/{run_id}/tests")
def run_tests(run_id: int, failed_only: bool = False) -> list[dict[str, Any]]:
    run = _get_run(repo(), run_id)
    return [
        {
            "test_id": t.test_id, "model": t.model, "repeat_index": t.repeat_index, "passed": t.passed,
            "score": t.score, "latency_ms": t.latency_ms, "cost": t.cost, "output": t.output, "error": t.error,
            "expected": t.expected, "vars": t.vars,
        }
        for t in run.results if not (failed_only and t.passed)
    ]


@app.get("/api/compare")
def compare(baseline: int, current: int, baseline_model: str | None = None, current_model: str | None = None) -> dict[str, Any]:
    r = repo()
    base, cur = _get_run(r, baseline), _get_run(r, current)
    try:
        return compare_runs(base, cur, baseline_model=baseline_model, current_model=current_model).to_dict()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/compare/{baseline}/{current}/tests/{test_id}")
def drilldown(baseline: int, current: int, test_id: str) -> dict[str, Any]:
    """Input, expected result, both outputs, scores, latency, cost and regression reason for one test."""
    r = repo()
    base, cur = _get_run(r, baseline), _get_run(r, current)

    def pick(run: Run) -> list[dict[str, Any]]:
        return [
            {
                "model": t.model, "repeat_index": t.repeat_index, "output": t.output, "error": t.error,
                "passed": t.passed, "score": t.score, "latency_ms": t.latency_ms, "cost": t.cost,
                "input_tokens": t.input_tokens, "output_tokens": t.output_tokens, "cached": t.cached,
                "rendered_prompt": t.input,
                "scorers": [
                    {"type": s.scorer_type, "score": s.score, "passed": s.passed, "reason": s.reason, "details": s.details}
                    for s in t.scorer_results
                ],
            }
            for t in run.results if t.test_id == test_id
        ]

    b, c = pick(base), pick(cur)
    if not b and not c:
        raise HTTPException(404, f"Test {test_id} not found in either run")
    any_row = next(t for t in (base.results + cur.results) if t.test_id == test_id)
    regression_reason = None
    if b and c and b[0]["passed"] and not c[0]["passed"]:
        regression_reason = "; ".join(f"{s['type']}: {s['reason']}" for s in c[0]["scorers"] if not s["passed"]) or c[0]["error"]
    return {
        "test_id": test_id, "input": any_row.vars, "expected": any_row.expected,
        "baseline": b, "current": c, "regression_reason": regression_reason,
    }


@app.get("/api/baselines")
def baselines() -> list[dict[str, Any]]:
    return [
        {"suite_name": b.suite_name, "run_id": b.run_id, "created_at": b.created_at.isoformat(), "note": b.note}
        for b in repo().list_baselines()
    ]


@app.get("/reports/{run_id}", response_class=HTMLResponse, include_in_schema=False)
def html_report(run_id: int, baseline: int | None = None) -> FileResponse:
    """Render the full HTML report on demand."""
    from app.reporting.html_report import render_report

    r = repo()
    run = _get_run(r, run_id)
    if baseline is None:
        b = r.get_baseline(run.suite_name)
        baseline = b.run_id if b and b.run_id != run_id else None
    base = r.get_run(baseline) if baseline else None
    comparison = None
    if base is not None:
        try:
            comparison = compare_runs(base, run)
        except ValueError:
            comparison = None
    p = pricing()
    out = get_settings().reports_dir / f"run_{run_id}.html"
    path = render_report(run, summarize(run.results, labels_for(run), p), p, comparison, base, out)
    return FileResponse(path)
