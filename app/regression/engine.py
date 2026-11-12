"""Baseline comparison and regression detection.

For every metric we report the baseline value, current value, absolute and
percentage difference, a paired bootstrap confidence interval, and one of
four verdicts:

* ``IMPROVED``      change beyond the threshold in the good direction AND the CI excludes 0
* ``REGRESSED``     change beyond the threshold in the bad direction AND the CI excludes 0
* ``NO_CHANGE``     change within the configured threshold
* ``INCONCLUSIVE``  change beyond the threshold but the CI includes 0,
                    or too few paired tests to say anything (n < min_samples)

A threshold being exceeded is necessary but not sufficient for a verdict:
we never claim a regression that could plausibly be noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from app.models.suite import RegressionThresholds
from app.regression.stats import CI, mcnemar_exact, paired_bootstrap_diff, paired_bootstrap_relative, pct_change
from app.storage.db import Run


QUALITY_METRICS = {"accuracy", "label_accuracy", "mean_score", "json_validity"}
EFFICIENCY_METRICS = {"latency_mean_ms", "cost_per_test"}


class Status(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    NO_CHANGE = "NO_CHANGE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class MetricComparison:
    name: str
    baseline: float
    current: float
    abs_diff: float
    pct_diff: float | None
    ci: dict[str, float] | None
    threshold: float
    threshold_kind: str  # "absolute" or "relative"
    higher_is_better: bool
    status: Status
    note: str = ""
    p_value: float | None = None


@dataclass
class TestChange:
    test_id: str
    model_baseline: str
    model_current: str
    change: str  # regressed | fixed | still_failing | still_passing | score_down | score_up
    baseline_passed: float
    current_passed: float
    baseline_score: float
    current_score: float
    baseline_output: str
    current_output: str
    baseline_latency_ms: float
    current_latency_ms: float
    baseline_cost: float
    current_cost: float
    input: dict[str, Any]
    expected: dict[str, Any]
    reasons: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    baseline_run_id: int
    current_run_id: int
    suite_name: str
    baseline_prompt: str
    current_prompt: str
    baseline_models: list[str]
    current_models: list[str]
    pairing: str
    n_pairs: int
    metrics: list[MetricComparison]
    tests: list[TestChange]
    only_in_baseline: list[str]
    only_in_current: list[str]
    thresholds: dict[str, Any]
    overall: Status
    quality: Status = Status.NO_CHANGE
    efficiency: Status = Status.NO_CHANGE
    mcnemar: dict[str, float] | None = None

    @property
    def regressed_metrics(self) -> list[str]:
        return [m.name for m in self.metrics if m.status == Status.REGRESSED]

    @property
    def improved_metrics(self) -> list[str]:
        return [m.name for m in self.metrics if m.status == Status.IMPROVED]

    @property
    def regression_detected(self) -> bool:
        return self.overall == Status.REGRESSED

    def metric(self, name: str) -> MetricComparison | None:
        return next((m for m in self.metrics if m.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall"] = self.overall.value
        d["quality"] = self.quality.value
        d["efficiency"] = self.efficiency.value
        d["regressed_metrics"] = self.regressed_metrics
        d["improved_metrics"] = self.improved_metrics
        for m in d["metrics"]:
            m["status"] = m["status"].value if isinstance(m["status"], Status) else m["status"]
        return d


def classify_change(
    delta: float,
    ci: CI | None,
    threshold: float,
    higher_is_better: bool,
    n: int,
    min_samples: int,
) -> tuple[Status, str]:
    harm = -delta if higher_is_better else delta  # > 0 means worse
    if n < min_samples:
        return Status.INCONCLUSIVE, f"only {n} paired tests (< min_samples={min_samples}); not enough data"
    if abs(harm) <= threshold:
        return Status.NO_CHANGE, "change within threshold"
    worse = harm > 0
    if ci is None:
        return (Status.REGRESSED if worse else Status.IMPROVED), "no confidence interval available"
    ci_confirms = (ci.high < 0 or ci.low > 0) and ((ci.high < 0) == (delta < 0))
    if not ci_confirms:
        return Status.INCONCLUSIVE, f"exceeds threshold but {int(ci.level * 100)}% CI includes 0"
    return (Status.REGRESSED, "exceeds threshold; CI excludes 0") if worse else (
        Status.IMPROVED,
        "exceeds threshold; CI excludes 0",
    )


def combine_statuses(statuses: list[Status], empty: bool = False) -> Status:
    """Any regression dominates; then improvement; then inconclusive; else no change."""
    found = set(statuses)
    if Status.REGRESSED in found:
        return Status.REGRESSED
    if Status.IMPROVED in found:
        return Status.IMPROVED
    if Status.INCONCLUSIVE in found or empty:
        return Status.INCONCLUSIVE
    return Status.NO_CHANGE


def _aggregate(results: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Average repeats per (model, test) so each pair is one observation."""
    groups: dict[tuple[str, str], list[Any]] = {}
    for r in results:
        groups.setdefault((r.model, r.test_id), []).append(r)
    out = {}
    for (model, test_id), rs in groups.items():
        failing = [
            f"{s.scorer_type}: {s.reason}" for s in rs[0].scorer_results if not s.passed
        ] if hasattr(rs[0], "scorer_results") else []
        if rs[0].error:
            failing.insert(0, f"error: {rs[0].error}")
        out[f"{model}::{test_id}"] = {
            "model": model,
            "test_id": test_id,
            "passed": float(np.mean([r.passed for r in rs])),
            "score": float(np.mean([r.score for r in rs])),
            "json_valid": float(np.mean([r.json_valid for r in rs])),
            "latency_ms": float(np.mean([r.latency_ms for r in rs])),
            "cost": float(np.mean([r.cost for r in rs])),
            "correct": (
                float(np.mean([r.predicted_label == r.expected_label for r in rs]))
                if rs[0].expected_label is not None else None
            ),
            "output": rs[0].output,
            "input": rs[0].vars,
            "expected": rs[0].expected,
            "reasons": failing,
            "binary": len(rs) == 1,
        }
    return out


def _pair(base: dict, cur: dict, baseline_model: str | None, current_model: str | None):
    b_models = {v["model"] for v in base.values()}
    c_models = {v["model"] for v in cur.values()}
    if baseline_model or current_model:
        bm = baseline_model or next(iter(b_models))
        cm = current_model or next(iter(c_models))
        mode = f"test_id ({bm} vs {cm})"
        b = {v["test_id"]: v for v in base.values() if v["model"] == bm}
        c = {v["test_id"]: v for v in cur.values() if v["model"] == cm}
    elif b_models & c_models:
        mode = "model + test_id"
        common = b_models & c_models
        b = {k: v for k, v in base.items() if v["model"] in common}
        c = {k: v for k, v in cur.items() if v["model"] in common}
    elif len(b_models) == 1 and len(c_models) == 1:
        mode = f"test_id ({next(iter(b_models))} vs {next(iter(c_models))})"
        b = {v["test_id"]: v for v in base.values()}
        c = {v["test_id"]: v for v in cur.values()}
    else:
        raise ValueError(
            "Runs share no model and contain several models each; pass baseline_model/current_model"
        )
    keys = sorted(set(b) & set(c))
    return mode, keys, b, c, sorted(set(b) - set(c)), sorted(set(c) - set(b))


def compare_runs(
    baseline: Run,
    current: Run,
    thresholds: RegressionThresholds | None = None,
    baseline_model: str | None = None,
    current_model: str | None = None,
) -> ComparisonReport:
    if thresholds is None:
        cfg = (current.suite_config or {}).get("regression") or {}
        thresholds = RegressionThresholds.model_validate(cfg)
    t = thresholds
    base = _aggregate(baseline.results)
    cur = _aggregate(current.results)
    mode, keys, b, c, only_b, only_c = _pair(base, cur, baseline_model, current_model)
    n = len(keys)
    kw = {"level": t.confidence_level, "iterations": t.bootstrap_iterations}

    def arr(src: dict, field_: str) -> np.ndarray:
        return np.array([src[k][field_] for k in keys], dtype=float)

    metrics: list[MetricComparison] = []

    def absolute_metric(name: str, field_: str, threshold: float, higher: bool = True):
        if n == 0:
            return None
        bv, cv = arr(b, field_), arr(c, field_)
        ci = paired_bootstrap_diff(bv, cv, **kw)
        delta = float(cv.mean() - bv.mean())
        status, note = classify_change(delta, ci, threshold, higher, n, t.min_samples)
        m = MetricComparison(
            name, float(bv.mean()), float(cv.mean()), delta, pct_change(bv.mean(), cv.mean()),
            ci.as_dict(), threshold, "absolute", higher, status, note,
        )
        metrics.append(m)
        return m

    def relative_metric(name: str, field_: str, threshold: float):
        if n == 0:
            return
        bv, cv = arr(b, field_), arr(c, field_)
        if bv.mean() == 0:
            status = Status.NO_CHANGE if cv.mean() == 0 else Status.INCONCLUSIVE
            note = "baseline is 0; relative change undefined" if cv.mean() else "both zero"
            metrics.append(MetricComparison(name, 0.0, float(cv.mean()), float(cv.mean()), None, None,
                                            threshold, "relative", False, status, note))
            return
        ci = paired_bootstrap_relative(bv, cv, **kw)
        status, note = classify_change(ci.estimate, ci, threshold, False, n, t.min_samples)
        metrics.append(MetricComparison(
            name, float(bv.mean()), float(cv.mean()), float(cv.mean() - bv.mean()), ci.estimate,
            ci.as_dict(), threshold, "relative", False, status, note,
        ))

    acc = absolute_metric("accuracy", "passed", t.accuracy_drop_threshold)
    mcn = None
    if acc and all(b[k]["binary"] and c[k]["binary"] for k in keys):
        mcn = mcnemar_exact(arr(b, "passed") > 0.5, arr(c, "passed") > 0.5)
        acc.p_value = mcn["p_value"]
    if n and all(b[k]["correct"] is not None and c[k]["correct"] is not None for k in keys):
        absolute_metric("label_accuracy", "correct", t.accuracy_drop_threshold)
    absolute_metric("mean_score", "score", t.score_drop_threshold)
    absolute_metric("json_validity", "json_valid", t.json_validity_drop_threshold)
    relative_metric("latency_mean_ms", "latency_ms", t.latency_increase_threshold)
    relative_metric("cost_per_test", "cost", t.cost_increase_threshold)

    tests = []
    for k in keys:
        bb, cc = b[k], c[k]
        if bb["passed"] > cc["passed"]:
            change = "regressed"
        elif bb["passed"] < cc["passed"]:
            change = "fixed"
        elif cc["score"] < bb["score"] - 1e-9:
            change = "score_down"
        elif cc["score"] > bb["score"] + 1e-9:
            change = "score_up"
        else:
            change = "still_passing" if cc["passed"] >= 0.5 else "still_failing"
        tests.append(TestChange(
            test_id=bb["test_id"], model_baseline=bb["model"], model_current=cc["model"], change=change,
            baseline_passed=bb["passed"], current_passed=cc["passed"],
            baseline_score=bb["score"], current_score=cc["score"],
            baseline_output=bb["output"], current_output=cc["output"],
            baseline_latency_ms=bb["latency_ms"], current_latency_ms=cc["latency_ms"],
            baseline_cost=bb["cost"], current_cost=cc["cost"],
            input=bb["input"], expected=bb["expected"], reasons=cc["reasons"],
        ))
    order = {"regressed": 0, "score_down": 1, "fixed": 2, "score_up": 3, "still_failing": 4, "still_passing": 5}
    tests.sort(key=lambda x: (order[x.change], x.test_id))

    overall = combine_statuses([m.status for m in metrics], empty=n == 0)
    quality = combine_statuses([m.status for m in metrics if m.name in QUALITY_METRICS], empty=n == 0)
    efficiency = combine_statuses([m.status for m in metrics if m.name in EFFICIENCY_METRICS], empty=n == 0)

    return ComparisonReport(
        baseline_run_id=baseline.id,
        current_run_id=current.id,
        suite_name=current.suite_name,
        baseline_prompt=baseline.prompt_version,
        current_prompt=current.prompt_version,
        baseline_models=list(baseline.models or []),
        current_models=list(current.models or []),
        pairing=mode,
        n_pairs=n,
        metrics=metrics,
        tests=tests,
        only_in_baseline=only_b,
        only_in_current=only_c,
        thresholds=t.model_dump(),
        overall=overall,
        quality=quality,
        efficiency=efficiency,
        mcnemar=mcn,
    )
