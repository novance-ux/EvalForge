"""Aggregate metrics over test results.

Functions accept any objects exposing the attributes of ``TestOutcome`` /
``storage.db.TestResult`` (duck typing), so they work both on fresh runner
output and on rows loaded from the database.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from app.runner.cost import PricingTable

INVALID_LABEL = "<invalid>"


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {k: 0.0 for k in ("mean", "median", "p50", "p95", "p99", "min", "max", "std")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def classification_metrics(pred: Sequence[str | None], gold: Sequence[str], labels: Sequence[str] | None = None) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix.

    Predictions outside the label set (invented categories, invalid JSON)
    count as errors and appear in the confusion matrix as their own column.
    """
    pred = [p if p else INVALID_LABEL for p in pred]
    labels = list(labels) if labels else sorted(set(gold))
    extra = sorted(set(pred) - set(labels))
    cols = labels + extra
    matrix = {g: {c: 0 for c in cols} for g in labels}
    for p, g in zip(pred, gold):
        if g in matrix:
            matrix[g][p] += 1

    per_class = {}
    for lab in labels:
        tp = sum(1 for p, g in zip(pred, gold) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(pred, gold) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(pred, gold) if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    present = [lab for lab in labels if per_class[lab]["support"] > 0]
    macro_f1 = float(np.mean([per_class[lab]["f1"] for lab in present])) if present else 0.0
    accuracy = sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "columns": cols, "matrix": matrix},
    }


def _judge_scorers(results: Iterable[Any]) -> list[Any]:
    return [s for r in results for s in getattr(r, "scorer_results", []) if s.scorer_type == "llm_judge"]


def judge_summary(results: Sequence[Any]) -> dict[str, Any] | None:
    judged = [s for s in _judge_scorers(results) if s.details.get("raw_scores")]
    if not judged:
        return None
    means = [s.details["mean_score"] for s in judged]
    stds = [s.details.get("std_score", 0.0) for s in judged]
    agree = [s.details.get("agreement_rate", 1.0) for s in judged]
    return {
        "judged_tests": len(judged),
        "judge_models": sorted({s.details.get("judge_model") for s in judged}),
        "runs_per_test": max(s.details.get("runs", 1) for s in judged),
        "mean_score": float(np.mean(means)),
        "mean_within_test_std": float(np.mean(stds)),
        "mean_agreement_rate": float(np.mean(agree)),
        "same_model_warnings": sum(1 for s in judged if s.details.get("same_model_as_evaluated")),
        "judge_cost": float(sum(s.details.get("judge_cost", 0.0) for s in judged)),
    }


def summarize(results: Sequence[Any], labels: Sequence[str] | None = None, pricing: PricingTable | None = None) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)
    ok_latencies = [r.latency_ms for r in results if not r.error]
    total_cost = float(sum(r.cost for r in results))
    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.output_tokens for r in results)

    summary: dict[str, Any] = {
        "n": n,
        "unique_tests": len({r.test_id for r in results}),
        "passed": passed,
        "failed": n - passed,
        "errors": errors,
        "pass_rate": passed / n if n else 0.0,
        "mean_score": float(np.mean([r.score for r in results])) if n else 0.0,
        "json_validity": sum(1 for r in results if r.json_valid) / n if n else 0.0,
        "latency": latency_stats(ok_latencies),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_cost": total_cost,
        "cost_per_test": total_cost / n if n else 0.0,
        "cost_per_success": total_cost / passed if passed else None,
        "cached": sum(1 for r in results if r.cached),
    }
    if pricing is not None:
        inp = out = 0.0
        for r in results:
            b = pricing.breakdown(r.provider, r.model.split("/", 1)[-1], r.input_tokens, r.output_tokens)
            inp, out = inp + b["input_cost"], out + b["output_cost"]
        summary["input_cost"], summary["output_cost"] = inp, out

    labelled = [r for r in results if r.expected_label is not None]
    if labelled:
        cm = classification_metrics(
            [r.predicted_label for r in labelled], [r.expected_label for r in labelled], labels
        )
        summary["classification"] = cm
        correct = sum(1 for r in labelled if r.predicted_label == r.expected_label)
        summary["cost_per_correct"] = total_cost / correct if correct else None

    judge = judge_summary(results)
    if judge:
        summary["judge"] = judge
    return summary


def group_by(results: Sequence[Any], key: Callable[[Any], str]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return dict(groups)


def summarize_by_model(results: Sequence[Any], labels: Sequence[str] | None = None, pricing: PricingTable | None = None) -> dict[str, dict[str, Any]]:
    return {m: summarize(rs, labels, pricing) for m, rs in group_by(results, lambda r: r.model).items()}


def repeat_variability(results: Sequence[Any]) -> dict[str, Any]:
    """Run-to-run variability when a suite is executed with repeats > 1."""
    by_repeat = group_by(results, lambda r: f"{r.model}#{r.repeat_index}")
    pass_rates = [sum(r.passed for r in rs) / len(rs) for rs in by_repeat.values()]
    by_test = group_by(results, lambda r: f"{r.model}::{r.test_id}")
    flips = sum(1 for rs in by_test.values() if len({r.passed for r in rs}) > 1)
    outputs_differ = sum(1 for rs in by_test.values() if len({r.output for r in rs}) > 1)
    return {
        "repeats": len({r.repeat_index for r in results}),
        "pass_rate_per_repeat": pass_rates,
        "pass_rate_mean": float(np.mean(pass_rates)) if pass_rates else 0.0,
        "pass_rate_std": float(np.std(pass_rates, ddof=1)) if len(pass_rates) > 1 else 0.0,
        "tests_with_flipping_verdict": flips,
        "tests_with_differing_output": outputs_differ,
        "tests": len(by_test),
    }
