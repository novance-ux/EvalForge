"""Compare LLM-judge scores with human labels.

Automated judge scores are NOT ground truth. This module quantifies how far
they can be trusted on a particular task by comparing them with a
human-labelled calibration set:

* exact agreement and within-one agreement
* mean absolute difference and mean bias (judge - human)
* Cohen's kappa (unweighted) and quadratic-weighted kappa for ordinal scales
* confusion matrix (human rows x judge columns)
* pass/fail agreement and kappa after binarising at a threshold
* Spearman rank correlation
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats as sps

from app.storage.db import Run


def load_human_labels_csv(path: str | Path) -> list[tuple[str, float]]:
    """Read ``test_id,human_score`` rows (header required, extra columns ignored)."""
    rows: list[tuple[str, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"test_id", "human_score"} <= set(reader.fieldnames):
            raise ValueError("CSV must have columns: test_id,human_score")
        for i, row in enumerate(reader, start=2):
            if not row["test_id"] or row["human_score"] in (None, ""):
                continue
            try:
                rows.append((row["test_id"].strip(), float(row["human_score"])))
            except ValueError as exc:
                raise ValueError(f"line {i}: human_score must be numeric") from exc
    return rows


def cohen_kappa(a: Sequence[Any], b: Sequence[Any], labels: Sequence[Any] | None = None, weights: str | None = None) -> float:
    """Cohen's kappa; ``weights`` may be None, 'linear' or 'quadratic' (ordinal labels)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty sequences")
    labels = list(labels) if labels is not None else sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weights is None:
        w = 1 - np.eye(k)
    else:
        i, j = np.meshgrid(range(k), range(k), indexing="ij")
        d = np.abs(i - j) / max(k - 1, 1)
        w = d if weights == "linear" else d**2
    denom = (w * exp).sum()
    if denom == 0:
        return 1.0  # perfect agreement on a single category
    return float(1 - (w * obs).sum() / denom)


def interpret_kappa(k: float) -> str:
    # Landis & Koch (1977) - a conventional, somewhat arbitrary scale.
    for limit, text in [(0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= limit:
            return text
    return "almost perfect"


def judge_scores_from_run(run: Run, model: str | None = None) -> dict[str, float]:
    """Mean raw judge score per test_id (averaged over judge runs and repeats)."""
    acc: dict[str, list[float]] = {}
    for r in run.results:
        if model and r.model != model:
            continue
        for s in r.scorer_results:
            if s.scorer_type == "llm_judge" and s.details.get("raw_scores"):
                acc.setdefault(r.test_id, []).append(float(s.details["mean_score"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def calibrate(
    judge: dict[str, float],
    human: dict[str, float],
    scale: tuple[int, int] = (1, 5),
    pass_threshold: float = 4,
) -> dict[str, Any]:
    ids = sorted(set(judge) & set(human))
    if not ids:
        raise ValueError("no overlapping test ids between judge scores and human labels")
    j = np.array([judge[i] for i in ids])
    h = np.array([human[i] for i in ids])
    jr = np.clip(np.rint(j), *scale).astype(int)  # categorical view of the judge
    hr = np.clip(np.rint(h), *scale).astype(int)
    labels = list(range(scale[0], scale[1] + 1))

    matrix = {int(a): {int(b): 0 for b in labels} for a in labels}
    for a, b in zip(hr, jr):
        matrix[int(a)][int(b)] += 1

    jp, hp = jr >= pass_threshold, hr >= pass_threshold
    rho = sps.spearmanr(h, j).statistic if len(ids) > 2 and h.std() > 0 and j.std() > 0 else float("nan")
    result = {
        "n": len(ids),
        "missing_human_labels": sorted(set(judge) - set(human)),
        "missing_judge_scores": sorted(set(human) - set(judge)),
        "exact_agreement": float(np.mean(jr == hr)),
        "within_one_agreement": float(np.mean(np.abs(jr - hr) <= 1)),
        "mean_absolute_difference": float(np.mean(np.abs(j - h))),
        "mean_bias_judge_minus_human": float(np.mean(j - h)),
        "cohen_kappa": cohen_kappa(list(hr), list(jr), labels),
        "weighted_kappa_quadratic": cohen_kappa(list(hr), list(jr), labels, weights="quadratic"),
        "spearman_rho": float(rho),
        "pass_fail_agreement": float(np.mean(jp == hp)),
        "pass_fail_kappa": cohen_kappa(list(hp), list(jp), [False, True]),
        "confusion_matrix": {"labels": labels, "rows_are": "human", "cols_are": "judge", "matrix": matrix},
        "pairs": [{"test_id": i, "judge": float(judge[i]), "human": float(human[i])} for i in ids],
    }
    result["kappa_interpretation"] = interpret_kappa(result["cohen_kappa"])
    result["small_sample_warning"] = len(ids) < 30
    return result
