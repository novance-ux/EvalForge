"""Statistical helpers for comparing two evaluation runs.

The same test cases are evaluated by both runs, so comparisons are **paired**:
we resample *test cases* (not individual results) and look at the per-test
difference. This removes between-test variance and gives much tighter
intervals than treating the two runs as independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class CI:
    estimate: float
    low: float
    high: float
    level: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "level": self.level}


def bootstrap_ci(values, level: float = 0.95, iterations: int = 2000, seed: int = 0) -> CI:
    """Percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(0.0, 0.0, 0.0, level)
    if arr.size == 1:
        return CI(float(arr[0]), float(arr[0]), float(arr[0]), level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(iterations, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return CI(float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), level)


def paired_bootstrap_diff(baseline, current, level: float = 0.95, iterations: int = 2000, seed: int = 0) -> CI:
    """CI for mean(current - baseline) over paired observations."""
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    if b.shape != c.shape:
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci(c - b, level, iterations, seed)


def paired_bootstrap_relative(baseline, current, level: float = 0.95, iterations: int = 2000, seed: int = 0) -> CI:
    """CI for the relative change mean(current)/mean(baseline) - 1 (paired resampling)."""
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    if b.size == 0 or b.mean() == 0:
        return CI(0.0, 0.0, 0.0, level)
    est = float(c.mean() / b.mean() - 1)
    if b.size == 1:
        return CI(est, est, est, level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.size, size=(iterations, b.size))
    bm = b[idx].mean(axis=1)
    cm = c[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(bm > 0, cm / bm - 1, 0.0)
    alpha = (1 - level) / 2
    return CI(est, float(np.quantile(rel, alpha)), float(np.quantile(rel, 1 - alpha)), level)


def mcnemar_exact(baseline_pass, current_pass) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes.

    Only *discordant* pairs carry information: b = pass->fail, c = fail->pass.
    Under H0 (no change) b ~ Binomial(b + c, 0.5).
    """
    bp = np.asarray(baseline_pass, dtype=bool)
    cp = np.asarray(current_pass, dtype=bool)
    b = int(np.sum(bp & ~cp))  # regressed
    c = int(np.sum(~bp & cp))  # fixed
    n = b + c
    p = 1.0 if n == 0 else float(sps.binomtest(b, n, 0.5).pvalue)
    return {"pass_to_fail": b, "fail_to_pass": c, "discordant": n, "p_value": p}


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a single proportion (e.g. pass rate)."""
    if n == 0:
        return (0.0, 0.0)
    z = sps.norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def pct_change(baseline: float, current: float) -> float | None:
    if baseline == 0:
        return None if current != 0 else 0.0
    return (current - baseline) / abs(baseline)
