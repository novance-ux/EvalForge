import numpy as np
import pytest

from app.evaluation.metrics import classification_metrics, latency_stats
from app.regression.stats import (
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_bootstrap_relative,
    pct_change,
    wilson_interval,
)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(10, 2, 200)
    ci = bootstrap_ci(x, seed=3)
    assert ci.low < x.mean() < ci.high
    assert ci.high - ci.low < 1.0
    assert bootstrap_ci(x, seed=3) == ci


def test_bootstrap_ci_width_shrinks_with_n():
    rng = np.random.default_rng(2)
    small = bootstrap_ci(rng.integers(0, 2, 20))
    large = bootstrap_ci(rng.integers(0, 2, 2000))
    assert (large.high - large.low) < (small.high - small.low)


def test_bootstrap_edge_cases():
    assert bootstrap_ci([]).estimate == 0
    one = bootstrap_ci([0.7])
    assert one.low == one.high == 0.7


def test_paired_diff_detects_consistent_shift():
    base = np.linspace(0, 1, 50)
    ci = paired_bootstrap_diff(base, base + 0.1)
    assert ci.estimate == pytest.approx(0.1)
    assert ci.excludes_zero()


def test_paired_diff_no_change_includes_zero():
    a = np.array([1, 0] * 30)
    b = a.copy()
    b[0], b[1] = 0, 1  # one regression and one fix: net zero
    ci = paired_bootstrap_diff(a, b)
    assert ci.estimate == 0
    assert not ci.excludes_zero()


def test_paired_diff_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1])


def test_relative_change():
    ci = paired_bootstrap_relative([100] * 30, [150] * 30)
    assert ci.estimate == pytest.approx(0.5)
    assert ci.low == pytest.approx(0.5)


def test_mcnemar():
    base = [True] * 30 + [False] * 10
    cur = [False] * 12 + [True] * 18 + [False] * 10
    r = mcnemar_exact(base, cur)
    assert r["pass_to_fail"] == 12 and r["fail_to_pass"] == 0
    assert r["p_value"] < 0.001
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_wilson_interval():
    lo, hi = wilson_interval(46, 50)
    assert 0.8 < lo < 0.92 < hi < 0.98
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_pct_change():
    assert pct_change(0.94, 0.90) == pytest.approx(-0.04255, abs=1e-4)
    assert pct_change(0, 0) == 0.0
    assert pct_change(0, 1) is None


def test_latency_stats():
    s = latency_stats(list(range(1, 101)))
    assert s["mean"] == 50.5 and s["median"] == 50.5
    assert s["p95"] == pytest.approx(95.05)
    assert s["p99"] == pytest.approx(99.01)
    assert latency_stats([])["p95"] == 0


def test_classification_metrics():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", None]
    m = classification_metrics(pred, gold, ["a", "b"])
    assert m["accuracy"] == 0.5
    assert m["per_class"]["a"]["precision"] == 1.0 and m["per_class"]["a"]["recall"] == 0.5
    assert m["confusion_matrix"]["matrix"]["b"]["<invalid>"] == 1
    assert 0 < m["macro_f1"] < 1
