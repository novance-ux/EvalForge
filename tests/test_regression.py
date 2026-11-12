from types import SimpleNamespace

import pytest

from app.models.suite import RegressionThresholds
from app.regression.engine import Status, classify_change, compare_runs
from app.regression.stats import CI


def result(test_id, passed, model="mock/m", latency=1000.0, cost=0.001, json_valid=True, repeat=0):
    return SimpleNamespace(
        test_id=test_id, model=model, passed=passed, score=1.0 if passed else 0.0, json_valid=json_valid,
        latency_ms=latency, cost=cost, predicted_label="x" if passed else "y", expected_label="x",
        output="out", vars={"q": test_id}, expected={"c": "x"}, error=None, scorer_results=[], repeat_index=repeat,
    )


def run(run_id, results, models=("mock/m",), prompt="v1"):
    return SimpleNamespace(id=run_id, results=results, models=list(models), prompt_version=prompt,
                           suite_config={}, suite_name="s")


T = RegressionThresholds(min_samples=20, bootstrap_iterations=1000)


def test_detects_regression():
    base = run(1, [result(f"t{i}", True) for i in range(50)])
    cur = run(2, [result(f"t{i}", i >= 10) for i in range(50)])  # 10 tests now fail
    rep = compare_runs(base, cur, T)
    acc = rep.metric("accuracy")
    assert acc.baseline == 1.0 and acc.current == 0.8
    assert acc.abs_diff == pytest.approx(-0.2)
    assert acc.pct_diff == pytest.approx(-0.2)
    assert acc.status == Status.REGRESSED
    assert acc.p_value < 0.01
    assert rep.overall == Status.REGRESSED and rep.regression_detected
    assert sum(t.change == "regressed" for t in rep.tests) == 10
    assert rep.tests[0].change == "regressed"  # regressions sorted first


def test_detects_improvement():
    base = run(1, [result(f"t{i}", i >= 15) for i in range(50)])
    cur = run(2, [result(f"t{i}", True) for i in range(50)])
    rep = compare_runs(base, cur, T)
    assert rep.metric("accuracy").status == Status.IMPROVED
    assert rep.overall == Status.IMPROVED


def test_no_change_within_threshold():
    base = run(1, [result(f"t{i}", True) for i in range(100)])
    cur = run(2, [result(f"t{i}", i != 0) for i in range(100)])  # -1pp < 2pp threshold
    rep = compare_runs(base, cur, T)
    assert rep.metric("accuracy").status == Status.NO_CHANGE


def test_small_sample_is_inconclusive():
    base = run(1, [result(f"t{i}", True) for i in range(5)])
    cur = run(2, [result(f"t{i}", False) for i in range(5)])
    rep = compare_runs(base, cur, T)
    assert rep.metric("accuracy").status == Status.INCONCLUSIVE
    assert "not enough data" in rep.metric("accuracy").note
    assert not rep.regression_detected


def test_threshold_exceeded_but_noisy_is_inconclusive():
    # 3 regressions and 1 fix out of 25: -8pp exceeds 2pp, but the CI includes 0.
    base = run(1, [result(f"t{i}", i != 3) for i in range(25)])
    cur = run(2, [result(f"t{i}", i not in (0, 1, 2)) for i in range(25)])
    rep = compare_runs(base, cur, T)
    acc = rep.metric("accuracy")
    assert acc.abs_diff < -T.accuracy_drop_threshold
    assert acc.status == Status.INCONCLUSIVE


def test_latency_and_cost_regressions_are_relative():
    base = run(1, [result(f"t{i}", True, latency=1000, cost=0.001) for i in range(30)])
    cur = run(2, [result(f"t{i}", True, latency=1500, cost=0.0012) for i in range(30)])
    rep = compare_runs(base, cur, T)
    assert rep.metric("latency_mean_ms").status == Status.REGRESSED  # +50% > 30%
    assert rep.metric("latency_mean_ms").pct_diff == pytest.approx(0.5)
    assert rep.metric("cost_per_test").status == Status.NO_CHANGE  # +20% < 50%
    assert rep.quality == Status.NO_CHANGE and rep.efficiency == Status.REGRESSED


def test_zero_cost_baseline():
    base = run(1, [result(f"t{i}", True, cost=0.0) for i in range(30)])
    cur = run(2, [result(f"t{i}", True, cost=0.0) for i in range(30)])
    assert compare_runs(base, cur, T).metric("cost_per_test").status == Status.NO_CHANGE


def test_cross_model_pairing_by_test_id():
    base = run(1, [result(f"t{i}", True, model="a/x") for i in range(30)], models=["a/x"])
    cur = run(2, [result(f"t{i}", True, model="b/y") for i in range(30)], models=["b/y"])
    rep = compare_runs(base, cur, T)
    assert rep.n_pairs == 30 and "a/x vs b/y" in rep.pairing


def test_repeats_are_averaged_per_test():
    base = run(1, [result(f"t{i}", True, repeat=r) for i in range(25) for r in range(3)])
    cur = run(2, [result(f"t{i}", r != 0, repeat=r) for i in range(25) for r in range(3)])
    rep = compare_runs(base, cur, T)
    assert rep.n_pairs == 25
    assert rep.metric("accuracy").current == pytest.approx(2 / 3)
    assert rep.metric("accuracy").p_value is None  # McNemar only for binary pairs


def test_unpaired_tests_reported():
    base = run(1, [result(f"t{i}", True) for i in range(22)])
    cur = run(2, [result(f"t{i}", True) for i in range(2, 24)])
    rep = compare_runs(base, cur, T)
    assert rep.n_pairs == 20
    assert len(rep.only_in_baseline) == 2 and len(rep.only_in_current) == 2


@pytest.mark.parametrize(
    "delta, ci, higher, expected",
    [
        (-0.05, CI(-0.05, -0.08, -0.02, 0.95), True, Status.REGRESSED),
        (-0.05, CI(-0.05, -0.10, 0.01, 0.95), True, Status.INCONCLUSIVE),
        (0.05, CI(0.05, 0.02, 0.08, 0.95), True, Status.IMPROVED),
        (0.01, CI(0.01, 0.00, 0.02, 0.95), True, Status.NO_CHANGE),
        (0.40, CI(0.40, 0.30, 0.50, 0.95), False, Status.REGRESSED),  # latency up
        (-0.40, CI(-0.40, -0.5, -0.3, 0.95), False, Status.IMPROVED),  # latency down
    ],
)
def test_classify_change(delta, ci, higher, expected):
    status, _ = classify_change(delta, ci, 0.02 if higher else 0.3, higher, n=100, min_samples=20)
    assert status == expected
