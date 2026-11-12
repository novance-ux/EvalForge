# Regression detection

## Why paired comparison

A baseline run and a current run evaluate **the same test cases**. Test cases differ a lot in difficulty. Comparing
the two runs as independent samples would mix that difficulty variance into the uncertainty and make real changes hard
to see. The engine therefore works **per test case**:

```text
d_i = current_i − baseline_i        (i = 1..n paired tests)
```

and asks whether the mean of `d` is meaningfully different from 0.

If a run used `--repeat N`, repeats are averaged per (model, test) first, so each test case counts once.

**Pairing rules.** Tests are paired on `(model, test_id)` when the runs share models. If each run has a single,
different model (a model comparison), they are paired on `test_id`. For multi-model runs you can choose models with
`--baseline-model` and `--current-model`.

## Statistics used

| Quantity | Method | File |
|---|---|---|
| CI of a mean difference (accuracy, score, JSON validity) | percentile **paired bootstrap**: resample test cases with replacement, 2000 iterations, seed 0 | `regression/stats.py::paired_bootstrap_diff` |
| CI of a relative change (latency, cost) | paired bootstrap of `mean(cur)/mean(base) − 1` | `paired_bootstrap_relative` |
| p-value for pass/fail changes | exact **McNemar** test on discordant pairs (pass→fail vs fail→pass) | `mcnemar_exact` |
| CI of a single pass rate | Wilson score interval | `wilson_interval` |

The bootstrap is used because it makes no normality assumption, works for bounded and binary data, and is easy to
explain: "if we had drawn a different set of test cases like these, how much could the difference move?"

## Verdicts

```text
harm = −delta   (quality metrics: higher is better)
harm = +delta   (latency, cost: lower is better)

if n < min_samples:                         INCONCLUSIVE  ("not enough data")
elif |harm| ≤ threshold:                    NO_CHANGE
elif CI excludes 0 in the same direction:   REGRESSED (harm > 0) / IMPROVED (harm < 0)
else:                                       INCONCLUSIVE  ("exceeds threshold but CI includes 0")
```

Both conditions are needed for a verdict:

- **Practical significance.** The change is larger than a threshold the team cares about.
- **Statistical evidence.** The change is unlikely to be noise from this particular set of test cases.

A tiny but "significant" change (for example 0.5 pp over 10,000 tests) is reported as NO_CHANGE if it is under the
threshold. A large change on 8 tests is INCONCLUSIVE. The harness never claims significance when n is too small.

### Overall status

- `overall`: REGRESSED if any metric regressed; otherwise IMPROVED if any improved; otherwise INCONCLUSIVE if any was inconclusive; otherwise NO_CHANGE.
- `quality`: the same rule over accuracy, label accuracy, mean score and JSON validity.
- `efficiency`: the same rule over latency and cost.

The CLI prints **Regression Status: PASS/FAIL** (FAIL only for REGRESSED). `--fail-on-regression` exits with code 2.

## Configuration

```yaml
regression:
  accuracy_drop_threshold: 0.02        # absolute, applies to accuracy and label_accuracy
  score_drop_threshold: 0.02           # absolute
  json_validity_drop_threshold: 0.02   # absolute
  latency_increase_threshold: 0.30     # relative
  cost_increase_threshold: 0.50        # relative
  min_samples: 20
  confidence_level: 0.95
  bootstrap_iterations: 2000
```

The thresholds from the **current** run's suite are used.

## Per-test changes

Each paired test is labelled `regressed` (pass→fail), `fixed` (fail→pass), `score_down`, `score_up`,
`still_failing` or `still_passing`. Regressions are listed first, with the failing scorers' reasons. This answers
"what exactly got worse?".

## Worked example (demo data, mock provider)

| Metric | v1 | v2 | Change | 95% CI | Verdict |
|---|---|---|---|---|---|
| accuracy | 56.7% | 88.1% | +31.3 pp | [+19.4, +44.8] pp | IMPROVED |
| json_validity | 85.1% | 100% | +14.9 pp | [+7.5, +23.9] pp | IMPROVED |
| latency (mean) | 0.45 s | 0.40 s | −10% | [−20%, +1%] | NO_CHANGE |
| cost per test | $0.000019 | $0.000065 | +239% | [+227%, +251%] | REGRESSED |

Reading: v2 is clearly more accurate (23 tests fixed, 2 broken), but its longer prompt more than triples the token
cost. Whether that is acceptable is a product decision. The harness makes the trade-off visible instead of hiding it.

## Detection sensitivity

Experiment 3 (part B) flips k passing tests to failing in a copy of the baseline and records the verdict. With
n = 67 and pure regressions (no compensating fixes), k ≤ 1 gives NO_CHANGE (under the 2 pp threshold), k = 2–3 gives
INCONCLUSIVE, and k ≥ 4 (about 6 pp) gives REGRESSED every time. With no injected change (k = 0) there are no false
alarms. Real changes usually mix fixes and breaks, which widens the CI, so treat this as a best case. The main way to
detect smaller effects is **more test cases**.

## Caveats

- The CI describes uncertainty over *this suite's* test cases. It says nothing about production traffic unless the
  suite is a representative sample of it.
- Several metrics are tested at once without multiple-comparison correction.
- At temperature > 0 each run is itself a random sample. Use `--repeat` or compare several runs (Experiment 4).
