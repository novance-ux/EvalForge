"""Experiment 3: intentionally degrade the prompt and verify detection.

Part A - real run: baseline = V2 prompt, candidate = classifier_regressed.txt
         (JSON requirement and category list removed). The harness must flag
         a regression.
Part B - sensitivity: starting from the baseline results, flip k passing
         tests to failing (k = 0, 1, 2, ...) and record the verdict. This shows
         the smallest regression the harness can reliably detect with this
         many test cases, and that k = 0 does not raise a false alarm.

    python experiments/exp3_regression_detection.py
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import plotly.graph_objects as go

from common import PROMPTS, SUITE, banner, base_parser, is_simulated, save_charts, save_json, setup, style

from app.regression.engine import Status, compare_runs
from app.runner.runner import RunOptions
from app.services import run_suite
from app.storage.repository import Repository


def degraded_copy(run, k: int, rng: random.Random):
    """A fake run where k randomly chosen passing tests now fail."""
    passing = [r for r in run.results if r.passed]
    flip = {id(r) for r in rng.sample(passing, min(k, len(passing)))}
    results = []
    for r in run.results:
        d = {c: getattr(r, c) for c in ("test_id", "model", "passed", "score", "json_valid", "latency_ms", "cost",
                                        "predicted_label", "expected_label", "output", "vars", "expected", "error",
                                        "repeat_index")}
        if id(r) in flip:
            d.update(passed=False, score=0.0, predicted_label="<flipped>")
        results.append(SimpleNamespace(**d, scorer_results=[]))
    return SimpleNamespace(id=-k, results=results, models=run.models, prompt_version=f"synthetic-k{k}",
                           suite_config=run.suite_config, suite_name=run.suite_name)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default="mock/mock-small")
    p.add_argument("--trials", type=int, default=30, help="random trials per k in part B")
    args = p.parse_args()
    banner([args.model])
    settings, db = setup(args.db)
    opts = RunOptions(use_cache=not args.no_cache)

    # ---------------- Part A
    base = run_suite(SUITE, prompt=str(PROMPTS / "classifier_v2.txt"), models=[args.model], options=opts,
                     db=db, settings=settings, auto_compare=False)
    Repository(db).set_baseline(base.run.id, note="exp3 baseline")
    bad = run_suite(SUITE, prompt=str(PROMPTS / "classifier_regressed.txt"), models=[args.model], options=opts,
                    db=db, settings=settings, compare_to=base.run.id)
    cmp = bad.comparison
    print(f"\nPart A: baseline run {base.run.id} (V2) vs degraded run {bad.run.id}")
    for m in cmp.metrics:
        print(f"  {m.name:16s} {m.baseline:9.4f} -> {m.current:9.4f}  {m.status.value}")
    detected = cmp.regression_detected
    print(f"  => regression detected: {detected}  ({'PASS' if detected else 'FAIL'}: harness "
          f"{'caught' if detected else 'missed'} the degraded prompt)")

    # ---------------- Part B
    rng = random.Random(42)
    n = len(base.run.results)
    ks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]
    rows = []
    for k in ks:
        counts = {s: 0 for s in Status}
        for _ in range(args.trials):
            rep = compare_runs(base.run, degraded_copy(base.run, k, rng))
            counts[rep.metric("accuracy").status] += 1
        rows.append({"k": k, "drop_pp": 100 * k / n, **{s.value: c / args.trials for s, c in counts.items()}})
    print(f"\nPart B: sensitivity (n = {n} tests, {args.trials} random trials per k)")
    print("  k  drop    REGRESSED  INCONCLUSIVE  NO_CHANGE")
    for r in rows:
        print(f"  {r['k']:<2d} {r['drop_pp']:4.1f}pp  {r['REGRESSED']:9.0%}  {r['INCONCLUSIVE']:12.0%}  {r['NO_CHANGE']:9.0%}")

    fig1 = go.Figure([
        go.Bar(name="baseline (V2)", x=[m.name for m in cmp.metrics if m.threshold_kind == "absolute"],
               y=[m.baseline for m in cmp.metrics if m.threshold_kind == "absolute"], marker_color="#9aa5b1"),
        go.Bar(name="degraded prompt", x=[m.name for m in cmp.metrics if m.threshold_kind == "absolute"],
               y=[m.current for m in cmp.metrics if m.threshold_kind == "absolute"], marker_color="#c0392b"),
    ])
    style(fig1, f"Part A: degraded prompt vs baseline (verdict: {cmp.overall.value})").update_layout(
        barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"))
    fig2 = go.Figure()
    for status, color in [("REGRESSED", "#c0392b"), ("INCONCLUSIVE", "#e6a23c"), ("NO_CHANGE", "#9aa5b1")]:
        fig2.add_bar(name=status, x=[f"k={r['k']} ({r['drop_pp']:.1f}pp)" for r in rows], y=[r[status] for r in rows],
                     marker_color=color)
    style(fig2, "Part B: verdict vs size of injected regression").update_layout(barmode="stack", yaxis_tickformat=".0%",
                                                                                 xaxis_title="tests flipped pass -> fail")
    path = save_charts("exp3_regression_detection", "Experiment 3: regression detection", [fig1, fig2], [
        f"Model: {args.model}. Part A detected regression: {detected}.",
        "Part B uses synthetic flips of the baseline results to measure detection sensitivity. "
        "k=0 rows show the false-alarm rate (should be 0% REGRESSED).",
    ], simulated=is_simulated([args.model]))
    data = {"baseline_run": base.run.id, "degraded_run": bad.run.id, "detected": detected,
            "comparison": cmp.to_dict(), "sensitivity": rows, "simulated": is_simulated([args.model])}
    print(f"\nCharts: {path}\nData:   {save_json('exp3_regression_detection', data)}")


if __name__ == "__main__":
    main()
