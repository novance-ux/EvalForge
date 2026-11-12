"""Experiment 4: run the same evaluation several times and measure variability.

For each temperature, the suite is executed N times with the cache disabled.
We report the spread of pass rates and latency across runs, how many tests
flip verdict between runs, and - importantly - how often comparing two runs
of the *same* configuration would falsely report a regression.

    python experiments/exp4_variability.py --runs 5
    python experiments/exp4_variability.py --model ollama/llama3.2 --temperatures 0 0.8
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from common import PROMPTS, SUITE, banner, base_parser, is_simulated, save_charts, save_json, setup, style

from app.regression.engine import Status, compare_runs
from app.runner.runner import RunOptions
from app.services import run_suite


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default="mock/mock-small")
    p.add_argument("--prompt", default=str(PROMPTS / "classifier_v1.txt"))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7])
    args = p.parse_args()
    banner([args.model])
    settings, db = setup(args.db)

    data: dict = {"model": args.model, "prompt": args.prompt, "by_temperature": {}, "simulated": is_simulated([args.model])}
    figs_pass = go.Figure()
    figs_flip = go.Figure()
    for temp in args.temperatures:
        runs = []
        for i in range(args.runs):
            r = run_suite(SUITE, prompt=args.prompt, models=[args.model], db=db, settings=settings, auto_compare=False,
                          options=RunOptions(use_cache=False, temperature=temp, label=f"exp4 t={temp} #{i}"))
            runs.append(r)
        rates = [r.summary["pass_rate"] for r in runs]
        lat = [r.summary["latency"]["mean"] for r in runs]
        per_test = defaultdict(list)
        for r in runs:
            for t in r.run.results:
                per_test[t.test_id].append(t.passed)
        flipping = sorted(tid for tid, v in per_test.items() if len(set(v)) > 1)
        false_alarms = 0
        verdicts = defaultdict(int)
        for a, b in itertools.combinations(runs, 2):
            rep = compare_runs(a.run, b.run)
            verdicts[rep.metric("accuracy").status.value] += 1
            false_alarms += rep.metric("accuracy").status == Status.REGRESSED
        pairs = len(runs) * (len(runs) - 1) // 2
        entry = {
            "run_ids": [r.run.id for r in runs],
            "pass_rates": rates,
            "pass_rate_mean": float(np.mean(rates)),
            "pass_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            "pass_rate_range": [min(rates), max(rates)],
            "latency_mean_per_run": lat,
            "latency_cv": float(np.std(lat) / np.mean(lat)) if np.mean(lat) else 0.0,
            "tests_flipping": flipping,
            "flip_rate": len(flipping) / len(per_test),
            "pairwise_accuracy_verdicts": dict(verdicts),
            "false_regression_rate": false_alarms / pairs if pairs else 0.0,
        }
        data["by_temperature"][str(temp)] = entry
        print(f"\ntemperature={temp}: pass rate {entry['pass_rate_mean']:.1%} +/- {entry['pass_rate_std']:.1%} "
              f"(range {min(rates):.1%}-{max(rates):.1%})")
        print(f"  tests that flip verdict across runs: {len(flipping)}/{len(per_test)} ({entry['flip_rate']:.0%})")
        print(f"  latency coefficient of variation: {entry['latency_cv']:.1%}")
        print(f"  same-config run pairs compared: {pairs}; verdicts {dict(verdicts)}; "
              f"false 'REGRESSED' rate {entry['false_regression_rate']:.0%}")
        figs_pass.add_trace(go.Scatter(x=list(range(1, len(rates) + 1)), y=rates, mode="lines+markers", name=f"T={temp}"))
        freq = sorted((float(np.mean(v)) for v in per_test.values()), reverse=True)
        figs_flip.add_trace(go.Bar(x=list(range(len(freq))), y=freq, name=f"T={temp}", opacity=0.7))

    style(figs_pass, "Pass rate of each repeated run").update_layout(xaxis_title="run #", yaxis=dict(tickformat=".0%"))
    style(figs_flip, "Per-test pass frequency across runs (sorted)").update_layout(
        barmode="overlay", xaxis_title="test (sorted)", yaxis=dict(tickformat=".0%"))
    temps = list(data["by_temperature"])
    fig3 = go.Figure(go.Bar(x=[f"T={t}" for t in temps],
                            y=[data["by_temperature"][t]["false_regression_rate"] for t in temps], marker_color="#c0392b"))
    style(fig3, "False REGRESSED verdicts when comparing identical configurations", 300).update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1]))
    path = save_charts("exp4_variability", "Experiment 4: run-to-run variability", [figs_pass, figs_flip, fig3], [
        f"Model {args.model}, prompt {args.prompt}, {args.runs} runs per temperature, cache disabled.",
        "A good regression detector should report ~0% REGRESSED between runs of the same configuration.",
    ], simulated=data["simulated"])
    print(f"\nCharts: {path}\nData:   {save_json('exp4_variability', data)}")


if __name__ == "__main__":
    main()
