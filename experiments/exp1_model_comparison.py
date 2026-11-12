"""Experiment 1: compare two models on the same suite and prompt.

Measures accuracy, macro-F1, JSON validity, latency, cost and cost per
correct prediction, plus a paired statistical comparison.

    python experiments/exp1_model_comparison.py
    python experiments/exp1_model_comparison.py --models openai/gpt-4o-mini ollama/llama3.2
"""

from __future__ import annotations

import plotly.graph_objects as go

from common import (PROMPTS, SUITE, banner, base_parser, fmt_pct, is_simulated, print_table, save_charts,
                    save_json, setup, style)

from app.regression.engine import compare_runs
from app.runner.runner import RunOptions
from app.services import run_suite


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--models", nargs=2, default=["mock/mock-small", "mock/mock-large"])
    p.add_argument("--prompt", default=str(PROMPTS / "classifier_v2.txt"))
    args = p.parse_args()
    banner(args.models)
    settings, db = setup(args.db)

    runs = {}
    for m in args.models:
        print(f"Running {m} ...")
        runs[m] = run_suite(SUITE, prompt=args.prompt, models=[m], options=RunOptions(use_cache=not args.no_cache),
                            db=db, settings=settings, auto_compare=False)

    rows, data = [], {}
    for m, r in runs.items():
        s = r.summary
        cls = s.get("classification", {})
        data[m] = {
            "run_id": r.run.id, "accuracy": s["pass_rate"], "label_accuracy": cls.get("accuracy"),
            "macro_f1": cls.get("macro_f1"), "json_validity": s["json_validity"],
            "latency_mean_ms": s["latency"]["mean"], "latency_p95_ms": s["latency"]["p95"],
            "total_cost": s["total_cost"], "cost_per_correct": s.get("cost_per_correct"),
            "per_class_f1": {k: v["f1"] for k, v in cls.get("per_class", {}).items()},
        }
        d = data[m]
        rows.append([m, fmt_pct(d["accuracy"]), f"{d['macro_f1']:.3f}", fmt_pct(d["json_validity"]),
                     f"{d['latency_mean_ms']:.0f} ms", f"{d['latency_p95_ms']:.0f} ms", f"${d['total_cost']:.5f}",
                     f"${d['cost_per_correct']:.6f}" if d["cost_per_correct"] else "n/a"])
    print()
    print_table(rows, ["Model", "Accuracy", "F1", "JSON valid", "Mean lat", "P95 lat", "Cost", "Cost/correct"])

    a, b = args.models
    cmp = compare_runs(runs[a].run, runs[b].run)
    print(f"\nPaired comparison {a} -> {b} (n = {cmp.n_pairs}):")
    for mc in cmp.metrics:
        ci = f"[{mc.ci['low']:+.3f}, {mc.ci['high']:+.3f}]" if mc.ci else "-"
        print(f"  {mc.name:16s} {mc.baseline:10.4f} -> {mc.current:10.4f}  CI {ci:22s} {mc.status.value}")
    data["comparison"] = cmp.to_dict()
    data["simulated"] = is_simulated(args.models)

    m = args.models
    colors = ["#2a6fdb", "#e0703c"]
    fig1 = go.Figure([go.Bar(name=x, x=["Accuracy", "Macro-F1", "JSON validity"],
                             y=[data[x]["accuracy"], data[x]["macro_f1"], data[x]["json_validity"]], marker_color=c)
                      for x, c in zip(m, colors)])
    style(fig1, "Quality by model").update_layout(barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"))
    fig2 = go.Figure([go.Box(y=[t.latency_ms for t in runs[x].run.results], name=x, marker_color=c) for x, c in zip(m, colors)])
    style(fig2, "Latency distribution (ms)")
    fig3 = go.Figure([go.Bar(x=m, y=[data[x]["cost_per_correct"] or 0 for x in m], marker_color=colors)])
    style(fig3, "Cost per correct prediction (USD)")
    labels = list(data[m[0]]["per_class_f1"])
    fig4 = go.Figure([go.Bar(name=x, x=labels, y=[data[x]["per_class_f1"][l] for l in labels], marker_color=c)
                      for x, c in zip(m, colors)])
    style(fig4, "F1 per category").update_layout(barmode="group", yaxis=dict(range=[0, 1.05]))
    path = save_charts("exp1_model_comparison", "Experiment 1: model comparison", [fig1, fig2, fig3, fig4],
                       [f"Prompt: {args.prompt}", f"Overall paired verdict {a} -> {b}: {cmp.overall.value}"],
                       simulated=data["simulated"])
    print(f"\nCharts: {path}\nData:   {save_json('exp1_model_comparison', data)}")


if __name__ == "__main__":
    main()
