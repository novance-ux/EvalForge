"""Experiment 2: compare prompt V1 and V2 on the same model and tests.

Shows which metrics improved, which regressed, and which individual tests
were fixed or broken by the new prompt.

    python experiments/exp2_prompt_comparison.py
    python experiments/exp2_prompt_comparison.py --model ollama/llama3.2
"""

from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go

from common import PROMPTS, SUITE, banner, base_parser, is_simulated, save_charts, save_json, setup, style

from app.regression.engine import compare_runs
from app.runner.runner import RunOptions
from app.services import run_suite


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default="mock/mock-small")
    p.add_argument("--v1", default=str(PROMPTS / "classifier_v1.txt"))
    p.add_argument("--v2", default=str(PROMPTS / "classifier_v2.txt"))
    args = p.parse_args()
    banner([args.model])
    settings, db = setup(args.db)
    opts = RunOptions(use_cache=not args.no_cache)

    v1 = run_suite(SUITE, prompt=args.v1, models=[args.model], options=opts, db=db, settings=settings, auto_compare=False)
    v2 = run_suite(SUITE, prompt=args.v2, models=[args.model], options=opts, db=db, settings=settings, auto_compare=False)
    cmp = compare_runs(v1.run, v2.run)

    print(f"\nV1 (run {v1.run.id}) -> V2 (run {v2.run.id}), model {args.model}, n = {cmp.n_pairs}")
    for m in cmp.metrics:
        pct = f"{m.pct_diff:+.1%}" if m.pct_diff is not None else "n/a"
        print(f"  {m.name:16s} {m.baseline:9.4f} -> {m.current:9.4f} ({pct:>8s})  {m.status.value:12s} {m.note}")
    changes = Counter(t.change for t in cmp.tests)
    print(f"\nPer-test: {dict(changes)}")
    print(f"Improved: {cmp.improved_metrics}\nWorse:    {cmp.regressed_metrics}")
    print("\nTests V2 broke:")
    for t in cmp.tests:
        if t.change == "regressed":
            print(f"  - {t.test_id}: {t.baseline_output!r} -> {t.current_output!r}")

    ms = [m for m in cmp.metrics if m.threshold_kind == "absolute"]
    fig1 = go.Figure([
        go.Bar(name="V1", x=[m.name for m in ms], y=[m.baseline for m in ms], marker_color="#9aa5b1"),
        go.Bar(name="V2", x=[m.name for m in ms], y=[m.current for m in ms], marker_color="#2a6fdb"),
    ])
    style(fig1, "Quality metrics: V1 vs V2").update_layout(barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"))
    fig2 = go.Figure(go.Scatter(
        x=[m.abs_diff for m in ms], y=[m.name for m in ms], mode="markers", marker=dict(size=12, color="#2a6fdb"),
        error_x=dict(type="data", symmetric=False, array=[m.ci["high"] - m.abs_diff for m in ms],
                     arrayminus=[m.abs_diff - m.ci["low"] for m in ms]),
    ))
    fig2.add_vline(x=0, line_dash="dash", line_color="#888")
    style(fig2, "Paired difference V2 - V1 with 95% bootstrap CI", 320).update_layout(xaxis_tickformat="+.0%")
    order = ["regressed", "score_down", "fixed", "score_up", "still_failing", "still_passing"]
    fig3 = go.Figure(go.Bar(x=order, y=[changes.get(o, 0) for o in order],
                            marker_color=["#c0392b", "#e67e22", "#27ae60", "#58d68d", "#7f8c8d", "#bdc3c7"]))
    style(fig3, "Per-test verdict changes", 320)
    labels = list(v1.summary["classification"]["per_class"])
    fig4 = go.Figure([
        go.Bar(name="V1", x=labels, y=[v1.summary["classification"]["per_class"][l]["f1"] for l in labels], marker_color="#9aa5b1"),
        go.Bar(name="V2", x=labels, y=[v2.summary["classification"]["per_class"][l]["f1"] for l in labels], marker_color="#2a6fdb"),
    ])
    style(fig4, "F1 per category").update_layout(barmode="group", yaxis=dict(range=[0, 1.05]))
    rel = [m for m in cmp.metrics if m.threshold_kind == "relative"]
    fig5 = go.Figure(go.Bar(x=[m.name for m in rel], y=[m.pct_diff or 0 for m in rel], marker_color="#e0703c"))
    style(fig5, "Relative change in latency and cost (V2 vs V1)", 300).update_layout(yaxis_tickformat="+.0%")

    path = save_charts("exp2_prompt_comparison", "Experiment 2: prompt V1 vs V2", [fig1, fig2, fig3, fig4, fig5],
                       [f"Model: {args.model}", f"Overall: {cmp.overall.value} (quality {cmp.quality.value}, "
                        f"efficiency {cmp.efficiency.value})"], simulated=is_simulated([args.model]))
    data = {"v1_run": v1.run.id, "v2_run": v2.run.id, "comparison": cmp.to_dict(),
            "v1_summary": v1.summary, "v2_summary": v2.summary, "simulated": is_simulated([args.model])}
    print(f"\nCharts: {path}\nData:   {save_json('exp2_prompt_comparison', data)}")


if __name__ == "__main__":
    main()
