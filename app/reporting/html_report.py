"""Standalone HTML report for a run (optionally compared with a baseline)."""

from __future__ import annotations

import difflib
from collections import Counter
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.evaluation.metrics import summarize_by_model
from app.regression.engine import ComparisonReport
from app.runner.cost import PricingTable
from app.storage.db import Run

TEMPLATES = Path(__file__).parent / "templates"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

SCORER_DOCS = {
    "exact_match": "Normalised output (trimmed, whitespace-collapsed, case-insensitive by default) must equal the expected value.",
    "equals": "Value selected by a JSONPath (e.g. $.category) must equal the expected value.",
    "contains": "Expected text must appear somewhere in the (normalised) output.",
    "regex": "Output (or JSONPath selection) must match a regular expression.",
    "json_schema": "Output must be valid JSON that satisfies the JSON Schema (Draft 2020-12).",
    "numeric_tolerance": "Number in the output must be within +/- tolerance of the expected value; partial credit decays to 0 at 2x tolerance.",
    "latency_under_ms": "Response latency must be below the limit (cached responses keep their original latency).",
    "semantic_similarity": "Cosine similarity between embeddings of output and reference must reach the threshold.",
    "llm_judge": "A separate judge LLM scores the output against a rubric; normalised to [0,1]. Automated opinion, not ground truth.",
}

COLORS = ["#2a6fdb", "#e0703c", "#2e9e6a", "#9b59b6", "#c0392b", "#7f8c8d"]


def _fig_html(fig: go.Figure, height: int = 340) -> str:
    fig.update_layout(
        height=height, margin=dict(l=40, r=20, t=40, b=40), template="plotly_white",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def build_charts(run: Run, per_model: dict[str, dict], comparison: ComparisonReport | None, summary: dict) -> dict[str, str]:
    charts: dict[str, str] = {}
    models = list(per_model)
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    charts["accuracy"] = _fig_html(go.Figure(
        [go.Bar(x=models, y=[per_model[m]["pass_rate"] for m in models], marker_color=colors,
                text=[f"{per_model[m]['pass_rate']:.1%}" for m in models], textposition="outside")],
        layout=dict(title="Accuracy (pass rate) by model", yaxis=dict(range=[0, 1.1], tickformat=".0%")),
    ))

    fig = go.Figure()
    for i, m in enumerate(models):
        fig.add_histogram(x=[r.score for r in run.results if r.model == m], name=m, xbins=dict(start=0, end=1.0001, size=0.1),
                          marker_color=colors[i], opacity=0.75)
    fig.update_layout(title="Score distribution", barmode="overlay", xaxis_title="score", yaxis_title="tests")
    charts["scores"] = _fig_html(fig)

    fig = go.Figure()
    for i, m in enumerate(models):
        fig.add_box(y=[r.latency_ms for r in run.results if r.model == m and not r.error], name=m,
                    marker_color=colors[i], boxpoints="outliers")
    fig.update_layout(title="Latency by model (ms)", yaxis_title="ms", showlegend=False)
    charts["latency"] = _fig_html(fig)

    charts["cost"] = _fig_html(go.Figure(
        [go.Bar(x=models, y=[per_model[m]["total_cost"] for m in models], marker_color=colors, name="total"),
         ],
        layout=dict(title="Total cost by model (USD, from pricing file)", yaxis_title="USD"),
    ))

    cls = summary.get("classification")
    if cls:
        cm = cls["confusion_matrix"]
        z = [[cm["matrix"][g][c] for c in cm["columns"]] for g in cm["labels"]]
        fig = go.Figure(go.Heatmap(z=z, x=cm["columns"], y=cm["labels"], colorscale="Blues",
                                   text=z, texttemplate="%{text}", showscale=False))
        fig.update_layout(title="Confusion matrix (rows = expected, cols = predicted)", yaxis_autorange="reversed")
        charts["confusion"] = _fig_html(fig, 420)

    if comparison:
        ms = [m for m in comparison.metrics if m.threshold_kind == "absolute"]
        fig = go.Figure([
            go.Bar(name=f"baseline (run {comparison.baseline_run_id})", x=[m.name for m in ms], y=[m.baseline for m in ms], marker_color="#9aa5b1"),
            go.Bar(name=f"current (run {comparison.current_run_id})", x=[m.name for m in ms], y=[m.current for m in ms], marker_color="#2a6fdb",
                   error_y=dict(type="data", symmetric=False,
                                array=[max(0.0, m.ci["high"] - m.abs_diff) if m.ci else 0 for m in ms],
                                arrayminus=[max(0.0, m.abs_diff - m.ci["low"]) if m.ci else 0 for m in ms])),
        ])
        fig.update_layout(title="Baseline vs current (error bars: CI of the paired difference)", barmode="group",
                          yaxis=dict(tickformat=".0%", range=[0, 1.15]))
        charts["comparison"] = _fig_html(fig)
        counts = Counter(t.change for t in comparison.tests)
        order = ["regressed", "score_down", "fixed", "score_up", "still_failing", "still_passing"]
        fig = go.Figure(go.Bar(x=order, y=[counts.get(o, 0) for o in order],
                               marker_color=["#c0392b", "#e67e22", "#27ae60", "#58d68d", "#7f8c8d", "#bdc3c7"]))
        fig.update_layout(title="Per-test changes vs baseline")
        charts["changes"] = _fig_html(fig, 300)
    return charts


def prompt_diff(baseline: Run | None, current: Run) -> str | None:
    if baseline is None or not baseline.prompt_template or not current.prompt_template:
        return None
    if baseline.prompt_template == current.prompt_template:
        return None
    return "\n".join(difflib.unified_diff(
        baseline.prompt_template.splitlines(), current.prompt_template.splitlines(),
        fromfile=f"{baseline.prompt_version} (run {baseline.id})",
        tofile=f"{current.prompt_version} (run {current.id})", lineterm="",
    ))


def render_report(
    run: Run,
    summary: dict[str, Any],
    pricing: PricingTable | None = None,
    comparison: ComparisonReport | None = None,
    baseline: Run | None = None,
    out_path: str | Path | None = None,
    offline: bool = False,
) -> Path:
    labels = ((run.suite_config or {}).get("classification") or {}).get("labels")
    per_model = summarize_by_model(run.results, labels, pricing)
    failed = [r for r in run.results if not r.passed]
    scorer_types = sorted({s.scorer_type for r in run.results for s in r.scorer_results})

    # Side-by-side outputs: one row per test, one column per model (first repeat only).
    models = list(per_model)
    grid: dict[str, dict[str, Any]] = {}
    for r in run.results:
        if r.repeat_index == 0:
            grid.setdefault(r.test_id, {"vars": r.vars, "expected": r.expected})[r.model] = r

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = lambda v: "n/a" if v is None else f"{v:.1%}"
    from app.reporting.cli_report import money

    env.filters["money"] = money
    env.filters["secs"] = lambda v: f"{v / 1000:.2f}s"
    if offline:
        from plotly.offline import get_plotlyjs

        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'

    html = env.get_template("report.html.j2").render(
        run=run,
        summary=summary,
        per_model=per_model,
        comparison=comparison,
        baseline=baseline,
        prompt_diff=prompt_diff(baseline, run),
        failed=failed,
        grid=grid,
        models=models,
        scorer_docs={t: SCORER_DOCS.get(t, "Custom scorer.") for t in scorer_types},
        charts=build_charts(run, per_model, comparison, summary),
        plotly_tag=plotly_tag,
        unpriced=[m for m in run.models if pricing and pricing.lookup(*m.split("/", 1)) is None],
    )
    out = Path(out_path) if out_path else Path("reports") / f"run_{run.id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
