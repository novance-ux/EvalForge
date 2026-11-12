"""Terminal reports rendered with Rich."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.regression.engine import ComparisonReport, Status

console = Console()

STATUS_STYLE = {
    Status.IMPROVED: "bold green",
    Status.REGRESSED: "bold red",
    Status.NO_CHANGE: "dim",
    Status.INCONCLUSIVE: "yellow",
}


def money(x: float | None) -> str:
    if x is None:
        return "n/a"
    if x == 0:
        return "$0.00"
    if x < 0.01:
        return f"${x:.6f}"
    return f"${x:.4f}" if x < 1 else f"${x:,.2f}"


def secs(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def print_run_summary(run: Any, summary: dict[str, Any], per_model: dict[str, dict[str, Any]],
                      cache: dict[str, Any] | None = None, regression: ComparisonReport | None = None,
                      unpriced: set[str] | None = None) -> None:
    console.rule("[bold]LLM Regression Harness")
    lines = [
        f"Suite:   [bold]{run.suite_name}[/]",
        f"Run ID:  {run.id}    Prompt: {run.prompt_version}    Models: {', '.join(run.models)}",
        "",
        f"Tests:   {summary['n']}" + (f"  ({summary['unique_tests']} cases x repeats/models)" if summary['n'] != summary['unique_tests'] else ""),
        f"Passed:  [green]{summary['passed']}[/]",
        f"Failed:  [red]{summary['failed']}[/]" + (f"  (errors: {summary['errors']})" if summary["errors"] else ""),
        "",
        f"Accuracy:       {summary['pass_rate']:.1%}",
        f"JSON Validity:  {summary['json_validity']:.1%}",
    ]
    if "classification" in summary:
        lines.append(f"Macro-F1:       {summary['classification']['macro_f1']:.3f}")
    lat = summary["latency"]
    lines += [
        "",
        f"Average Latency: {secs(lat['mean'])}",
        f"P95 Latency:     {secs(lat['p95'])}",
        "",
        f"Total Cost:      {money(summary['total_cost'])}   (per test {money(summary['cost_per_test'])},"
        f" per success {money(summary['cost_per_success'])})",
    ]
    if summary.get("judge"):
        j = summary["judge"]
        lines.append(
            f"Judge:           mean {j['mean_score']:.2f}, agreement {j['mean_agreement_rate']:.0%}, "
            f"eval cost {money(j['judge_cost'])} ({', '.join(j['judge_models'])})"
        )
    if cache:
        lines.append(
            f"Cache:           {cache['hits']} hits / {cache['misses']} misses, est. saved {money(cache['cost_saved'])}"
        )
    if unpriced:
        lines.append(f"[yellow]Unpriced models (cost shown as $0): {', '.join(sorted(unpriced))}[/]")
    lines.append("")
    if regression is None:
        lines.append("Regression Status: [dim]NO BASELINE[/] (set one with `llm-harness baseline RUN_ID`)")
    else:
        verdict = "[bold red]FAIL[/]" if regression.regression_detected else "[bold green]PASS[/]"
        lines.append(
            f"Regression Status: {verdict}  (vs baseline run {regression.baseline_run_id}: "
            f"[{STATUS_STYLE[regression.overall]}]{regression.overall.value}[/]; "
            f"quality {regression.quality.value}, efficiency {regression.efficiency.value})"
        )
    console.print(Panel("\n".join(lines), expand=False))

    if len(per_model) > 1:
        print_model_table(per_model)


def print_model_table(per_model: dict[str, dict[str, Any]]) -> None:
    t = Table(title="By model")
    for col in ["Model", "Accuracy", "F1", "JSON", "Mean lat", "P95 lat", "Cost", "Cost/correct"]:
        t.add_column(col, justify="right" if col != "Model" else "left")
    for model, s in per_model.items():
        f1 = s.get("classification", {}).get("macro_f1")
        t.add_row(
            model, f"{s['pass_rate']:.1%}", f"{f1:.3f}" if f1 is not None else "-", f"{s['json_validity']:.1%}",
            secs(s["latency"]["mean"]), secs(s["latency"]["p95"]), money(s["total_cost"]),
            money(s.get("cost_per_correct", s["cost_per_success"])),
        )
    console.print(t)


def _fmt_value(name: str, v: float) -> str:
    if "latency" in name:
        return secs(v)
    if "cost" in name:
        return money(v)
    return f"{v:.1%}"


def print_comparison(report: ComparisonReport, max_tests: int = 15) -> None:
    console.rule(f"[bold]Comparison: baseline run {report.baseline_run_id} -> current run {report.current_run_id}")
    console.print(
        f"Suite [bold]{report.suite_name}[/]  |  prompt {report.baseline_prompt} -> {report.current_prompt}  |  "
        f"models {', '.join(report.baseline_models)} -> {', '.join(report.current_models)}"
    )
    console.print(f"Paired on {report.pairing}; n = {report.n_pairs} paired tests")

    t = Table()
    for col in ["Metric", "Baseline", "Current", "Change", "% Change", "95% CI of change", "Threshold", "Status"]:
        t.add_column(col, justify="left" if col in ("Metric", "Status") else "right")
    for m in report.metrics:
        if m.threshold_kind == "relative":
            change = _fmt_value(m.name, m.abs_diff)
            ci = f"[{m.ci['low']:+.1%}, {m.ci['high']:+.1%}] rel" if m.ci else "-"
            thr = f"+{m.threshold:.0%} rel"
        else:
            change = f"{m.abs_diff * 100:+.1f} pp"
            ci = f"[{m.ci['low'] * 100:+.1f}, {m.ci['high'] * 100:+.1f}] pp" if m.ci else "-"
            thr = f"{m.threshold * 100:.1f} pp"
        pct = f"{m.pct_diff:+.1%}" if m.pct_diff is not None else "n/a"
        status = f"[{STATUS_STYLE[m.status]}]{m.status.value}[/]"
        t.add_row(m.name, _fmt_value(m.name, m.baseline), _fmt_value(m.name, m.current), change, pct, ci, thr, status)
    console.print(t)
    for m in report.metrics:
        if m.status in (Status.INCONCLUSIVE, Status.REGRESSED, Status.IMPROVED):
            p = f" (McNemar exact p = {m.p_value:.4f})" if m.p_value is not None else ""
            console.print(f"  [dim]{m.name}: {m.note}{p}[/]")

    counts: dict[str, int] = {}
    for tc in report.tests:
        counts[tc.change] = counts.get(tc.change, 0) + 1
    console.print(
        "Per-test changes: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    changed = [tc for tc in report.tests if tc.change in ("regressed", "fixed")]
    if changed:
        tt = Table(title=f"Changed tests (showing {min(len(changed), max_tests)} of {len(changed)})")
        for col in ["Test", "Change", "Baseline output", "Current output", "Why (current)"]:
            tt.add_column(col, overflow="fold")
        for tc in changed[:max_tests]:
            style = "red" if tc.change == "regressed" else "green"
            tt.add_row(tc.test_id, f"[{style}]{tc.change}[/]", tc.baseline_output[:60], tc.current_output[:60],
                       "; ".join(tc.reasons)[:90] or "-")
        console.print(tt)
    if report.only_in_baseline or report.only_in_current:
        console.print(f"[yellow]Unpaired: {len(report.only_in_baseline)} only in baseline, "
                      f"{len(report.only_in_current)} only in current[/]")

    verdict = {
        Status.REGRESSED: "[bold red]REGRESSION DETECTED[/]",
        Status.IMPROVED: "[bold green]IMPROVED (no regressions)[/]",
        Status.NO_CHANGE: "[bold]NO SIGNIFICANT CHANGE[/]",
        Status.INCONCLUSIVE: "[bold yellow]INCONCLUSIVE[/] (differences could be noise or n is too small)",
    }[report.overall]
    console.print(f"\nOverall: {verdict}")
    console.print(
        f"  Quality (accuracy, score, JSON): [{STATUS_STYLE[report.quality]}]{report.quality.value}[/]   "
        f"Efficiency (latency, cost): [{STATUS_STYLE[report.efficiency]}]{report.efficiency.value}[/]"
    )
    if report.improved_metrics:
        console.print(f"  Better: {', '.join(report.improved_metrics)}")
    if report.regressed_metrics:
        console.print(f"  Worse:  {', '.join(report.regressed_metrics)}")
