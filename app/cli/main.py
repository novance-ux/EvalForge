"""Command-line interface: ``llm-harness <command>``."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from app.config.settings import get_settings
from app.config.suite_loader import SuiteError, load_suite
from app.providers.registry import available_providers
from app.reporting.cli_report import console, money, print_comparison, print_run_summary
from app.runner.runner import RunOptions
from app.storage.repository import Repository

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LLM Regression Testing & Evaluation Harness: detect whether a prompt/model change improved or regressed your LLM app.",
)


def _repo() -> Repository:
    from app.services import get_db

    return Repository(get_db())


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[bold red]Error:[/] {msg}")
    raise typer.Exit(code)


def _get_run(repo: Repository, run_id: int):
    run = repo.get_run(run_id)
    if run is None:
        _fail(f"run {run_id} not found (see `llm-harness runs`)")
    return run


# --------------------------------------------------------------------------- init / validate
@app.command()
def init(force: bool = typer.Option(False, help="Overwrite an existing .env")) -> None:
    """Create folders, a .env from .env.example, and the results database."""
    for d in ["suites", "prompts", "schemas", "reports", "data", "config"]:
        Path(d).mkdir(exist_ok=True)
    env, example = Path(".env"), Path(".env.example")
    if example.exists() and (force or not env.exists()):
        shutil.copy(example, env)
        console.print("Created [bold].env[/] from .env.example; add your API keys there (optional for the mock provider).")
    from app.services import get_db

    get_db()
    s = get_settings()
    console.print(f"Database ready at [bold]{s.db_path}[/]. Providers: {', '.join(available_providers())}")
    console.print("Next: [bold]llm-harness validate suites/support_classifier.yaml[/]")


@app.command()
def validate(
    suite: Path = typer.Argument(..., help="Path to a suite YAML file"),
    prompt: Optional[str] = typer.Option(None, help="Override the suite's prompt file"),
) -> None:
    """Validate a suite file without calling any model."""
    try:
        loaded = load_suite(suite, prompt_override=prompt)
    except SuiteError as exc:
        console.print(f"[bold red]Suite is invalid:[/] {exc.path}")
        for p in exc.problems:
            console.print(f"  [red]-[/] {p}")
        raise typer.Exit(1)
    s = loaded.suite
    n_assert = sum(len(s.assertions_for(t)) for t in s.tests)
    console.print(f"[green]OK[/] {s.name}: {len(s.tests)} tests, {n_assert} assertions, "
                  f"prompt {loaded.prompt_version} (vars: {', '.join(sorted(loaded.template.variables))}), "
                  f"models: {', '.join(m.key() for m in s.models)}")


# --------------------------------------------------------------------------- run
@app.command()
def run(
    suite: Path = typer.Argument(..., help="Path to a suite YAML file"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Override the prompt file (e.g. prompts/classifier_v2.txt)"),
    model: Optional[list[str]] = typer.Option(None, "--model", "-m", help="Override models, e.g. -m openai/gpt-4o-mini -m ollama/llama3.2"),
    repeat: int = typer.Option(1, "--repeat", "-r", min=1, help="Run every test N times (variability)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the response cache"),
    concurrency: Optional[int] = typer.Option(None, help="Max concurrent requests"),
    temperature: Optional[float] = typer.Option(None, help="Override temperature"),
    label: Optional[str] = typer.Option(None, help="Free-text label for this run"),
    baseline: Optional[int] = typer.Option(None, "--baseline", "-b", help="Compare against this run id instead of the suite baseline"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Make this run the suite baseline"),
    report: bool = typer.Option(True, "--report/--no-report", help="Write an HTML report"),
    fail_on_regression: bool = typer.Option(False, help="Exit with code 2 when a regression is detected (for CI)"),
) -> None:
    """Run a suite, store results, and compare with the baseline."""
    from app.services import run_suite

    try:
        loaded = load_suite(suite, prompt_override=prompt, models_override=model)
    except SuiteError as exc:
        _fail(str(exc))
    total = len(loaded.suite.tests) * len(loaded.suite.models) * repeat
    opts = RunOptions(repeats=repeat, use_cache=not no_cache, max_concurrency=concurrency,
                      temperature=temperature, label=label)
    with Progress(TextColumn("[bold]Evaluating"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console, transient=True) as progress:
        task = progress.add_task("run", total=total)
        result = run_suite(suite, prompt, model, opts, compare_to=baseline,
                           progress=lambda _o: progress.advance(task))
    cache = result.result.cache.model_dump() if not no_cache else None
    print_run_summary(result.run, result.summary, result.per_model, cache, result.comparison,
                      result.result.unpriced_models)
    if result.comparison:
        print_comparison(result.comparison, max_tests=10)
    if set_baseline:
        _repo().set_baseline(result.run.id, note=label)
        console.print(f"Run {result.run.id} is now the baseline for [bold]{result.run.suite_name}[/].")
    if report:
        from app.reporting.html_report import render_report
        from app.runner.cost import PricingTable

        repo = _repo()
        base = repo.get_run(result.comparison.baseline_run_id) if result.comparison else None
        path = render_report(result.run, result.summary, PricingTable.from_file(get_settings().pricing_path),
                             result.comparison, base, get_settings().reports_dir / f"run_{result.run.id}.html")
        console.print(f"HTML report: [bold]{path}[/]")
    if fail_on_regression and result.comparison and result.comparison.regression_detected:
        raise typer.Exit(2)


# --------------------------------------------------------------------------- compare / report / baseline
@app.command()
def compare(
    baseline_run: int = typer.Argument(..., help="Baseline run id"),
    current_run: int = typer.Argument(..., help="Current run id"),
    baseline_model: Optional[str] = typer.Option(None, help="Model to take from the baseline run (multi-model runs)"),
    current_model: Optional[str] = typer.Option(None, help="Model to take from the current run"),
    html: bool = typer.Option(False, "--html", help="Also write an HTML comparison report"),
    as_json: bool = typer.Option(False, "--json", help="Print the comparison as JSON"),
    fail_on_regression: bool = typer.Option(False, help="Exit with code 2 on regression"),
) -> None:
    """Compare two runs (paired, with bootstrap confidence intervals)."""
    from app.services import compare as do_compare

    try:
        report = do_compare(baseline_run, current_run, baseline_model=baseline_model, current_model=current_model)
    except (KeyError, ValueError) as exc:
        _fail(str(exc))
    if as_json:
        console.print_json(json.dumps(report.to_dict(), default=str))
    else:
        print_comparison(report)
    if html:
        from app.reporting.html_report import render_report
        from app.runner.cost import PricingTable
        from app.services import run_summary

        repo = _repo()
        cur, base = repo.get_run(current_run), repo.get_run(baseline_run)
        path = render_report(cur, run_summary(cur), PricingTable.from_file(get_settings().pricing_path), report, base,
                             get_settings().reports_dir / f"compare_{baseline_run}_vs_{current_run}.html")
        console.print(f"HTML report: [bold]{path}[/]")
    if fail_on_regression and report.regression_detected:
        raise typer.Exit(2)


@app.command()
def report(
    run_id: int = typer.Argument(..., help="Run id"),
    baseline: Optional[int] = typer.Option(None, "--baseline", "-b", help="Compare against this run (default: suite baseline)"),
    out: Optional[Path] = typer.Option(None, help="Output HTML path"),
    offline: bool = typer.Option(False, help="Embed plotly.js so the report works without internet"),
) -> None:
    """Generate an HTML report for a run."""
    from app.regression.engine import compare_runs
    from app.reporting.html_report import render_report
    from app.runner.cost import PricingTable
    from app.services import run_summary

    repo = _repo()
    run_ = _get_run(repo, run_id)
    base_id = baseline
    if base_id is None:
        b = repo.get_baseline(run_.suite_name)
        base_id = b.run_id if b and b.run_id != run_id else None
    base = repo.get_run(base_id) if base_id else None
    comparison = None
    if base is not None:
        try:
            comparison = compare_runs(base, run_)
        except ValueError as exc:
            console.print(f"[yellow]Skipping comparison: {exc}[/]")
    path = render_report(run_, run_summary(run_), PricingTable.from_file(get_settings().pricing_path), comparison,
                         base, out or get_settings().reports_dir / f"run_{run_id}.html", offline=offline)
    console.print(f"HTML report: [bold]{path}[/]")


@app.command()
def baseline(
    run_id: int = typer.Argument(..., help="Run id to use as the baseline for its suite"),
    note: Optional[str] = typer.Option(None, help="Why this run is the baseline"),
) -> None:
    """Mark a run as the baseline for its suite."""
    repo = _repo()
    try:
        b = repo.set_baseline(run_id, note)
    except (KeyError, ValueError) as exc:
        _fail(str(exc))
    console.print(f"Baseline for [bold]{b.suite_name}[/] is now run {run_id}.")


@app.command()
def baselines() -> None:
    """List baseline history (the latest per suite is active)."""
    t = Table("Suite", "Run", "Set at", "Note", "Active")
    seen: set[str] = set()
    for b in _repo().list_baselines():
        t.add_row(b.suite_name, str(b.run_id), b.created_at.strftime("%Y-%m-%d %H:%M"), b.note or "",
                  "yes" if b.suite_name not in seen else "")
        seen.add(b.suite_name)
    console.print(t)


@app.command()
def runs(
    suite: Optional[str] = typer.Option(None, help="Filter by suite name"),
    limit: int = typer.Option(20, help="How many runs to show"),
) -> None:
    """List recent runs."""
    from app.services import run_summary

    repo = _repo()
    base_ids = repo.baseline_run_ids()
    t = Table("ID", "Date", "Suite", "Prompt", "Models", "Tests", "Accuracy", "P95 lat", "Cost", "Status")
    for col in t.columns:
        col.overflow = "fold"
    for r in repo.list_runs(suite, limit):
        full = repo.get_run(r.id)
        s = run_summary(full) if full.results else None
        t.add_row(
            f"{r.id}{' *' if r.id in base_ids else ''}", r.timestamp.strftime("%Y-%m-%d %H:%M"), r.suite_name,
            r.prompt_version, ", ".join(r.models), str(s["n"]) if s else "0",
            f"{s['pass_rate']:.1%}" if s else "-", f"{s['latency']['p95'] / 1000:.2f}s" if s else "-",
            money(s["total_cost"]) if s else "-", r.status,
        )
    console.print(t)
    console.print("[dim]* = has been a baseline[/]")


# --------------------------------------------------------------------------- judge reliability
@app.command()
def calibrate(
    run_id: int = typer.Argument(..., help="A run whose suite uses an llm_judge assertion"),
    labels_csv: Path = typer.Argument(..., help="CSV with columns test_id,human_score"),
    annotator: str = typer.Option("default", help="Annotator name stored with the labels"),
    model: Optional[str] = typer.Option(None, help="Only use judge scores for this evaluated model"),
    pass_threshold: float = typer.Option(4, help="Score >= threshold counts as 'pass' for binary agreement"),
    out: Optional[Path] = typer.Option(None, help="Write the full calibration result as JSON"),
) -> None:
    """Import human labels and measure agreement with the LLM judge."""
    from app.evaluation.calibration import calibrate as do_calibrate, judge_scores_from_run, load_human_labels_csv

    repo = _repo()
    run_ = _get_run(repo, run_id)
    try:
        rows = load_human_labels_csv(labels_csv)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    repo.import_human_labels(run_.suite_name, rows, annotator, str(labels_csv))
    judge = judge_scores_from_run(run_, model)
    if not judge:
        _fail(f"run {run_id} has no llm_judge scores")
    try:
        res = do_calibrate(judge, repo.get_human_labels(run_.suite_name, annotator), pass_threshold=pass_threshold)
    except ValueError as exc:
        _fail(str(exc))

    console.rule("[bold]Judge calibration vs human labels")
    console.print(f"n = {res['n']} paired items  (imported {len(rows)} labels for annotator '{annotator}')")
    t = Table("Metric", "Value")
    for k in ["exact_agreement", "within_one_agreement", "pass_fail_agreement"]:
        t.add_row(k, f"{res[k]:.1%}")
    for k in ["cohen_kappa", "weighted_kappa_quadratic", "pass_fail_kappa", "mean_absolute_difference",
              "mean_bias_judge_minus_human", "spearman_rho"]:
        t.add_row(k, f"{res[k]:.3f}")
    t.add_row("kappa interpretation", res["kappa_interpretation"] + " (Landis & Koch)")
    console.print(t)
    cm = res["confusion_matrix"]
    ct = Table(*(["human \\ judge"] + [str(c) for c in cm["labels"]]), title="Confusion matrix")
    for h in cm["labels"]:
        ct.add_row(str(h), *[str(cm["matrix"][h][j]) for j in cm["labels"]])
    console.print(ct)
    if res["small_sample_warning"]:
        console.print("[yellow]Fewer than 30 paired items: agreement statistics are very uncertain.[/]")
    console.print("[dim]Reminder: the judge is an automated opinion; human labels are the reference here.[/]")
    if out:
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        console.print(f"Wrote {out}")


@app.command()
def label(
    run_id: int = typer.Argument(..., help="Run whose outputs you want to label by hand"),
    out: Path = typer.Option(Path("data/human_labels.csv"), help="CSV to append labels to"),
    scale_max: int = typer.Option(5, help="Top of the 1..N scale"),
    limit: int = typer.Option(100, help="Max items to label"),
) -> None:
    """Interactively create human labels (1..N) for a run's outputs."""
    repo = _repo()
    run_ = _get_run(repo, run_id)
    done: set[str] = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            done = {r["test_id"] for r in csv.DictReader(fh)}
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    items = [r for r in run_.results if r.repeat_index == 0 and r.test_id not in done][:limit]
    console.print(f"{len(items)} items to label ({len(done)} already in {out}). Enter 1-{scale_max}, 's' to skip, 'q' to quit.")
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["test_id", "human_score"])
        for i, r in enumerate(items, 1):
            console.rule(f"{i}/{len(items)}  {r.test_id}")
            console.print(f"[bold]Input:[/] {json.dumps(r.vars, ensure_ascii=False)}")
            console.print(f"[bold]Expected:[/] {json.dumps(r.expected, ensure_ascii=False)}")
            console.print(f"[bold]Output:[/]\n{r.output}")
            ans = typer.prompt("score").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            if not ans.isdigit() or not 1 <= int(ans) <= scale_max:
                console.print("[yellow]skipped (invalid)[/]")
                continue
            w.writerow([r.test_id, int(ans)])
            fh.flush()
    console.print(f"Labels saved to {out}")


@app.command()
def pairwise(
    run_a: int = typer.Argument(...),
    run_b: int = typer.Argument(...),
    judge: str = typer.Option("mock/mock-judge", help="Judge model provider/model"),
    rubric: str = typer.Option("Which response classifies the ticket correctly as clean JSON?", help="Judging rubric"),
    randomize: bool = typer.Option(True, "--randomize/--no-randomize", help="Randomise A/B order to cancel position bias"),
    seed: int = typer.Option(0),
) -> None:
    """Pairwise LLM-judge comparison of two runs (with order randomisation)."""
    from app.evaluation.pairwise import pairwise_compare_runs
    from app.providers.registry import ProviderRegistry
    from app.runner.cache import ResponseCache
    from app.runner.cost import PricingTable
    from app.runner.generation import Generator
    from app.services import get_db

    repo = _repo()
    a, b = _get_run(repo, run_a), _get_run(repo, run_b)
    s = get_settings()
    gen = Generator(ProviderRegistry(s), PricingTable.from_file(s.pricing_path), ResponseCache(get_db()), s.max_concurrency)
    res = asyncio.run(pairwise_compare_runs(gen, a, b, judge, rubric, randomize, seed))
    console.print(f"Run {run_a} wins {res['a_wins']}, run {run_b} wins {res['b_wins']}, ties {res['ties']} (n = {res['n']})")
    console.print(f"Response shown first won {res['first_position_win_rate']:.0%} of the time "
                  f"({'randomised' if randomize else 'fixed'} order). Far from 50% with fixed order suggests position bias.")


# --------------------------------------------------------------------------- misc
@app.command("cache-clear")
def cache_clear() -> None:
    """Delete all cached responses."""
    from app.runner.cache import ResponseCache
    from app.services import get_db

    n = ResponseCache(get_db()).clear()
    console.print(f"Removed {n} cached responses.")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the dashboard (FastAPI). Optional: the CLI never needs it."""
    import uvicorn

    console.print(f"Dashboard: http://{host}:{port}   API docs: http://{host}:{port}/docs")
    uvicorn.run("app.dashboard.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
