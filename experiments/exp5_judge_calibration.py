"""Experiment 5: how reliable is the LLM judge?

1. Runs the judged suite (each output scored 3x by a separate judge model).
2. Reports judge self-consistency (std, agreement rate across the 3 runs).
3. Compares judge scores with human labels: Cohen's kappa, weighted kappa,
   agreement, mean absolute difference, confusion matrix.
4. Pairwise judging of V1 vs V2 outputs with fixed vs randomised order to
   expose position bias.

HUMAN LABELS: put real annotations in data/human_labels.csv
(columns: test_id,human_score; scale 1-5). Create them with:

    llm-harness label RUN_ID --out data/human_labels.csv

If that file does not exist, the script generates
data/human_labels_SIMULATED.csv from the rubric plus random noise so the
pipeline can be demonstrated. SIMULATED LABELS ARE NOT HUMAN JUDGEMENTS and
must not be reported as such.

    python experiments/exp5_judge_calibration.py
    python experiments/exp5_judge_calibration.py --labels data/human_labels.csv --judge openai/gpt-4o-mini
"""

from __future__ import annotations

import asyncio
import csv
import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from common import JUDGED_SUITE, PROMPTS, ROOT, SUITE, banner, base_parser, save_charts, save_json, setup, style

from app.evaluation.calibration import calibrate, judge_scores_from_run, load_human_labels_csv
from app.evaluation.pairwise import pairwise_compare_runs
from app.providers.registry import ProviderRegistry
from app.runner.cache import ResponseCache
from app.runner.cost import PricingTable
from app.runner.generation import Generator
from app.runner.runner import RunOptions
from app.services import run_suite
from app.storage.repository import Repository
from app.utils.jsonparse import parse_json_output


def simulate_labels(run, path: Path, seed: int = 7) -> None:
    """Rubric-based pseudo-labels with 'annotator noise'. FOR DEMONSTRATION ONLY."""
    rng = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "human_score", "note"])
        for r in run.results:
            ok, parsed = parse_json_output(r.output)
            expected = str(r.expected.get("category", "")).lower()
            correct = expected and expected in r.output.lower()
            score = 5 if correct and ok else 3 if correct else 2 if ok and isinstance(parsed, dict) else 1
            if rng.random() < 0.15:
                score = min(5, max(1, score + rng.choice([-1, 1])))
            w.writerow([r.test_id, score, "SIMULATED"])


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default="mock/mock-small", help="evaluated model")
    p.add_argument("--judge", default=None, help="judge model (default: the suite's judge.model)")
    p.add_argument("--labels", default=str(ROOT / "data" / "human_labels.csv"))
    args = p.parse_args()
    banner([args.model] + ([args.judge] if args.judge else ["mock/mock-judge"]))
    settings, db = setup(args.db)

    judged = run_suite(JUDGED_SUITE, models=[args.model], db=db, settings=settings, auto_compare=False,
                       options=RunOptions(use_cache=not args.no_cache))
    if args.judge:
        print("Note: --judge only affects the pairwise part; set judge.model in the suite for the scored run.")
    js = judged.summary.get("judge", {})
    print(f"\nJudge self-consistency over {js.get('runs_per_test')} runs/test: mean within-test std "
          f"{js.get('mean_within_test_std', 0):.2f}, mean agreement rate {js.get('mean_agreement_rate', 0):.0%}")

    labels_path = Path(args.labels)
    simulated_labels = not labels_path.exists()
    if simulated_labels:
        labels_path = ROOT / "data" / "human_labels_SIMULATED.csv"
        simulate_labels(judged.run, labels_path)
        print(f"\n!! {args.labels} not found. Generated SIMULATED labels at {labels_path}.")
        print("!! Replace them with real human annotations (llm-harness label RUN_ID) before reporting results.")
    rows = load_human_labels_csv(labels_path)
    repo = Repository(db)
    annotator = "simulated" if simulated_labels else "human"
    repo.import_human_labels(judged.run.suite_name, rows, annotator, str(labels_path))
    judge = judge_scores_from_run(judged.run)
    cal = calibrate(judge, repo.get_human_labels(judged.run.suite_name, annotator))
    print(f"\nJudge vs {'SIMULATED ' if simulated_labels else ''}human labels (n = {cal['n']}):")
    for k in ["exact_agreement", "within_one_agreement", "pass_fail_agreement"]:
        print(f"  {k:28s} {cal[k]:.1%}")
    for k in ["cohen_kappa", "weighted_kappa_quadratic", "pass_fail_kappa", "mean_absolute_difference",
              "mean_bias_judge_minus_human", "spearman_rho"]:
        print(f"  {k:28s} {cal[k]:.3f}")
    print(f"  kappa interpretation         {cal['kappa_interpretation']} (Landis & Koch)")

    # Pairwise with / without order randomisation
    opts = RunOptions(use_cache=not args.no_cache)
    v1 = run_suite(SUITE, prompt=str(PROMPTS / "classifier_v1.txt"), models=[args.model], options=opts, db=db,
                   settings=settings, auto_compare=False)
    v2 = run_suite(SUITE, prompt=str(PROMPTS / "classifier_v2.txt"), models=[args.model], options=opts, db=db,
                   settings=settings, auto_compare=False)
    judge_model = args.judge or "mock/mock-judge"
    gen = Generator(ProviderRegistry(settings), PricingTable.from_file(settings.pricing_path), ResponseCache(db, enabled=False))
    rubric = "Which response gives the correct category as clean JSON?"
    pw = {}
    for randomize in (False, True):
        res = asyncio.run(pairwise_compare_runs(gen, v1.run, v2.run, judge_model, rubric, randomize=randomize, seed=1))
        pw["randomized" if randomize else "fixed"] = {k: v for k, v in res.items() if k != "verdicts"}
        print(f"\nPairwise V1 vs V2 ({'randomised' if randomize else 'fixed'} order): V1 wins {res['a_wins']}, "
              f"V2 wins {res['b_wins']}, ties {res['ties']}; first-shown response won {res['first_position_win_rate']:.0%}")

    # Charts
    pairs = cal["pairs"]
    rng = np.random.default_rng(0)
    fig1 = go.Figure(go.Scatter(
        x=[p_["human"] + rng.uniform(-0.15, 0.15) for p_ in pairs], y=[p_["judge"] + rng.uniform(-0.15, 0.15) for p_ in pairs],
        mode="markers", marker=dict(size=8, opacity=0.6, color="#2a6fdb"), text=[p_["test_id"] for p_ in pairs]))
    fig1.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(dash="dash", color="#888"))
    style(fig1, "Judge vs human score (jittered; dashed = perfect agreement)").update_layout(
        xaxis_title="human", yaxis_title="judge (mean of runs)")
    cm = cal["confusion_matrix"]
    z = [[cm["matrix"][h][j] for j in cm["labels"]] for h in cm["labels"]]
    fig2 = go.Figure(go.Heatmap(z=z, x=[f"judge {j}" for j in cm["labels"]], y=[f"human {h}" for h in cm["labels"]],
                                colorscale="Blues", text=z, texttemplate="%{text}", showscale=False))
    style(fig2, f"Confusion matrix (kappa = {cal['cohen_kappa']:.2f}, weighted = {cal['weighted_kappa_quadratic']:.2f})",
          420).update_layout(yaxis_autorange="reversed")
    agree = [s.details.get("agreement_rate", 1) for r in judged.run.results for s in r.scorer_results
             if s.scorer_type == "llm_judge"]
    fig3 = go.Figure(go.Histogram(x=agree, marker_color="#e0703c", xbins=dict(size=0.1)))
    style(fig3, "Judge self-agreement across repeated judge runs", 300).update_layout(xaxis_title="agreement rate")
    fig4 = go.Figure([
        go.Bar(name="first-shown wins", x=list(pw), y=[pw[k]["first_position_win_rate"] for k in pw], marker_color="#9b59b6"),
        go.Bar(name="V2 win rate", x=list(pw), y=[pw[k]["win_rate_b"] for k in pw], marker_color="#2a6fdb"),
    ])
    style(fig4, "Pairwise judging: fixed vs randomised order", 320).update_layout(barmode="group", yaxis_tickformat=".0%")

    notes = [f"Evaluated model {args.model}; judge {', '.join(js.get('judge_models', []))}.",
             "Automated judge scores are opinions, not ground truth."]
    if simulated_labels:
        notes.insert(0, "<b>The 'human' labels here are SIMULATED.</b> Replace data/human_labels.csv with real annotations.")
    path = save_charts("exp5_judge_calibration", "Experiment 5: LLM judge reliability", [fig1, fig2, fig3, fig4], notes,
                       simulated=True if simulated_labels else args.model.startswith("mock"))
    data = {"judged_run": judged.run.id, "labels_file": str(labels_path), "simulated_labels": simulated_labels,
            "judge_self_consistency": js, "calibration": {k: v for k, v in cal.items() if k != "pairs"},
            "pairwise": pw}
    print(f"\nCharts: {path}\nData:   {save_json('exp5_judge_calibration', data)}")


if __name__ == "__main__":
    main()
