"""Pairwise (A vs B) judging of two runs, with output-order randomisation.

Running with ``randomize=False`` and then ``randomize=True`` is a simple way
to demonstrate position bias: a biased judge prefers whichever response is
shown first, which randomisation spreads evenly across both runs.
"""

from __future__ import annotations

import random
from typing import Any

from app.scorers.judge import pairwise_judge
from app.storage.db import Run


async def pairwise_compare_runs(
    generator,
    run_a: Run,
    run_b: Run,
    judge_model: str,
    rubric: str,
    randomize: bool = True,
    seed: int = 0,
    model_a: str | None = None,
    model_b: str | None = None,
) -> dict[str, Any]:
    def first_by_test(run: Run, model: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for r in run.results:
            if (model is None or r.model == model) and r.repeat_index == 0:
                out.setdefault(r.test_id, r)
        return out

    a, b = first_by_test(run_a, model_a), first_by_test(run_b, model_b)
    rng = random.Random(seed)
    verdicts = []
    for tid in sorted(set(a) & set(b)):
        v = await pairwise_judge(
            generator, judge_model, a[tid].vars, a[tid].output, b[tid].output, rubric,
            expected=a[tid].expected, randomize=randomize, rng=rng,
        )
        verdicts.append({"test_id": tid, **v})

    n = len(verdicts) or 1
    first_wins = sum(1 for v in verdicts if v["shown_winner"] == "A")
    return {
        "run_a": run_a.id,
        "run_b": run_b.id,
        "judge_model": judge_model,
        "randomized": randomize,
        "n": len(verdicts),
        "a_wins": sum(v["winner"] == "a" for v in verdicts),
        "b_wins": sum(v["winner"] == "b" for v in verdicts),
        "ties": sum(v["winner"] == "tie" for v in verdicts),
        "win_rate_a": sum(v["winner"] == "a" for v in verdicts) / n,
        "win_rate_b": sum(v["winner"] == "b" for v in verdicts) / n,
        "first_position_win_rate": first_wins / n,
        "verdicts": verdicts,
    }
