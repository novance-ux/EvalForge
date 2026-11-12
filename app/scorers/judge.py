"""LLM-as-a-judge scorer and pairwise judge.

Reliability features:
* **Structured output** - a JSON schema is sent to providers that support it.
* **Multiple runs** - the judge can be sampled N times; mean, standard
  deviation and agreement rate are recorded for every test.
* **Model separation** - the judge model is configured independently of the
  evaluated model, and a warning is recorded when they are the same.
* **Order randomisation** (pairwise) - the two candidate outputs are shown in
  random order to cancel position bias.

Judge scores are *automated opinions*, not ground truth. See
``app/evaluation/calibration.py`` for measuring agreement with humans.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from typing import Any

from app.models.results import ScoreResult
from app.models.suite import ModelSpec
from app.scorers.base import Scorer, ScorerInputError, ScoringContext, register_scorer
from app.utils.jsonparse import parse_json_output

JUDGE_TEMPLATE = """You are an IMPARTIAL EVALUATOR of AI assistant outputs.
Score the MODEL OUTPUT against the RUBRIC using an integer from {scale_min} (worst) to {scale_max} (best).
Judge only what is asked in the rubric. Do not reward length.

<INPUT>
{input}
</INPUT>

<MODEL OUTPUT>
{output}
</MODEL OUTPUT>

<REFERENCE>
{reference}
</REFERENCE>

<RUBRIC>
{rubric}
</RUBRIC>

Respond ONLY with JSON of the form {{"score": <integer>, "reason": "<one sentence>"}}."""

PAIRWISE_TEMPLATE = """You are an IMPARTIAL EVALUATOR. PAIRWISE comparison.
Two AI responses answer the same input. Decide which better satisfies the RUBRIC.
Ignore the order in which the responses are presented and do not reward length.

<INPUT>
{input}
</INPUT>

<REFERENCE>
{reference}
</REFERENCE>

<RESPONSE A>
{a}
</RESPONSE A>

<RESPONSE B>
{b}
</RESPONSE B>

<RUBRIC>
{rubric}
</RUBRIC>

Respond ONLY with JSON: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}."""


def judge_schema(scale_min: int, scale_max: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": scale_min, "maximum": scale_max},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
    }


PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}, "reason": {"type": "string"}},
    "required": ["winner", "reason"],
}


def format_input(vars_: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in vars_.items())


def format_reference(expected: dict[str, Any]) -> str:
    if not expected:
        return "(none)"
    if len(expected) == 1:
        return str(next(iter(expected.values())))
    return json.dumps(expected, ensure_ascii=False)


def parse_judge_score(text: str, scale_min: int, scale_max: int) -> tuple[int, str]:
    ok, data = parse_json_output(text)
    if ok and isinstance(data, dict) and "score" in data:
        score, reason = data["score"], str(data.get("reason", ""))
    else:  # tolerate judges that ignore the JSON instruction
        m = re.search(r'"?score"?\s*[:=]\s*(\d+)', text, re.I)
        if not m:
            raise ValueError(f"unparseable judge output: {text[:120]!r}")
        score, reason = m.group(1), text.strip()[:200]
    score = int(round(float(score)))
    if not scale_min <= score <= scale_max:
        raise ValueError(f"judge score {score} outside [{scale_min}, {scale_max}]")
    return score, reason


def agreement_rate(scores: list[int]) -> float:
    """Fraction of runs that agree with the most common score."""
    if not scores:
        return 0.0
    return Counter(scores).most_common(1)[0][1] / len(scores)


@register_scorer("llm_judge")
class LLMJudgeScorer(Scorer):
    async def score(self, ctx: ScoringContext) -> ScoreResult:
        if ctx.generator is None:
            raise ScorerInputError("llm_judge requires a generator (runner context)")
        suite_judge = ctx.loaded.suite.judge if ctx.loaded else None
        judge_model = self.spec.judge_model or (suite_judge.model if suite_judge else "mock/mock-judge")
        runs = self.spec.runs or (suite_judge.runs if suite_judge else 1)
        temperature = (
            self.spec.temperature
            if self.spec.temperature is not None
            else (suite_judge.temperature if suite_judge else 0.0)
        )
        lo, hi = self.spec.scale_min, self.spec.scale_max
        threshold = self.spec.pass_threshold if self.spec.pass_threshold is not None else lo + 0.75 * (hi - lo)

        prompt = JUDGE_TEMPLATE.format(
            scale_min=lo,
            scale_max=hi,
            input=format_input(ctx.test.vars),
            output=ctx.output,
            reference=format_reference(ctx.test.expected),
            rubric=self.spec.rubric.strip(),
        )
        scores: list[int] = []
        reasons: list[str] = []
        errors: list[str] = []
        judge_cost = 0.0
        for i in range(runs):
            resp = await ctx.generator.generate(
                judge_model, prompt, temperature=temperature,
                response_schema=judge_schema(lo, hi), sample_index=i,
            )
            judge_cost += resp.cost
            try:
                s, r = parse_judge_score(resp.text, lo, hi)
                scores.append(s)
                reasons.append(r)
            except ValueError as exc:
                errors.append(str(exc))

        evaluated = ctx.extra.get("model_key")
        same_model = evaluated is not None and ModelSpec.parse(judge_model).key() == evaluated
        details: dict[str, Any] = {
            "judge_model": judge_model,
            "runs": runs,
            "temperature": temperature,
            "raw_scores": scores,
            "judge_errors": errors,
            "judge_cost": judge_cost,
            "scale": [lo, hi],
            "same_model_as_evaluated": same_model,
        }
        if not scores:
            return self.result(0.0, False, f"Judge produced no parseable score ({errors[0] if errors else ''})", **details)

        mean = statistics.fmean(scores)
        details.update(
            mean_score=mean,
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            agreement_rate=agreement_rate(scores),
        )
        passed = mean >= threshold
        reason = f"Judge mean {mean:.2f}/{hi} over {len(scores)} run(s): {reasons[0]}"
        if same_model:
            reason += " [warning: judge model == evaluated model]"
        return self.result((mean - lo) / (hi - lo), passed, reason, **details)


async def pairwise_judge(
    generator,
    judge_model: str,
    input_vars: dict[str, Any],
    output_a: str,
    output_b: str,
    rubric: str,
    expected: dict[str, Any] | None = None,
    randomize: bool = True,
    rng: random.Random | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Compare two outputs. Returns winner in terms of the *original* labels a/b."""
    rng = rng or random.Random()
    swapped = randomize and rng.random() < 0.5
    first, second = (output_b, output_a) if swapped else (output_a, output_b)
    prompt = PAIRWISE_TEMPLATE.format(
        input=format_input(input_vars),
        reference=format_reference(expected or {}),
        a=first,
        b=second,
        rubric=rubric.strip(),
    )
    resp = await generator.generate(judge_model, prompt, temperature=temperature, response_schema=PAIRWISE_SCHEMA)
    ok, data = parse_json_output(resp.text)
    shown = str(data.get("winner", "")).strip() if ok and isinstance(data, dict) else ""
    if shown not in {"A", "B", "tie"}:
        m = re.search(r"\b(A|B|tie)\b", resp.text)
        shown = m.group(1) if m else "tie"
    if shown == "tie":
        winner = "tie"
    else:
        picked_first = shown == "A"
        winner = ("b" if picked_first else "a") if swapped else ("a" if picked_first else "b")
    return {"winner": winner, "shown_winner": shown, "swapped": swapped, "cost": resp.cost}
