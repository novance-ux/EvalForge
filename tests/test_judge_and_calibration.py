import random

import pytest

from app.evaluation.calibration import calibrate, cohen_kappa, load_human_labels_csv
from app.models.results import LLMResponse
from app.models.suite import LLMJudgeAssertion, TestCase
from app.providers.registry import ProviderRegistry
from app.runner.cost import PricingTable
from app.runner.generation import Generator
from app.scorers.base import ScoringContext, build_scorer
from app.scorers.judge import agreement_rate, pairwise_judge, parse_judge_score
from tests.conftest import ScriptedProvider


def gen(provider, settings):
    return Generator(ProviderRegistry(settings, overrides={"scripted": provider}), PricingTable(), cache=None)


def ctx(generator, output='{"category": "billing"}'):
    return ScoringContext(
        output=output, parsed_ok=True, parsed={}, rendered_prompt="p", generator=generator,
        test=TestCase(id="t", vars={"ticket": "charged twice"}, expected={"category": "billing"}),
        extra={"model_key": "other/model"},
    )


def test_parse_judge_score():
    assert parse_judge_score('{"score": 4, "reason": "good"}', 1, 5) == (4, "good")
    assert parse_judge_score("Score: 3 because...", 1, 5)[0] == 3
    with pytest.raises(ValueError):
        parse_judge_score('{"score": 9}', 1, 5)
    with pytest.raises(ValueError):
        parse_judge_score("no idea", 1, 5)


def test_agreement_rate():
    assert agreement_rate([4, 4, 4]) == 1.0
    assert agreement_rate([4, 4, 2]) == pytest.approx(2 / 3)


async def test_judge_multiple_runs_statistics(settings):
    class Varying(ScriptedProvider):
        scores = iter([5, 4, 5])

        async def _request(self, *a):
            return LLMResponse(text=f'{{"score": {next(self.scores)}, "reason": "r"}}', model="j", provider="scripted")

    spec = LLMJudgeAssertion(type="llm_judge", rubric="correct?", judge_model="scripted/j", runs=3)
    r = await build_scorer(spec).score(ctx(gen(Varying(), settings)))
    assert r.details["raw_scores"] == [5, 4, 5]
    assert r.details["mean_score"] == pytest.approx(14 / 3)
    assert r.details["std_score"] > 0
    assert r.details["agreement_rate"] == pytest.approx(2 / 3)
    assert r.passed and 0.9 < r.score < 1
    assert r.details["same_model_as_evaluated"] is False


async def test_judge_sends_structured_output_schema(settings):
    seen = {}

    class Capture(ScriptedProvider):
        supports_structured_output = True

        async def _request(self, prompt, model, temperature, max_tokens, response_schema, seed):
            seen["schema"] = response_schema
            seen["prompt"] = prompt
            return LLMResponse(text='{"score": 2, "reason": "wrong"}', model=model, provider="scripted")

    spec = LLMJudgeAssertion(type="llm_judge", rubric="Is it correct?", judge_model="scripted/j")
    r = await build_scorer(spec).score(ctx(gen(Capture(), settings)))
    assert not r.passed
    assert seen["schema"]["properties"]["score"]["maximum"] == 5
    assert "Is it correct?" in seen["prompt"] and "charged twice" in seen["prompt"]


async def test_judge_warns_when_same_model(settings):
    p = ScriptedProvider(default='{"score": 5, "reason": "ok"}')
    spec = LLMJudgeAssertion(type="llm_judge", rubric="r", judge_model="scripted/j")
    c = ctx(gen(p, settings))
    c.extra["model_key"] = "scripted/j"
    r = await build_scorer(spec).score(c)
    assert r.details["same_model_as_evaluated"] and "warning" in r.reason


async def test_pairwise_randomisation_maps_winner_back(settings):
    # A judge that always prefers whatever is shown first (pure position bias).
    p = ScriptedProvider(default='{"winner": "A", "reason": "first"}')
    g = gen(p, settings)
    rng = random.Random(0)
    wins = {"a": 0, "b": 0}
    for _ in range(200):
        v = await pairwise_judge(g, "scripted/j", {"q": 1}, "out a", "out b", "rubric", randomize=True, rng=rng)
        wins[v["winner"]] += 1
    assert 70 < wins["a"] < 130  # randomisation spreads the bias across both sides
    fixed = await pairwise_judge(g, "scripted/j", {"q": 1}, "out a", "out b", "rubric", randomize=False)
    assert fixed["winner"] == "a" and not fixed["swapped"]


# ------------------------------------------------------------------ calibration
def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)
    # Classic 2x2 example: po=0.7, pe=0.5 -> kappa 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_weighted_kappa_penalises_near_misses_less():
    human = [1, 2, 3, 4, 5] * 4
    near = [min(5, h + 1) for h in human]
    assert cohen_kappa(human, near, labels=[1, 2, 3, 4, 5], weights="quadratic") > cohen_kappa(human, near, labels=[1, 2, 3, 4, 5])


def test_calibrate():
    judge = {"a": 5, "b": 4, "c": 2, "d": 1, "e": 4.33}
    human = {"a": 5, "b": 3, "c": 2, "d": 2, "e": 4, "z": 3}
    res = calibrate(judge, human)
    assert res["n"] == 5
    assert res["exact_agreement"] == pytest.approx(3 / 5)
    assert res["within_one_agreement"] == 1.0
    assert res["mean_absolute_difference"] == pytest.approx((0 + 1 + 0 + 1 + 0.33) / 5, abs=1e-3)
    assert res["missing_judge_scores"] == ["z"]
    assert res["confusion_matrix"]["matrix"][3][4] == 1  # human 3, judge 4
    assert res["small_sample_warning"]


def test_load_human_labels_csv(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("test_id,human_score,notes\n1,4,x\n2,2,\n3,,skip\n", encoding="utf-8")
    assert load_human_labels_csv(p) == [("1", 4.0), ("2", 2.0)]
    bad = tmp_path / "bad.csv"
    bad.write_text("id,score\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test_id,human_score"):
        load_human_labels_csv(bad)
