"""End-to-end runner tests using in-process providers (no network)."""

import asyncio

from app.config.suite_loader import load_suite
from app.providers.registry import ProviderRegistry
from app.runner.cost import ModelPrice, PricingTable
from app.runner.runner import EvaluationRunner, RunOptions
from app.services import run_suite
from app.storage.repository import Repository
from tests.conftest import ScriptedProvider

REPLIES = {"charged": '{"category": "billing"}', "password": '{"category": "account"}', "crashes": "not json"}


def make_runner(db, settings, provider):
    reg = ProviderRegistry(settings, overrides={"scripted": provider})
    pricing = PricingTable({"scripted/scripted-model": ModelPrice(1_000_000, 2_000_000)})  # $1/in-token, $2/out-token
    return EvaluationRunner(db, settings, registry=reg, pricing=pricing)


def test_run_scores_and_persists(db, settings, mini_suite):
    provider = ScriptedProvider(REPLIES)
    result = asyncio.run(make_runner(db, settings, provider).run(load_suite(mini_suite)))
    by_id = {o.test_id: o for o in result.outcomes}
    assert by_id["t1"].passed and by_id["t2"].passed
    assert not by_id["t3"].passed and not by_id["t3"].json_valid
    assert by_id["t1"].cost == 100 * 1 + 10 * 2  # from the pricing table, not the provider
    assert by_id["t1"].predicted_label == "billing"

    run = Repository(db).get_run(result.run_id)
    assert run.status == "completed" and run.prompt_version == "prompt"
    assert run.prompt_template.startswith("Classify")
    assert len(run.results) == 3
    assert 'Ticket: """charged twice"""' in run.results[0].input


def test_cache_prevents_second_call(db, settings, mini_suite):
    provider = ScriptedProvider(REPLIES)
    r1 = asyncio.run(make_runner(db, settings, provider).run(load_suite(mini_suite)))
    r2 = asyncio.run(make_runner(db, settings, provider).run(load_suite(mini_suite)))
    assert provider.calls == 3
    assert (r1.cache.misses, r2.cache.hits) == (3, 3)
    assert r2.cache.cost_saved == sum(o.cost for o in r1.outcomes)
    assert all(o.cached for o in r2.outcomes)


def test_no_cache_option(db, settings, mini_suite):
    provider = ScriptedProvider(REPLIES)
    for _ in range(2):
        asyncio.run(make_runner(db, settings, provider).run(load_suite(mini_suite), RunOptions(use_cache=False)))
    assert provider.calls == 6


def test_repeats(db, settings, mini_suite):
    provider = ScriptedProvider(REPLIES)
    res = asyncio.run(make_runner(db, settings, provider).run(load_suite(mini_suite), RunOptions(repeats=3)))
    assert len(res.outcomes) == 9
    assert sorted({o.repeat_index for o in res.outcomes}) == [0, 1, 2]


def test_provider_error_is_recorded_not_raised(db, settings, mini_suite):
    class Broken(ScriptedProvider):
        async def _request(self, *a):
            from app.providers.base import ProviderError

            raise ProviderError("invalid request")

    res = asyncio.run(make_runner(db, settings, Broken()).run(load_suite(mini_suite)))
    assert all(o.error and not o.passed for o in res.outcomes)


def test_concurrency_is_limited(db, settings, mini_suite):
    class Counting(ScriptedProvider):
        active = peak = 0

        async def _request(self, *a):
            Counting.active += 1
            Counting.peak = max(Counting.peak, Counting.active)
            await asyncio.sleep(0.01)
            Counting.active -= 1
            return await super()._request(*a)

    asyncio.run(make_runner(db, settings, Counting()).run(load_suite(mini_suite),
                                                          RunOptions(repeats=4, max_concurrency=2, use_cache=False)))
    assert Counting.peak == 2


def test_run_suite_service_compares_with_baseline(db, settings, mini_suite):
    good = ScriptedProvider(REPLIES)
    bad = ScriptedProvider({}, default="nope")
    reg = lambda p: ProviderRegistry(settings, overrides={"scripted": p})  # noqa: E731
    first = run_suite(mini_suite, db=db, settings=settings, registry=reg(good))
    assert first.comparison is None
    Repository(db).set_baseline(first.run.id)
    second = run_suite(mini_suite, db=db, settings=settings, registry=reg(bad), options=RunOptions(use_cache=False))
    assert second.comparison is not None
    assert second.comparison.baseline_run_id == first.run.id
    assert second.summary["pass_rate"] == 0


def test_demo_suite_with_mock_provider(db, settings, root):
    """The bundled demo runs fully offline and the regressed prompt is caught."""
    suite = root / "suites" / "support_classifier.yaml"
    v2 = run_suite(suite, prompt=str(root / "prompts" / "classifier_v2.txt"), db=db, settings=settings)
    Repository(db).set_baseline(v2.run.id)
    bad = run_suite(suite, prompt=str(root / "prompts" / "classifier_regressed.txt"), db=db, settings=settings)
    assert v2.summary["pass_rate"] > 0.8
    assert bad.comparison.regression_detected
    assert "accuracy" in bad.comparison.regressed_metrics
