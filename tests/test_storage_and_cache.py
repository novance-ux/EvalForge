import pytest

from app.models.results import LLMResponse, ScoreResult, TestOutcome
from app.runner.cache import ResponseCache, cache_key
from app.storage.repository import Repository


def outcome(test_id="t1", passed=True):
    return TestOutcome(
        test_id=test_id, model="mock/mock-small", provider="mock", rendered_prompt="prompt " + test_id,
        output='{"category": "billing"}', passed=passed, score=1.0 if passed else 0.0, latency_ms=123,
        input_tokens=10, output_tokens=5, cost=0.0001, json_valid=True,
        scorer_results=[ScoreResult(scorer_type="equals", score=1.0, passed=passed, reason="ok", details={"name": "eq"})],
    )


def test_run_roundtrip(db):
    repo = Repository(db)
    run = repo.create_run(suite_name="s", prompt_version="v1", models=["mock/mock-small"], parameters={"temperature": 0})
    repo.save_outcomes(run.id, [outcome("t1"), outcome("t2", passed=False)])
    repo.finish_run(run.id, "completed")

    loaded = repo.get_run(run.id)
    assert loaded.status == "completed" and loaded.finished_at is not None
    assert [r.test_id for r in loaded.results] == ["t1", "t2"]
    assert loaded.results[0].input == "prompt t1"  # exact rendered prompt stored
    assert loaded.results[1].scorer_results[0].passed is False
    assert loaded.results[0].scorer_results[0].name == "eq"
    assert [r.id for r in repo.list_runs("s")] == [run.id]
    assert repo.count_test_results() == 2


def test_baselines(db):
    repo = Repository(db)
    r1 = repo.create_run(suite_name="s", prompt_version="v1", models=[], status="completed")
    r2 = repo.create_run(suite_name="s", prompt_version="v2", models=[], status="completed")
    assert repo.get_baseline("s") is None
    repo.set_baseline(r1.id)
    repo.set_baseline(r2.id, note="new prompt")
    assert repo.get_baseline("s").run_id == r2.id  # latest wins
    assert repo.baseline_run_ids() == {r1.id, r2.id}


def test_cannot_baseline_failed_or_missing_run(db):
    repo = Repository(db)
    bad = repo.create_run(suite_name="s", prompt_version="v1", models=[], status="failed")
    with pytest.raises(ValueError):
        repo.set_baseline(bad.id)
    with pytest.raises(KeyError):
        repo.set_baseline(9999)


def test_human_labels_upsert(db):
    repo = Repository(db)
    assert repo.import_human_labels("s", [("t1", 4), ("t2", 2)]) == 2
    repo.import_human_labels("s", [("t1", 5)])
    assert repo.get_human_labels("s") == {"t1": 5, "t2": 2}


# ------------------------------------------------------------------ cache
def test_cache_key_depends_on_all_inputs():
    base = dict(provider="p", model="m", prompt="x", parameters={"temperature": 0}, input_vars={"a": 1})
    k = cache_key(**base)
    assert k == cache_key(**base)
    for change in [{"provider": "q"}, {"model": "n"}, {"prompt": "y"}, {"parameters": {"temperature": 1}},
                   {"input_vars": {"a": 2}}, {"sample_index": 1}]:
        assert cache_key(**{**base, **change}) != k


def test_cache_hit_miss_and_cost_saved(db):
    cache = ResponseCache(db)
    resp = LLMResponse(text="hi", model="m", provider="p", input_tokens=1, output_tokens=1, latency_ms=50, cost=0.25)
    assert cache.get("k") is None
    cache.put("k", resp)
    hit = cache.get("k")
    assert hit.text == "hi" and hit.cached and hit.latency_ms == 50
    assert (cache.stats.hits, cache.stats.misses) == (1, 1)
    assert cache.stats.cost_saved == pytest.approx(0.25)
    assert cache.clear() == 1 and cache.get("k") is None


def test_disabled_cache(db):
    cache = ResponseCache(db, enabled=False)
    cache.put("k", LLMResponse(text="hi", model="m", provider="p"))
    assert cache.get("k") is None
    assert cache.stats.hits == cache.stats.misses == 0
