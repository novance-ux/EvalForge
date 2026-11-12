# Architecture

## Design goals

1. **Provider-agnostic runner.** The evaluation engine does not know which vendor it is calling.
2. **One path for every LLM call.** The model under test and the judge both go through `Generator.generate`.
   Caching, cost and concurrency therefore behave the same way for both.
3. **Reproducible and auditable.** Every run stores the suite configuration, the prompt template and its hash, the git
   commit, and, for each test, the exact rendered prompt, raw output and every scorer verdict.
4. **Honest statistics.** Verdicts need both practical significance (a threshold) and statistical evidence (a CI that excludes 0).
5. **Small modules.** Each file does one thing and can be explained in a few minutes.

## Request lifecycle

```text
llm-harness run suite.yaml
  │
  ├─ config/suite_loader.load_suite()
  │     YAML → Pydantic Suite (structural validation)
  │     prompt exists? schemas valid? vars cover template? (semantic validation)
  │
  ├─ runner/runner.EvaluationRunner.run()
  │     create Run row (status=running, suite snapshot, prompt hash, git commit)
  │     for model × test × repeat  (asyncio.gather)
  │        templating.render(vars)                 ← sandboxed Jinja2, StrictUndefined
  │        generation.Generator.generate()
  │            cache.get(key) ── hit ──► return cached LLMResponse (cached=True)
  │            semaphore ► provider.generate()     ← retry/backoff/timeout in base class
  │            cost = pricing.cost(tokens)
  │            cache.put(key, response)
  │        parse JSON (tolerates one ``` fence)
  │        for assertion: build_scorer(spec).score(ctx)   ← registry lookup by `type`
  │        test score = weighted mean; passed = all passed
  │     save TestResults + ScorerResults; Run.status = completed
  │
  ├─ services.run_suite(): summarize() + compare_runs(baseline, run)
  └─ reporting: Rich terminal output, HTML report
```

## Key abstractions

| Abstraction | File | Contract |
|---|---|---|
| `LLMProvider` | `app/providers/base.py` | subclasses implement `_request()` (one call); the base class adds timeout, retries with exponential backoff and jitter, and latency measurement |
| `ProviderRegistry` | `app/providers/registry.py` | name → factory; one instance per provider per run |
| `Generator` | `app/runner/generation.py` | cache → semaphore → provider → pricing → cache |
| `Scorer` | `app/scorers/base.py` | `async score(ctx) -> ScoreResult(score∈[0,1], passed, reason, details)` |
| `EmbeddingProvider` | `app/scorers/embeddings.py` | `async embed(texts) -> vectors`; local, openai, ollama |
| `Repository` | `app/storage/repository.py` | the only module that queries the database |
| `compare_runs` | `app/regression/engine.py` | two runs → `ComparisonReport` with per-metric verdicts |

## Error handling

| Failure | Behaviour |
|---|---|
| Invalid suite | `SuiteError` listing every problem with its location. Nothing is executed. |
| 429 / 5xx / timeout / connection error | Retried up to `HARNESS_MAX_RETRIES` with exponential backoff. |
| 4xx (bad key, bad request) | Not retried. The test is recorded with `error` set and counted as failed. |
| Scorer cannot evaluate (no JSON, path missing) | That scorer fails with a clear reason. Other scorers still run. |
| Crash during a run | Run is marked `failed` with the exception stored. It cannot become a baseline. |

## Database schema

```text
runs            id, suite_name, suite_path, suite_hash, suite_config(JSON), prompt_version, prompt_path,
                prompt_hash, prompt_template, models(JSON), parameters(JSON), repeats, label,
                timestamp, finished_at, git_commit, status, error, cache_hits, cache_misses, cost_saved
test_results    id, run_id→runs, test_id, model, provider, repeat_index, input (rendered prompt), vars, expected,
                output, error, score, passed, latency_ms, input_tokens, output_tokens, cost, cached,
                json_valid, predicted_label, expected_label
scorer_results  id, test_result_id→test_results, scorer_type, name, score, passed, reason, weight, details(JSON)
baselines       id, suite_name, run_id→runs, created_at, note          (latest per suite is active)
cache_entries   key (sha256), provider, model, response(JSON), cost, created_at, hit_count
human_labels    id, suite_name, test_id, human_score, annotator, source, created_at
```

## Cache key

`sha256(provider, model, rendered prompt, {temperature, max_tokens, response_schema}, input vars, sample_index)`

The `sample_index` (the repeat number) means that repeated samples at temperature > 0 are cached separately. Without
it, every repeat would return the first sample. Use `--no-cache` to force fresh calls, as in the variability
experiment.

## Concurrency

A single `asyncio.Semaphore(max_concurrency)` in the `Generator` limits in-flight requests across all models and
judge calls in a run. Scoring runs outside the semaphore, but judge calls inside a scorer acquire it again.
