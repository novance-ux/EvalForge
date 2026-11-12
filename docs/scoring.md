# Scoring

Every scorer returns:

```json
{"scorer_type": "equals", "score": 1.0, "passed": true, "reason": "$.category == 'billing'", "details": {...}}
```

- `score` is normalised to **[0, 1]**. Some scorers give partial credit (numeric tolerance, semantic similarity, LLM judge).
- `passed` is the binary verdict used for accuracy.
- `reason` is a one-line, human-readable explanation. It appears in the CLI, the HTML report and the dashboard.

A **test passes** only if every assertion passes. Its **score** is the weight-averaged scorer score.

## Output parsing

Before scoring, the raw output is parsed as JSON. Surrounding whitespace and a single Markdown code fence
(```` ```json … ``` ````) are accepted, because many models add them. Any other prose makes the output invalid JSON.
This strictness is intentional: downstream code would fail to parse such output too. The parse result drives the
`json_validity` metric.

## Scorers

| Type | Pass condition | Score |
|---|---|---|
| `exact_match` | normalised output (or `path` value) equals the expected value. Normalisation trims, collapses whitespace, strips quotes and a trailing period, and lowercases unless `case_sensitive` | 0/1 |
| `equals` | JSONPath value equals the expected value (strings normalised; other types use `==`) | 0/1 |
| `contains` | expected text is a substring of the normalised output | 0/1 |
| `regex` | `re.search(pattern)` matches; flags `IGNORECASE`, `MULTILINE`, `DOTALL` | 0/1 |
| `json_schema` | output is JSON and validates against the schema (Draft 2020-12). The first violation is reported with its location | 0/1 |
| `numeric_tolerance` | \|actual − expected\| ≤ tolerance; numbers are also extracted from text such as "$1,250.00" | 1 inside tolerance, decaying linearly to 0 at 2× tolerance |
| `latency_under_ms` | response latency ≤ limit | 0/1 |
| `semantic_similarity` | cosine(embed(output), embed(reference)) ≥ threshold | the similarity |
| `llm_judge` | mean judge score ≥ `pass_threshold` (default: 75% of the scale, i.e. 4 on 1–5) | (mean − min)/(max − min) |

Any value-based scorer accepts `value_from: expected.<field>` (or `vars.<field>`), so one `default_assert` can cover
a whole dataset.

## Embedding providers (semantic similarity)

`embedding:` picks the backend, so the system is not tied to one vendor:

- `local`: hashed bag-of-words plus character trigrams. Deterministic and offline; only suitable for tests and near-duplicate checks.
- `openai` or `openai/<model>`: `POST {OPENAI_BASE_URL}/embeddings`.
- `ollama` or `ollama/<model>`: `POST {OLLAMA_BASE_URL}/api/embed`.

## LLM judge

The judge prompt contains the **original input** (test vars), the **model output**, an optional **reference** (the
test's `expected`) and the **rubric**. It asks for `{"score": int, "reason": str}`.

Reliability features:

| Feature | How |
|---|---|
| Structured output | A JSON schema is sent as `response_format` (OpenAI) or `format` (Ollama). If the judge ignores it, a regex fallback reads `score: N`. |
| Multiple runs | `runs: N` samples the judge N times (use temperature > 0 so the samples can differ). Stored per test: `raw_scores`, `mean_score`, `std_score`, `agreement_rate` (share of runs that match the most common score). |
| Model separation | The judge model is configured separately (`judge.model` or `judge_model`). If it equals the evaluated model, the scorer records `same_model_as_evaluated=true` and adds a warning to the reason. |
| Pairwise + order randomisation | `pairwise_judge()` shows two outputs in random order and maps the verdict back. `llm-harness pairwise A B --no-randomize` lets you measure position bias. |
| Separate cost | Judge spend is stored as `judge_cost` and reported as evaluation cost, not as application cost. |

### Human calibration

```bash
llm-harness label RUN_ID --out data/human_labels.csv     # interactive 1–5 labelling
llm-harness calibrate RUN_ID data/human_labels.csv        # compare with the judge
```

CSV format:

```csv
test_id,human_score
refund-request,4
password-reset,2
```

Reported metrics (`app/evaluation/calibration.py`):

- **exact agreement** and **within-one agreement**
- **Cohen's kappa** (unweighted, treats scores as categories) and **quadratic-weighted kappa**, which gives partial credit for near misses on an ordinal scale
- **pass/fail agreement and kappa**, after binarising at the pass threshold
- **mean absolute difference** and **mean bias** (judge − human; positive means the judge is lenient)
- **Spearman ρ**
- **confusion matrix** (rows are human, columns are judge)

Kappa is interpreted with the Landis & Koch scale (slight, fair, moderate, substantial, almost perfect). That scale is
a convention, not a law. With fewer than 30 items the result is flagged as very uncertain.

> **Automated judge scores are not ground truth.** Treat them as an instrument that must be calibrated for each task.
> If kappa against humans is low, do not use the judge to make regression decisions.
