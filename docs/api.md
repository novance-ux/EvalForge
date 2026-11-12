# Dashboard API

Start with `llm-harness serve` (default `http://127.0.0.1:8000`). Interactive OpenAPI docs are at `/docs`.
The API is **read-only** and has **no authentication**. Run it locally only.

| Method | Path | Description |
|---|---|---|
| GET | `/` | Single-page dashboard (overview, run history, comparison, drilldown) |
| GET | `/api/overview` | Totals: runs, tests, pass rate, regressions, average cost per run, average latency, suites, active baselines |
| GET | `/api/runs?suite=&limit=50` | Run history: id, timestamp, suite, prompt version, models, accuracy, JSON validity, cost, mean/P95 latency, `is_baseline`, `regression_status` against the current baseline |
| GET | `/api/runs/{id}` | One run: the history fields plus parameters, prompt template and hash, cache stats, full `summary`, `by_model` summaries |
| GET | `/api/runs/{id}/tests?failed_only=false` | Per-test results of a run |
| GET | `/api/compare?baseline=A&current=B[&baseline_model=&current_model=]` | Full comparison report (metrics, CIs, verdicts, per-test changes) |
| GET | `/api/compare/{A}/{B}/tests/{test_id}` | Drilldown: input, expected result, baseline and current outputs, scores, per-scorer reasons, latency, cost, tokens, rendered prompt, regression reason |
| GET | `/api/baselines` | Baseline history |
| GET | `/reports/{id}?baseline=A` | Renders and returns the HTML report |

## Example

```bash
curl "http://127.0.0.1:8000/api/compare?baseline=1&current=2" | jq '.overall, .metrics[] | {name, baseline, current, status}'
```

```json
{
  "baseline_run_id": 1,
  "current_run_id": 2,
  "overall": "REGRESSED",
  "quality": "IMPROVED",
  "efficiency": "REGRESSED",
  "regressed_metrics": ["cost_per_test"],
  "improved_metrics": ["accuracy", "label_accuracy", "mean_score", "json_validity"],
  "n_pairs": 67,
  "metrics": [
    {"name": "accuracy", "baseline": 0.567, "current": 0.881, "abs_diff": 0.313, "pct_diff": 0.553,
     "ci": {"estimate": 0.313, "low": 0.194, "high": 0.448, "level": 0.95},
     "threshold": 0.02, "threshold_kind": "absolute", "status": "IMPROVED",
     "note": "exceeds threshold; CI excludes 0", "p_value": 0.00001}
  ],
  "tests": [{"test_id": "return-label", "change": "regressed", "baseline_output": "...", "current_output": "...", "reasons": ["equals: ..."]}]
}
```

## Using the API from Python

The dashboard is a thin layer over `app.services` and `app.regression.engine`. Scripts can call those directly
without running the server:

```python
from app.services import compare
report = compare(1, 2)
print(report.overall, report.regressed_metrics)
```
