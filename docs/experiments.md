# Experiments

All scripts live in `experiments/`. Each one writes an HTML page of charts and a JSON file to `reports/experiments/`,
and uses its own database (`reports/experiments/experiments.db`, which you can change with `--db`).

> **Simulated vs real results.** By default every experiment uses the offline **mock provider**. It is a
> deterministic keyword classifier whose accuracy depends on prompt features (JSON instructions, category definitions,
> a list of allowed labels). This lets the whole pipeline be demonstrated with no API key, but **the numbers are
> simulated**. For the project report, rerun each experiment with real models, for example
> `--models openai/gpt-4o-mini ollama/llama3.2` or `--model ollama/llama3.2`. The HTML outputs carry a "Simulated
> data" banner whenever a mock model was used.

## Experiment 1: model comparison

`python experiments/exp1_model_comparison.py [--models A B] [--prompt FILE]`

- **Question:** which model is better for this task, and at what price?
- **Method:** the same suite and prompt (v2) on two models. Paired comparison A → B.
- **Metrics:** accuracy (pass rate), macro-F1, JSON validity, mean and P95 latency, total cost, cost per correct prediction, F1 per category.
- **Charts:** quality bars, latency box plot, cost per correct, F1 per category.
- **Reading the output:** a more accurate model is only worth it if the accuracy gain is not INCONCLUSIVE and the
  extra cost per correct answer is acceptable. In the simulated run, mock-large is +3 pp more accurate
  (INCONCLUSIVE, CI includes 0), 2.7× slower and 10× more expensive.

## Experiment 2: prompt v1 vs v2

`python experiments/exp2_prompt_comparison.py [--model M]`

- v1: category list plus a loose "respond in JSON".
- v2: adds category definitions, a strict "ONLY valid JSON" instruction, "do not invent categories" and a
  prompt-injection guard.
- **Outputs:** every metric with its paired CI and verdict, per-test changes (fixed or broken), the tests v2 broke,
  F1 per category, and the relative latency and cost change.

## Experiment 3: detecting an intentional regression

`python experiments/exp3_regression_detection.py [--model M] [--trials 30]`

- **Part A:** the baseline is v2. The candidate is `classifier_regressed.txt`, which has no JSON requirement and no
  category list. **Success criterion:** `regression_detected == True` (the script prints PASS or FAIL).
- **Part B (sensitivity):** k = 0…12 random passing tests are flipped to failing in a copy of the baseline, with 30
  trials per k. The chart shows how often each verdict occurs. k = 0 measures the false-alarm rate. The smallest k that
  is always REGRESSED is the detection limit for this suite size.

## Experiment 4: run-to-run variability

`python experiments/exp4_variability.py [--runs 5] [--temperatures 0 0.7] [--model M]`

- The same configuration is run N times with the cache disabled, at each temperature.
- **Measures:** mean and standard deviation of the pass rate, its range, latency coefficient of variation, the number
  of tests whose verdict flips between runs, and the **false regression rate**. The false regression rate is the
  share of pairs of identical-configuration runs that the engine calls REGRESSED; it should be about 0%.
- **Why it matters:** if repeated runs of the same prompt differ by ±4 pp, a 3 pp "improvement" from a new prompt is
  meaningless. The engine should say INCONCLUSIVE, not IMPROVED.

## Experiment 5: judge reliability and human calibration

`python experiments/exp5_judge_calibration.py [--labels data/human_labels.csv] [--model M] [--judge J]`

1. Runs `suites/support_classifier_judged.yaml`. Each output is scored 3 times by a separate judge model at temperature 0.7.
2. **Self-consistency:** mean within-test standard deviation and agreement rate across the judge runs.
3. **Human agreement:** Cohen's kappa, quadratic-weighted kappa, exact and within-one agreement, pass/fail kappa,
   mean absolute difference, bias, Spearman ρ, and a confusion matrix.
4. **Position bias:** pairwise judging of v1 vs v2 outputs with a fixed order (v1 always shown first) and with a
   randomised order. The "first-shown win rate" shows how much the judge favours position A.

**Collecting human labels (required for real results).** Label 50–100 outputs yourself, ideally with a second
annotator for inter-annotator agreement:

```bash
llm-harness run suites/support_classifier_judged.yaml         # note the run id
llm-harness label <RUN_ID> --out data/human_labels.csv         # score each output 1–5 using the rubric
python experiments/exp5_judge_calibration.py --labels data/human_labels.csv
```

If `data/human_labels.csv` is missing, the script writes `data/human_labels_SIMULATED.csv` (rubric plus random
noise) so the pipeline can run. **Simulated labels are not human judgements and must not be reported as such.**

## Suggested write-up table

| Experiment | Question | Key number(s) | Verdict |
|---|---|---|---|
| 1 | Is model B worth it? | Δaccuracy [CI], cost per correct A vs B | |
| 2 | Did v2 improve on v1? | Δaccuracy [CI], fixed / broken tests, Δcost | |
| 3 | Is a degraded prompt caught? | detected (yes/no), detection limit k | |
| 4 | How noisy are runs? | pass-rate SD, flip rate, false-alarm rate | |
| 5 | Can we trust the judge? | κ, weighted κ, MAD, first-position win rate | |
