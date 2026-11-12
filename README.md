# LLM Regression Harness

LLM Regression Harness is a Python command-line tool for checking whether a change to an LLM prompt or model improves or hurts your application. Define repeatable test cases and expected results in YAML, run them against a model, and compare the results with a saved baseline.

The project includes an offline mock provider, so you can try the workflow without an API key. It also supports OpenAI-compatible APIs and Ollama.

## What it does

- Loads and validates YAML evaluation suites and Jinja2 prompt templates.
- Runs test cases against one or more configured models.
- Scores outputs with assertions such as exact match, JSON Schema, JSONPath, regex, numeric tolerance, semantic similarity, and an optional LLM judge.
- Saves prompts, outputs, scores, token usage, latency, and estimated cost in a local SQLite database.
- Compares runs against a baseline and reports changes with paired statistical checks.
- Produces terminal and HTML reports. An optional FastAPI dashboard shows run history and comparisons.

This is a small, self-contained evaluation project. It does not guarantee that a prompt will behave the same way in production; results depend on the quality of your test cases, model settings, and provider.

## Requirements

- Python 3.11 or newer
- An API key only when using a hosted provider; the mock provider works offline
- Ollama installed and running only when using an Ollama model

## Install

From the project directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package and development tools:

```bash
python -m pip install -e ".[dev]"
llm-harness init
```

`llm-harness init` creates the local results database and copies `.env.example` to `.env` if needed. Add credentials to `.env` only if you plan to use a real provider. Do not commit `.env`.

## Quick start: offline demo

The included support ticket suite uses the mock provider by default:

```bash
llm-harness validate suites/support_classifier.yaml
llm-harness run suites/support_classifier.yaml --set-baseline
llm-harness run suites/support_classifier.yaml --prompt prompts/classifier_v2.txt
llm-harness runs
```

The first run becomes the baseline. The second run is evaluated with the alternate prompt and compared to it. Each run also writes an HTML report under `reports/` by default. Mock results are simulated and are useful for learning the tool, not for judging real model quality.

## Run with a real model

For an OpenAI-compatible provider, set `OPENAI_API_KEY` in `.env` (and `OPENAI_BASE_URL` if using a compatible service):

```bash
llm-harness run suites/support_classifier.yaml --model openai/gpt-4o-mini
```

For Ollama, start the Ollama service and pull a model first:

```bash
ollama pull llama3.2
llm-harness run suites/support_classifier.yaml --model ollama/llama3.2
```

You can run multiple models together by repeating `--model`:

```bash
llm-harness run suites/support_classifier.yaml \
  --model openai/gpt-4o-mini \
  --model ollama/llama3.2
```

Hosted model calls may incur charges. Estimated costs use `config/pricing.yaml`; update its example prices before relying on cost figures. Models without a configured price are marked unpriced.

## Compare and manage baselines

```bash
# List stored runs
llm-harness runs

# Mark an existing run as the baseline for its suite
llm-harness baseline 1 --note "Approved prompt"

# Compare two run IDs; add --html to save an HTML comparison
llm-harness compare 1 2 --html

# Set a baseline while running, and fail with exit code 2 on regression
llm-harness run suites/support_classifier.yaml --set-baseline
llm-harness run suites/support_classifier.yaml --fail-on-regression
```

Regression results can be `IMPROVED`, `REGRESSED`, `NO_CHANGE`, or `INCONCLUSIVE`. The suite config sets metric thresholds and a minimum paired sample count. An inconclusive result means the observed change did not meet the configured statistical evidence requirements; it is not proof that the runs are equivalent.

Useful options for `run` include `--repeat N`, `--no-cache`, `--temperature VALUE`, `--label TEXT`, `--baseline RUN_ID`, and `--no-report`. See all commands and options with:

```bash
llm-harness --help
llm-harness run --help
```

## Create your own suite

A suite YAML file connects a prompt, model, test data, and assertions. This shortened example shows the main pieces:

```yaml
name: sentiment-check
prompt: prompts/sentiment.txt
models:
  - mock/mock-small
parameters:
  temperature: 0

default_assert:
  - type: equals
    path: $.sentiment
    value_from: expected.sentiment

tests:
  - id: positive-message
    vars:
      text: "I love this product."
    expected:
      sentiment: positive
```

The prompt can use variables such as `{{ text }}`. Test variables are substituted with a sandboxed Jinja2 template; undefined variables cause validation to fail. Paths to prompt and schema files are resolved from the project directory or the suite file's directory. Assertions may be shared across all tests with `default_assert` or specified on an individual test with `assert`.

Check a suite before running it:

```bash
llm-harness validate suites/my-suite.yaml
llm-harness run suites/my-suite.yaml
```

See [suites/support_classifier.yaml](suites/support_classifier.yaml) for a full example and [docs/scoring.md](docs/scoring.md) for assertion details.

## Providers and configuration

Available providers are:

- `mock`: offline deterministic demo provider
- `openai`: OpenAI and compatible chat completion APIs
- `ollama`: a local Ollama server

Provider and harness defaults are configured with environment variables. Copy `.env.example` to `.env` (or run `llm-harness init`) and set only the values you need. Common settings include `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`, `HARNESS_DB_PATH`, `HARNESS_REPORTS_DIR`, `HARNESS_MAX_CONCURRENCY`, `HARNESS_TIMEOUT_S`, and `HARNESS_MAX_RETRIES`.

## Reports and dashboard

Run reports are generated automatically unless `--no-report` is used. To regenerate one for a saved run:

```bash
llm-harness report 1
```

Start the optional local dashboard with:

```bash
llm-harness serve
```

Open <http://127.0.0.1:8000>; API documentation is at <http://127.0.0.1:8000/docs>. The dashboard reads data from the same local SQLite database used by the CLI.

## Experiments and tests

The `experiments/` directory contains scripts for model and prompt comparisons, regression detection, run variability, and judge calibration. For example:

```bash
python experiments/exp2_prompt_comparison.py
```

These scripts use the mock provider by default. Their mock results demonstrate the evaluation workflow and should not be presented as real model measurements. Some experiments support selecting real models; use each script's `--help` for options. Judge calibration should use actual human labels when drawing conclusions about judge agreement.

Run the automated test suite with:

```bash
pytest -q
```

## Project layout

```text
app/          CLI, providers, runner, scorers, evaluation, storage, reports, dashboard
config/       Example model pricing
data/         Label templates and example data
docs/         Architecture, scoring, regression, experiments, and API notes
experiments/  Evaluation and calibration scripts
prompts/      Example prompt versions
schemas/      JSON Schemas used by example suites
suites/       Example YAML test suites
tests/        Automated tests
```

## Further reading

- [Architecture](docs/architecture.md)
- [Scoring and assertions](docs/scoring.md)
- [Regression comparison](docs/regression.md)
- [Experiments](docs/experiments.md)
- [Dashboard API](docs/api.md)

## License

No license file is currently included. Add a license before publishing if you want to specify how others may use, modify, or distribute this project.
