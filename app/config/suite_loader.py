"""Load and validate YAML test suites.

Validation happens in two passes:

1. **Structural** - Pydantic checks types, required fields and assertion types.
2. **Semantic**   - referenced files exist, JSON schemas are valid, every
   template variable used by the prompt is provided by every test case.

All problems are collected and raised together in a single ``SuiteError``
so users can fix a broken suite in one go.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import ValidationError

from app.models.suite import JsonSchemaAssertion, Suite
from app.runner.templating import PromptRenderError, PromptTemplate


class SuiteError(Exception):
    def __init__(self, path: str | Path, problems: list[str]):
        self.path = str(path)
        self.problems = problems
        super().__init__(f"Invalid suite {path}:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass
class LoadedSuite:
    suite: Suite
    path: Path
    template: PromptTemplate
    schemas: dict[str, dict[str, Any]]  # resolved json_schema references
    raw: dict[str, Any]

    @property
    def prompt_version(self) -> str:
        return self.template.version


def resolve_path(ref: str | Path, suite_path: Path) -> Path:
    """Resolve a path relative to the CWD first, then to the suite file's folder."""
    p = Path(ref)
    if p.is_absolute() or p.exists():
        return p
    alt = suite_path.parent / p
    return alt if alt.exists() else p


def _format_loc(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out or "<root>"


def _format_validation_error(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        # Drop the discriminator tag Pydantic inserts (e.g. 'equals') for readability.
        loc = tuple(p for p in err["loc"] if not (isinstance(p, str) and p.endswith("Assertion")))
        msg = err["msg"].replace("Value error, ", "")
        problems.append(f"{_format_loc(loc)}: {msg}")
    return problems


def schema_key(ref: str | dict[str, Any]) -> str:
    """Key under which a resolved schema is stored in ``LoadedSuite.schemas``."""
    return json.dumps(ref, sort_keys=True) if isinstance(ref, dict) else ref


def load_schema(ref: str | dict[str, Any], suite_path: Path) -> dict[str, Any]:
    if isinstance(ref, dict):
        schema = ref
    else:
        schema_path = resolve_path(ref, suite_path)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def load_suite(
    path: str | Path,
    prompt_override: str | Path | None = None,
    models_override: list[str] | None = None,
) -> LoadedSuite:
    path = Path(path)
    if not path.exists():
        raise SuiteError(path, [f"file not found: {path}"])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteError(path, [f"YAML syntax error: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise SuiteError(path, ["top level of the suite must be a mapping"])

    # `include_tests: other_suite.yaml` reuses the test cases of another suite.
    include = raw.pop("include_tests", None)
    if include:
        inc_path = resolve_path(include, path)
        try:
            inc_raw = yaml.safe_load(inc_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise SuiteError(path, [f"include_tests: cannot read {include}: {exc}"]) from exc
        raw = {**raw, "tests": [*(inc_raw.get("tests") or []), *(raw.get("tests") or [])]}

    if prompt_override:
        raw = {**raw, "prompt": str(prompt_override)}
    if models_override:
        raw = {**raw, "models": list(models_override)}

    try:
        suite = Suite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteError(path, _format_validation_error(exc)) from exc

    problems: list[str] = []

    # Prompt template
    template: PromptTemplate | None = None
    prompt_path = resolve_path(suite.prompt, path)
    if not prompt_path.exists():
        problems.append(f"prompt: file not found: {suite.prompt}")
    else:
        try:
            template = PromptTemplate.from_file(prompt_path)
        except PromptRenderError as exc:
            problems.append(f"prompt: {exc}")

    # JSON schemas
    schemas: dict[str, dict[str, Any]] = {}
    all_assertions = [("default_assert", i, a) for i, a in enumerate(suite.default_assert)]
    for ti, test in enumerate(suite.tests):
        all_assertions += [(f"tests[{ti}].assert", i, a) for i, a in enumerate(test.assertions)]
    for where, i, a in all_assertions:
        if isinstance(a, JsonSchemaAssertion):
            key = schema_key(a.schema_)
            if key in schemas:
                continue
            try:
                schemas[key] = load_schema(a.schema_, path)
            except FileNotFoundError:
                problems.append(f"{where}[{i}].schema: file not found: {a.schema_}")
            except (json.JSONDecodeError, jsonschema.SchemaError) as exc:
                problems.append(f"{where}[{i}].schema: invalid JSON schema: {exc}")

    # Every test must supply every template variable
    if template is not None:
        for ti, test in enumerate(suite.tests):
            missing = template.variables - set(test.vars)
            if missing:
                problems.append(
                    f"tests[{ti}] ({test.id}): missing prompt variables {sorted(missing)}"
                )

    if problems:
        raise SuiteError(path, problems)

    assert template is not None
    return LoadedSuite(suite=suite, path=path, template=template, schemas=schemas, raw=raw)
