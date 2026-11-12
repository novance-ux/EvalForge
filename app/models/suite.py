"""Pydantic models describing the declarative YAML test-suite format.

Validation happens here so that a malformed suite fails fast with a precise
error location (e.g. ``tests[3].assert[1].path``) before any API call is made.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Models under test
# --------------------------------------------------------------------------- #
class ModelSpec(_Strict):
    """A provider + model pair, e.g. ``openai/gpt-4o-mini``.

    In YAML a model may be written as a plain string:
      * ``"openai"``               -> provider default model (from .env)
      * ``"ollama/llama3.2"``      -> explicit model
    or as a mapping with extra per-model parameters.
    """

    provider: str
    model: str | None = None
    label: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def parse(cls, value: "str | dict | ModelSpec") -> "ModelSpec":
        if isinstance(value, ModelSpec):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        provider, _, model = str(value).partition("/")
        return cls(provider=provider.strip(), model=model.strip() or None)

    def key(self) -> str:
        return f"{self.provider}/{self.model}" if self.model else self.provider


class GenerationParameters(_Strict):
    temperature: float = 0.0
    max_tokens: int | None = None


# --------------------------------------------------------------------------- #
# Assertions (one class per scorer type -> discriminated union)
# --------------------------------------------------------------------------- #
class _AssertionBase(_Strict):
    name: str | None = None
    weight: float = Field(1.0, ge=0)
    path: str | None = Field(
        None, description="Optional JSONPath applied to the parsed output before scoring."
    )

    @field_validator("path")
    @classmethod
    def _path_starts_with_dollar(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("$"):
            raise ValueError("JSONPath must start with '$', e.g. '$.category'")
        return v


class _ValueAssertion(_AssertionBase):
    value: Any = None
    value_from: str | None = Field(
        None, description="Take the expected value from the test case, e.g. 'expected.category'."
    )

    @model_validator(mode="after")
    def _need_value(self):
        if self.value is None and self.value_from is None:
            raise ValueError("either 'value' or 'value_from' is required")
        return self


class ExactMatchAssertion(_ValueAssertion):
    type: Literal["exact_match"]
    case_sensitive: bool = False


class EqualsAssertion(_ValueAssertion):
    type: Literal["equals"]
    case_sensitive: bool = False


class ContainsAssertion(_ValueAssertion):
    type: Literal["contains"]
    case_sensitive: bool = False


class RegexAssertion(_AssertionBase):
    type: Literal["regex"]
    pattern: str
    flags: list[Literal["IGNORECASE", "MULTILINE", "DOTALL"]] = Field(default_factory=list)


class JsonSchemaAssertion(_AssertionBase):
    type: Literal["json_schema"]
    schema_: str | dict[str, Any] = Field(alias="schema")


class NumericToleranceAssertion(_AssertionBase):
    type: Literal["numeric_tolerance"]
    expected: float | None = None
    value_from: str | None = None
    tolerance: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _need_expected(self):
        if self.expected is None and self.value_from is None:
            raise ValueError("either 'expected' or 'value_from' is required")
        return self


class LatencyAssertion(_AssertionBase):
    type: Literal["latency_under_ms"]
    value: float = Field(gt=0)


class SemanticSimilarityAssertion(_AssertionBase):
    type: Literal["semantic_similarity"]
    reference: str | None = None
    value_from: str | None = None
    threshold: float = Field(0.8, ge=-1, le=1)
    embedding: str = Field(
        "local", description="Embedding provider: 'local', 'openai[/model]' or 'ollama[/model]'."
    )


class LLMJudgeAssertion(_AssertionBase):
    type: Literal["llm_judge"]
    rubric: str
    judge_model: str | None = Field(None, description="Overrides the suite-level judge model.")
    runs: int | None = Field(None, ge=1, le=20)
    scale_min: int = 1
    scale_max: int = 5
    pass_threshold: float | None = None
    temperature: float | None = None


Assertion = Annotated[
    Union[
        ExactMatchAssertion,
        EqualsAssertion,
        ContainsAssertion,
        RegexAssertion,
        JsonSchemaAssertion,
        NumericToleranceAssertion,
        LatencyAssertion,
        SemanticSimilarityAssertion,
        LLMJudgeAssertion,
    ],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Suite-level configuration
# --------------------------------------------------------------------------- #
class RegressionThresholds(_Strict):
    accuracy_drop_threshold: float = Field(0.02, ge=0)
    score_drop_threshold: float = Field(0.02, ge=0)
    json_validity_drop_threshold: float = Field(0.02, ge=0)
    latency_increase_threshold: float = Field(0.30, ge=0)
    cost_increase_threshold: float = Field(0.50, ge=0)
    min_samples: int = Field(20, ge=1)
    confidence_level: float = Field(0.95, gt=0.5, lt=1)
    bootstrap_iterations: int = Field(2000, ge=100)


class JudgeConfig(_Strict):
    model: str = "mock/mock-judge"
    runs: int = Field(1, ge=1, le=20)
    temperature: float = 0.0


class ClassificationConfig(_Strict):
    """Optional: lets the harness compute accuracy / macro-F1 / confusion matrix."""

    path: str = "$.category"
    expected_field: str = "category"
    labels: list[str] = Field(default_factory=list)


class TestCase(_Strict):
    __test__ = False  # not a pytest test class

    id: str
    description: str | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list, alias="assert")

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("test id must not be empty")
        return v


class Suite(_Strict):
    name: str
    description: str | None = None
    prompt: str = Field(description="Path to the prompt template file.")
    models: list[ModelSpec] = Field(min_length=1)
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    default_assert: list[Assertion] = Field(
        default_factory=list, description="Assertions applied to every test case."
    )
    classification: ClassificationConfig | None = None
    regression: RegressionThresholds = Field(default_factory=RegressionThresholds)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    tests: list[TestCase] = Field(min_length=1)

    @field_validator("models", mode="before")
    @classmethod
    def _parse_models(cls, v: Any) -> Any:
        if not isinstance(v, list):
            raise ValueError("'models' must be a list")
        return [ModelSpec.parse(m) for m in v]

    @model_validator(mode="after")
    def _unique_test_ids(self):
        seen: set[str] = set()
        dupes = [t.id for t in self.tests if t.id in seen or seen.add(t.id)]
        if dupes:
            raise ValueError(f"duplicate test ids: {sorted(set(dupes))}")
        return self

    def assertions_for(self, test: TestCase) -> list[Any]:
        return [*self.default_assert, *test.assertions]
