import pytest

from app.models.results import LLMResponse
from app.models.suite import (
    ContainsAssertion,
    EqualsAssertion,
    ExactMatchAssertion,
    JsonSchemaAssertion,
    LatencyAssertion,
    NumericToleranceAssertion,
    RegexAssertion,
    SemanticSimilarityAssertion,
    TestCase,
)
from app.scorers.base import ScoringContext, build_scorer, registered_scorers, run_scorer
from app.utils.jsonparse import parse_json_output

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["billing", "account"]}},
    "required": ["category"],
}


def ctx(output: str, expected: dict | None = None, latency: float = 100.0) -> ScoringContext:
    ok, parsed = parse_json_output(output)
    return ScoringContext(
        output=output, parsed_ok=ok, parsed=parsed,
        test=TestCase(id="t", vars={"ticket": "x"}, expected=expected or {}),
        rendered_prompt="p",
        response=LLMResponse(text=output, model="m", provider="p", latency_ms=latency),
    )


async def score(spec, c):
    return await run_scorer(build_scorer(spec), c)


def test_all_scorers_registered():
    assert {"exact_match", "equals", "contains", "regex", "json_schema", "numeric_tolerance",
            "latency_under_ms", "semantic_similarity", "llm_judge"} <= set(registered_scorers())


# ------------------------------------------------------------------ exact match
async def test_exact_match_normalises():
    spec = ExactMatchAssertion(type="exact_match", value="Billing")
    assert (await score(spec, ctx("  billing.\n"))).passed
    assert not (await score(spec, ctx("billing issue"))).passed


async def test_exact_match_case_sensitive():
    spec = ExactMatchAssertion(type="exact_match", value="Billing", case_sensitive=True)
    assert not (await score(spec, ctx("billing"))).passed


async def test_exact_match_value_from_expected():
    spec = ExactMatchAssertion(type="exact_match", value_from="expected.category")
    r = await score(spec, ctx("account", {"category": "account"}))
    assert r.passed and r.score == 1.0


# ------------------------------------------------------------------ equals / contains
async def test_equals_jsonpath():
    spec = EqualsAssertion(type="equals", path="$.category", value="billing")
    assert (await score(spec, ctx('{"category": "billing"}'))).passed
    r = await score(spec, ctx('{"category": "account"}'))
    assert not r.passed and "account" in r.reason


async def test_equals_on_invalid_json_fails_gracefully():
    spec = EqualsAssertion(type="equals", path="$.category", value="billing")
    r = await score(spec, ctx("Category: billing"))
    assert not r.passed and "not valid JSON" in r.reason


async def test_equals_missing_path():
    spec = EqualsAssertion(type="equals", path="$.label", value="billing")
    r = await score(spec, ctx('{"category": "billing"}'))
    assert not r.passed and "not found" in r.reason


async def test_contains():
    spec = ContainsAssertion(type="contains", value="refund")
    assert (await score(spec, ctx("I will process your REFUND now"))).passed
    assert not (await score(spec, ctx("no luck"))).passed


# ------------------------------------------------------------------ regex
async def test_regex_scorer():
    spec = RegexAssertion(type="regex", pattern=r"^INV-\d+$")
    assert (await score(spec, ctx("INV-1234"))).passed
    assert not (await score(spec, ctx("invoice 1234"))).passed


async def test_regex_flags_and_path():
    spec = RegexAssertion(type="regex", pattern="^billing$", flags=["IGNORECASE"], path="$.category")
    assert (await score(spec, ctx('{"category": "BILLING"}'))).passed


async def test_invalid_regex_is_reported():
    r = await score(RegexAssertion(type="regex", pattern="("), ctx("x"))
    assert not r.passed and "invalid regex" in r.reason


# ------------------------------------------------------------------ json schema
async def test_json_schema_valid():
    r = await score(JsonSchemaAssertion(type="json_schema", schema=SCHEMA), ctx('{"category": "billing"}'))
    assert r.passed


async def test_json_schema_violation():
    r = await score(JsonSchemaAssertion(type="json_schema", schema=SCHEMA), ctx('{"category": "payments"}'))
    assert not r.passed and "category" in r.reason


async def test_json_schema_not_json():
    r = await score(JsonSchemaAssertion(type="json_schema", schema=SCHEMA), ctx("Sure! billing"))
    assert not r.passed and r.reason == "Output is not valid JSON"


# ------------------------------------------------------------------ numeric
async def test_numeric_tolerance():
    spec = NumericToleranceAssertion(type="numeric_tolerance", expected=100, tolerance=5)
    assert (await score(spec, ctx("104"))).passed
    r = await score(spec, ctx("The total is $107.50"))
    assert not r.passed and 0 < r.score < 1  # partial credit
    assert (await score(spec, ctx("200"))).score == 0


async def test_numeric_from_json_path_and_expected():
    spec = NumericToleranceAssertion(type="numeric_tolerance", path="$.total", value_from="expected.total", tolerance=0.01)
    assert (await score(spec, ctx('{"total": 1250.0}', {"total": 1250}))).passed


# ------------------------------------------------------------------ latency
async def test_latency_scorer():
    spec = LatencyAssertion(type="latency_under_ms", value=3000)
    assert (await score(spec, ctx("x", latency=2999))).passed
    r = await score(spec, ctx("x", latency=3500))
    assert not r.passed and "3500" in r.reason


# ------------------------------------------------------------------ semantic
async def test_semantic_similarity_local():
    spec = SemanticSimilarityAssertion(type="semantic_similarity", reference="refund my order please", threshold=0.5)
    assert (await score(spec, ctx("please refund my order"))).passed
    assert not (await score(spec, ctx("the weather is sunny"))).passed


def test_invalid_assertion_config():
    with pytest.raises(ValueError):
        EqualsAssertion(type="equals", path="$.a")  # neither value nor value_from
