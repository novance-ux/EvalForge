import pytest

from app.config.suite_loader import SuiteError, load_suite
from app.models.suite import EqualsAssertion, ModelSpec
from tests.conftest import write


def test_load_demo_suite(root):
    loaded = load_suite(root / "suites" / "support_classifier.yaml")
    s = loaded.suite
    assert s.name == "support-ticket-classifier"
    assert len(s.tests) >= 50
    assert {t.expected["category"] for t in s.tests} == {
        "billing", "account", "technical", "refund", "shipping", "cancellation"
    }
    assert loaded.prompt_version == "classifier_v1"
    assert loaded.template.variables == {"ticket"}


def test_all_bundled_suites_validate(root):
    for path in (root / "suites").glob("*.yaml"):
        load_suite(path)


def test_model_string_parsing():
    assert ModelSpec.parse("openai/gpt-4o-mini").model == "gpt-4o-mini"
    spec = ModelSpec.parse("ollama")
    assert spec.provider == "ollama" and spec.model is None


def test_value_from_assertion(mini_suite):
    loaded = load_suite(mini_suite)
    eq = loaded.suite.default_assert[1]
    assert isinstance(eq, EqualsAssertion) and eq.value_from == "expected.category"


def test_prompt_and_model_overrides(mini_suite, tmp_path):
    other = write(tmp_path, "other.txt", "Other {{ ticket }}")
    loaded = load_suite(mini_suite, prompt_override=other, models_override=["mock/mock-large"])
    assert loaded.prompt_version == "other"
    assert loaded.suite.models[0].key() == "mock/mock-large"


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("name: x\nprompt: p.txt\nmodels: [mock]\ntests: []\n", "tests"),
        ("name: x\nprompt: p.txt\nmodels: [mock]\ntests:\n  - {id: a, vars: {ticket: q}, assert: [{type: nope}]}\n", "tests[0].assert[0]"),
        ("name: x\nprompt: p.txt\nmodels: [mock]\ntests:\n  - {id: a, vars: {ticket: q}, assert: [{type: equals, path: category, value: 1}]}\n", "JSONPath must start"),
        ("name: x\nprompt: p.txt\nmodels: [mock]\ntests:\n  - {id: a, vars: {ticket: q}}\n  - {id: a, vars: {ticket: q}}\n", "duplicate test ids"),
        ("name: x\nprompt: p.txt\nmodels: [mock]\ntests:\n  - {id: a, vars: {other: q}}\n", "missing prompt variables"),
        ("name: x\nprompt: missing.txt\nmodels: [mock]\ntests:\n  - {id: a}\n", "file not found"),
        ("name: x\nprompt: p.txt\nmodels: [mock]\ntests:\n  - {id: a, vars: {ticket: q}, assert: [{type: json_schema, schema: nope.json}]}\n", "nope.json"),
        ("name: [unclosed\n", "YAML syntax error"),
        ("name: x\nprompt: p.txt\nmodels: [mock]\nunknown_key: 1\ntests:\n  - {id: a, vars: {ticket: q}}\n", "unknown_key"),
    ],
)
def test_invalid_suites_give_useful_errors(tmp_path, body, fragment):
    write(tmp_path, "p.txt", "T: {{ ticket }}")
    path = write(tmp_path, "s.yaml", body)
    with pytest.raises(SuiteError) as exc:
        load_suite(path)
    assert fragment in str(exc.value), str(exc.value)


def test_include_tests(root):
    loaded = load_suite(root / "suites" / "support_classifier_judged.yaml")
    base = load_suite(root / "suites" / "support_classifier.yaml")
    assert [t.id for t in loaded.suite.tests] == [t.id for t in base.suite.tests]
