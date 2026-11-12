import pytest

from app.utils.jsonparse import parse_json_output
from app.utils.jsonpath import JSONPathError, extract, find_all

DOC = {"category": "billing", "items": [{"name": "a", "n": 1}, {"name": "b", "n": 2}], "odd key": {"x": 1}}


@pytest.mark.parametrize(
    "path, expected",
    [
        ("$", DOC),
        ("$.category", "billing"),
        ("$.items[0].name", "a"),
        ("$.items[-1].n", 2),
        ("$['odd key'].x", 1),
    ],
)
def test_extract(path, expected):
    assert extract(DOC, path) == expected


def test_wildcard():
    assert find_all(DOC, "$.items[*].name") == ["a", "b"]


def test_missing_path():
    assert find_all(DOC, "$.nope") == []
    assert extract(DOC, "$.nope", default=None) is None
    with pytest.raises(KeyError):
        extract(DOC, "$.items[5]")


def test_invalid_path():
    with pytest.raises(JSONPathError):
        find_all(DOC, "category")
    with pytest.raises(JSONPathError):
        find_all(DOC, "$..category")


@pytest.mark.parametrize(
    "text, ok",
    [
        ('{"a": 1}', True),
        ('  {"a": 1}\n', True),
        ('```json\n{"a": 1}\n```', True),
        ('Sure! {"a": 1}', False),
        ("Category: billing", False),
        ("", False),
    ],
)
def test_parse_json_output(text, ok):
    assert parse_json_output(text)[0] is ok
