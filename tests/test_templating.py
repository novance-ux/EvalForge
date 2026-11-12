import pytest

from app.runner.templating import PromptRenderError, PromptTemplate


def test_render_variables():
    t = PromptTemplate("Ticket:\n{{ ticket }}\nEnd")
    assert t.render({"ticket": "I was charged twice"}) == "Ticket:\nI was charged twice\nEnd"
    assert t.variables == {"ticket"}


def test_missing_variable_is_an_error():
    with pytest.raises(PromptRenderError, match="ticket"):
        PromptTemplate("{{ ticket }}").render({})


def test_no_html_escaping_in_prompts():
    assert PromptTemplate("{{ x }}").render({"x": "<b>&\"'</b>"}) == "<b>&\"'</b>"


def test_sandbox_blocks_python_internals():
    t = PromptTemplate("{{ x.__class__.__mro__[1].__subclasses__() }}")
    with pytest.raises(Exception):
        t.render({"x": ""})


def test_invalid_template_syntax():
    with pytest.raises(PromptRenderError):
        PromptTemplate("{{ unclosed ")


def test_version_from_filename(tmp_path):
    p = tmp_path / "classifier_v9.txt"
    p.write_text("{{ a }}", encoding="utf-8")
    assert PromptTemplate.from_file(p).version == "classifier_v9"
