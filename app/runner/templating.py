"""Safe prompt rendering with Jinja2.

* A sandboxed environment prevents templates from calling arbitrary Python.
* ``StrictUndefined`` makes a missing variable an error instead of silently
  rendering an empty string (a classic source of silent prompt regressions).
* Autoescaping is off: prompts are plain text, not HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, TemplateError, meta
from jinja2.sandbox import SandboxedEnvironment

_env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)


class PromptRenderError(ValueError):
    pass


class PromptTemplate:
    def __init__(self, source: str, path: Path | None = None):
        self.source = source
        self.path = path
        try:
            self._template = _env.from_string(source)
            self.variables = meta.find_undeclared_variables(_env.parse(source))
        except TemplateError as exc:
            raise PromptRenderError(f"Invalid prompt template {path or ''}: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        p = Path(path)
        return cls(p.read_text(encoding="utf-8"), p)

    @property
    def version(self) -> str:
        """Human-readable version: the file stem, e.g. ``classifier_v2``."""
        return self.path.stem if self.path else "inline"

    def render(self, variables: dict[str, Any]) -> str:
        missing = self.variables - set(variables)
        if missing:
            raise PromptRenderError(f"Missing template variables: {sorted(missing)}")
        try:
            return self._template.render(**variables)
        except TemplateError as exc:
            raise PromptRenderError(f"Failed to render prompt: {exc}") from exc
