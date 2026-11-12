"""Shared fixtures. No test needs network access or an API key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config.settings import Settings  # noqa: E402
from app.models.results import LLMResponse  # noqa: E402
from app.providers.base import LLMProvider  # noqa: E402
from app.storage.db import Database  # noqa: E402


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        pricing_path=ROOT / "config" / "pricing.yaml",
        reports_dir=tmp_path / "reports",
        max_retries=2,
        timeout_s=5,
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    return Database(settings.db_path)


class ScriptedProvider(LLMProvider):
    """Returns canned text per prompt substring; counts calls."""

    name = "scripted"

    def __init__(self, replies: dict[str, str] | None = None, default: str = '{"category": "billing"}', latency_ms: float = 100):
        super().__init__(default_model="scripted-model", max_retries=0)
        self.replies = replies or {}
        self.default = default
        self.latency_ms = latency_ms
        self.calls = 0

    async def _request(self, prompt, model, temperature, max_tokens, response_schema, seed):
        self.calls += 1
        text = next((v for k, v in self.replies.items() if k in prompt), self.default)
        return LLMResponse(text=text, model=model, provider=self.name, input_tokens=100, output_tokens=10,
                           latency_ms=self.latency_ms)


@pytest.fixture
def scripted() -> ScriptedProvider:
    return ScriptedProvider()


def write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def mini_suite(tmp_path: Path) -> Path:
    """A tiny, self-contained suite in a temp folder."""
    write(tmp_path, "prompt.txt", 'Classify. Return ONLY valid JSON.\nTicket: """{{ ticket }}"""\n')
    write(tmp_path, "schema.json", '{"type":"object","properties":{"category":{"type":"string"}},"required":["category"]}')
    return write(
        tmp_path,
        "suite.yaml",
        """
name: mini
prompt: prompt.txt
models: [scripted/scripted-model]
classification: {path: $.category, expected_field: category}
default_assert:
  - {type: json_schema, schema: schema.json}
  - {type: equals, path: $.category, value_from: expected.category}
regression: {min_samples: 2}
tests:
  - {id: t1, vars: {ticket: "charged twice"}, expected: {category: billing}}
  - {id: t2, vars: {ticket: "forgot password"}, expected: {category: account}}
  - {id: t3, vars: {ticket: "app crashes"}, expected: {category: technical}}
""",
    )
