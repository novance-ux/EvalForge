"""Runtime settings loaded from environment variables (and an optional .env file).

Secrets such as API keys are only ever read from the environment; nothing is
hard-coded here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    db_path: Path = Path("harness.db")
    pricing_path: Path = Path("config/pricing.yaml")
    reports_dir: Path = Path("reports")
    max_concurrency: int = 4
    timeout_s: float = 60.0
    max_retries: int = 3

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)
    return Settings(
        db_path=Path(_env("HARNESS_DB_PATH", "harness.db")),
        pricing_path=Path(_env("HARNESS_PRICING_PATH", "config/pricing.yaml")),
        reports_dir=Path(_env("HARNESS_REPORTS_DIR", "reports")),
        max_concurrency=int(_env("HARNESS_MAX_CONCURRENCY", "4")),
        timeout_s=float(_env("HARNESS_TIMEOUT_S", "60")),
        max_retries=int(_env("HARNESS_MAX_RETRIES", "3")),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.2"),
        ollama_embedding_model=_env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )
