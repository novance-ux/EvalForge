"""SQLAlchemy ORM models and session factory (SQLite).

Enough is stored to *reproduce and audit* any evaluation: the exact prompt
template and its hash, the full suite configuration, the rendered prompt sent
for every test, the raw output, every scorer's verdict and the git commit.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.utils.misc import utcnow


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite_name: Mapped[str] = mapped_column(String(200), index=True)
    suite_path: Mapped[str | None] = mapped_column(String(500))
    suite_hash: Mapped[str | None] = mapped_column(String(64))
    suite_config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    prompt_version: Mapped[str] = mapped_column(String(200))
    prompt_path: Mapped[str | None] = mapped_column(String(500))
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    prompt_template: Mapped[str | None] = mapped_column(Text)
    models: Mapped[list[str]] = mapped_column(JSON, default=list)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    repeats: Mapped[int] = mapped_column(Integer, default=1)
    label: Mapped[str | None] = mapped_column(String(200))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="running")
    error: Mapped[str | None] = mapped_column(Text)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_misses: Mapped[int] = mapped_column(Integer, default=0)
    cost_saved: Mapped[float] = mapped_column(Float, default=0.0)

    results: Mapped[list["TestResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="TestResult.id"
    )


class TestResult(Base):
    __test__ = False  # not a pytest test class
    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    test_id: Mapped[str] = mapped_column(String(200), index=True)
    model: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str] = mapped_column(String(50))
    repeat_index: Mapped[int] = mapped_column(Integer, default=0)
    input: Mapped[str] = mapped_column(Text)  # the exact rendered prompt
    vars: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    expected: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    json_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    predicted_label: Mapped[str | None] = mapped_column(String(200))
    expected_label: Mapped[str | None] = mapped_column(String(200))

    run: Mapped[Run] = relationship(back_populates="results")
    scorer_results: Mapped[list["ScorerResult"]] = relationship(
        back_populates="test_result", cascade="all, delete-orphan", order_by="ScorerResult.id"
    )


class ScorerResult(Base):
    __tablename__ = "scorer_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    test_result_id: Mapped[int] = mapped_column(
        ForeignKey("test_results.id", ondelete="CASCADE"), index=True
    )
    scorer_type: Mapped[str] = mapped_column(String(50))
    name: Mapped[str | None] = mapped_column(String(200))
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text, default="")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    test_result: Mapped[TestResult] = relationship(back_populates="scorer_results")


class Baseline(Base):
    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite_name: Mapped[str] = mapped_column(String(200), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    note: Mapped[str | None] = mapped_column(Text)


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(200))
    response: Mapped[dict[str, Any]] = mapped_column(JSON)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)


class HumanLabel(Base):
    __tablename__ = "human_labels"
    __table_args__ = (UniqueConstraint("suite_name", "test_id", "annotator"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite_name: Mapped[str] = mapped_column(String(200), index=True)
    test_id: Mapped[str] = mapped_column(String(200))
    human_score: Mapped[float] = mapped_column(Float)
    annotator: Mapped[str] = mapped_column(String(100), default="default")
    source: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Database:
    def __init__(self, url_or_path: str | Path):
        url = str(url_or_path)
        if "://" not in url:
            Path(url).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{url}"
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def session(self):
        return self.Session()
