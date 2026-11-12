import pytest
from fastapi.testclient import TestClient

import app.dashboard.api as api
from app.services import run_suite
from app.storage.repository import Repository


@pytest.fixture
def client(db, settings, root, monkeypatch):
    suite = root / "suites" / "support_classifier.yaml"
    a = run_suite(suite, db=db, settings=settings)
    Repository(db).set_baseline(a.run.id)
    run_suite(suite, prompt=str(root / "prompts" / "classifier_regressed.txt"), db=db, settings=settings)
    monkeypatch.setattr(api, "get_db", lambda: db)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    return TestClient(api.app)


def test_overview(client):
    o = client.get("/api/overview").json()
    assert o["total_runs"] == 2 and o["regressions"] == 1
    assert 0 <= o["pass_rate"] <= 1


def test_runs_and_detail(client):
    runs = client.get("/api/runs").json()
    assert [r["regression_status"] for r in runs] == ["REGRESSED", None]
    detail = client.get(f"/api/runs/{runs[0]['id']}").json()
    assert detail["summary"]["n"] >= 50 and "latency" in detail["summary"]
    assert client.get("/api/runs/999").status_code == 404


def test_compare_and_drilldown(client):
    c = client.get("/api/compare?baseline=1&current=2").json()
    assert c["overall"] == "REGRESSED"
    regressed = next(t for t in c["tests"] if t["change"] == "regressed")
    d = client.get(f"/api/compare/1/2/tests/{regressed['test_id']}").json()
    assert d["regression_reason"]
    assert d["baseline"][0]["passed"] and not d["current"][0]["passed"]
    assert "expected" in d and "input" in d


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "LLM Regression Harness" in r.text
