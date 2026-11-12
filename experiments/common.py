"""Shared helpers for experiment scripts.

Every experiment:
  * uses its own SQLite file (default: reports/experiments/experiments.db) so the
    main harness.db stays clean,
  * writes <name>_results.json and an HTML page of Plotly charts to
    reports/experiments/,
  * defaults to the offline mock provider. Pass --model to use real models.

Numbers produced with mock/* models are SIMULATED. Do not report them as
real model behaviour; rerun with real providers for your final results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402

from app.config.settings import get_settings  # noqa: E402
from app.storage.db import Database  # noqa: E402

OUT = ROOT / "reports" / "experiments"
SUITE = ROOT / "suites" / "support_classifier.yaml"
JUDGED_SUITE = ROOT / "suites" / "support_classifier_judged.yaml"
PROMPTS = ROOT / "prompts"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--db", default=str(OUT / "experiments.db"), help="SQLite file for this experiment")
    p.add_argument("--no-cache", action="store_true", help="Disable the response cache")
    return p


def setup(db_path: str):
    OUT.mkdir(parents=True, exist_ok=True)
    settings = get_settings().model_copy(update={"db_path": Path(db_path)})
    return settings, Database(settings.db_path)


def is_simulated(models: list[str]) -> bool:
    return any(m.startswith("mock") for m in models)


def banner(models: list[str]) -> None:
    if is_simulated(models):
        print("=" * 78)
        print(" NOTE: using the offline MOCK provider. Results are SIMULATED and only")
        print(" demonstrate the harness. Rerun with --model openai/... or ollama/... ")
        print("=" * 78)


def save_json(name: str, data: dict[str, Any]) -> Path:
    path = OUT / f"{name}_results.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(title=title, height=height, template="plotly_white", margin=dict(l=50, r=20, t=50, b=50),
                      font=dict(family="system-ui, Segoe UI, sans-serif", size=12))
    return fig


def save_charts(name: str, title: str, figs: list[go.Figure], notes: list[str] | None = None, simulated: bool = False) -> Path:
    parts = [f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}) for f in figs]
    warn = ('<p style="background:#fff5e0;padding:10px;border-radius:6px"><b>Simulated data.</b> These results were '
            'produced by the offline mock provider and only demonstrate the harness.</p>') if simulated else ""
    note_html = "".join(f"<li>{n}</li>" for n in (notes or []))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="{PLOTLY_CDN}"></script>
<style>body{{font:14px/1.5 system-ui,Segoe UI,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2933}}
.fig{{border:1px solid #e4e7eb;border-radius:8px;margin:16px 0;padding:8px}}</style></head>
<body><h1>{title}</h1>{warn}<ul>{note_html}</ul>{''.join(f'<div class="fig">{p}</div>' for p in parts)}</body></html>"""
    path = OUT / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    return path


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def print_table(rows: list[list[Any]], headers: list[str]) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
