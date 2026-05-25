"""Datadog metric pushers — keep the AV-Triage Pipeline dashboard live.

Three entry points used across the pipeline:

  push_spend(cost_json_path=Path("eval/cost.json"))
      Reads cost.json + pushes:
        avtri.pipeline.spend_usd            (gauge, total)
        avtri.pipeline.spend_usd            (gauge, per model tag)
        avtri.pipeline.spend_usd            (gauge, per phase tag)
        avtri.pipeline.budget_remaining_usd (gauge)

  push_embed_throughput(model, dataset, gpu_type, imgs_per_sec)
      Called from src/embed/{siglip,clip,dinov2,cosmos}_batch.py after each batch.

  push_query_latency(latency_ms, k, sql_filter_used)
      Called from src/query/hybrid.py at end of each query.

  push_eval(model, needle_id, precision_at_k, recall_at_10, mrr)
      Called from src/eval/run.py per (model, needle) row.

CLI:
  python -m src.integrations.datadog_push spend
  python -m src.integrations.datadog_push test       # bonus: ping a sentinel

Auth: SAME env vars as datadog_setup.py — source secrets/datadog.env first.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.integrations.datadog_setup import _configuration

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


_SERVICE_TAGS = ["service:physical-ai-triage", "env:demo"]


def _submit(series):
    """Submit a list of MetricSeries via the v2 API."""
    from datadog_api_client import ApiClient
    from datadog_api_client.v2.api.metrics_api import MetricsApi
    from datadog_api_client.v2.model.metric_payload import MetricPayload

    config, _ = _configuration()
    with ApiClient(config) as client:
        api = MetricsApi(client)
        api.submit_metrics(body=MetricPayload(series=series))


def _gauge(name: str, value: float, tags: list[str] | None = None):
    """Build one gauge MetricSeries — single point at now."""
    from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
    from datadog_api_client.v2.model.metric_point import MetricPoint
    from datadog_api_client.v2.model.metric_resource import MetricResource
    from datadog_api_client.v2.model.metric_series import MetricSeries

    return MetricSeries(
        metric=name,
        type=MetricIntakeType.GAUGE,
        points=[MetricPoint(timestamp=int(time.time()), value=float(value))],
        resources=[MetricResource(name="physical-ai-triage", type="host")],
        tags=_SERVICE_TAGS + (tags or []),
    )


def push_spend(cost_json_path: Path = Path("eval/cost.json")):
    """Read eval/cost.json + push spend + budget metrics."""
    if not cost_json_path.exists():
        console.print(f"[yellow]{cost_json_path} not found; skipping push_spend[/yellow]")
        return 0

    doc = json.loads(cost_json_path.read_text())
    total = doc.get("total_usd", 0)
    remaining = doc.get("remaining_usd")
    series = [_gauge("avtri.pipeline.spend_usd", total)]

    for model, usd in (doc.get("per_model_usd") or {}).items():
        if usd and float(usd) > 0:
            series.append(_gauge("avtri.pipeline.spend_usd", usd, tags=[f"model:{model}"]))

    for phase, usd in (doc.get("per_phase_usd") or {}).items():
        if usd and float(usd) > 0:
            series.append(_gauge("avtri.pipeline.spend_usd", usd, tags=[f"phase:{phase}"]))

    if remaining is not None:
        series.append(_gauge("avtri.pipeline.budget_remaining_usd", remaining))

    _submit(series)
    return len(series)


def push_embed_throughput(model: str, dataset: str, gpu_type: str, imgs_per_sec: float):
    """Pipeline-side: called from embed scripts after each batch (or end-of-run)."""
    _submit([
        _gauge(
            "avtri.embed.throughput_imgs_per_sec",
            imgs_per_sec,
            tags=[f"model:{model}", f"dataset:{dataset}", f"gpu_type:{gpu_type}"],
        )
    ])


def push_query_latency(latency_ms: float, k: int, sql_filter_used: bool):
    """Pipeline-side: called from hybrid.query() at end of each query."""
    _submit([
        _gauge(
            "avtri.query.latency_ms",
            latency_ms,
            tags=[f"k:{k}", f"sql_prefilter:{str(sql_filter_used).lower()}"],
        )
    ])


def push_eval(model: str, needle_id: str, precision_at_k: float, recall_at_10: float | None, mrr: float):
    """Pipeline-side: called from eval/run.py for each (model, needle) row."""
    tags = [f"model:{model}", f"needle:{needle_id}"]
    series = [_gauge("avtri.eval.precision_at_k", precision_at_k, tags=tags)]
    if recall_at_10 is not None:
        series.append(_gauge("avtri.eval.recall_at_10", recall_at_10, tags=tags))
    series.append(_gauge("avtri.eval.mrr", mrr, tags=tags))
    _submit(series)


@app.command()
def spend(cost_json: Path = typer.Option(Path("eval/cost.json"), help="cost.json path")):
    """CLI: push the current cost.json into Datadog."""
    load_dotenv()
    n = push_spend(cost_json)
    console.print(f"[green]pushed {n} spend metrics[/green]")


@app.command()
def test():
    """CLI: push a sentinel `avtri.test.push` = 1 to confirm wiring."""
    load_dotenv()
    _submit([_gauge("avtri.test.push", 1.0, tags=["source:datadog_push_cli"])])
    console.print("[green]sentinel submitted — check metric explorer[/green]")


if __name__ == "__main__":
    app()
