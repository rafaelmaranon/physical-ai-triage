"""Datadog live integration — push pipeline metrics + create a starter dashboard.

(upgraded from stub to live, 2026-05-21).

PIPELINE METRICS (planned — instrumentation hooks land in src/embed/* + src/query/*):
  avtri.embed.throughput_imgs_per_sec   gauge per (model, dataset, gpu_type)
  avtri.embed.batch_latency_ms          histogram per model
  avtri.query.latency_ms                histogram per (sql_filter_used, k)
  avtri.query.candidates_after_prefilter gauge
  avtri.index.size_bytes                gauge per LanceDB table
  avtri.index.row_count                 gauge per LanceDB table
  avtri.eval.precision_at_k             gauge per (model, needle)
  avtri.eval.recall_at_10               gauge per (model, needle)
  avtri.pipeline.spend_usd              gauge

THREE COMMANDS:

  python -m src.integrations.datadog_setup verify
    Confirm API+App keys work, list orgs/dashboards I can see.

  python -m src.integrations.datadog_setup push-test
    Submit a test metric `avtri.test.startup=1` so we can validate end-to-end
    plumbing before any real pipeline run.

  python -m src.integrations.datadog_setup dashboard
    Create (or replace) the "AV-Triage Pipeline" dashboard with 6 widgets:
    embed throughput, query latency, index size, eval recall, pipeline spend,
    plus a Pipeline status note. Saves the dashboard URL to docs/datadog_dashboard.txt.

CREDENTIALS (set in secrets/datadog.env; source before running):
  DATADOG_API_KEY     32-char hex from Org Settings → API Keys
  DATADOG_APP_KEY     ddapp_... from Personal Settings → Application Keys
  DATADOG_SITE        us3.datadoghq.com (or us1/us5/eu)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _configuration():
    """Build a datadog_api_client Configuration from env vars."""
    try:
        from datadog_api_client import Configuration
    except ImportError:
        console.print("[red]datadog-api-client not installed.[/red] Run `uv sync`.")
        raise typer.Exit(1)

    load_dotenv()
    api_key = os.environ.get("DATADOG_API_KEY") or os.environ.get("DD_API_KEY")
    app_key = os.environ.get("DATADOG_APP_KEY") or os.environ.get("DD_APP_KEY")
    site = os.environ.get("DATADOG_SITE") or os.environ.get("DD_SITE", "datadoghq.com")

    if not api_key:
        console.print("[red]DATADOG_API_KEY missing.[/red] Source secrets/datadog.env first.")
        raise typer.Exit(2)
    if not app_key:
        console.print(
            "[yellow]DATADOG_APP_KEY missing — verify will work, dashboard create will fail.[/yellow]"
        )

    config = Configuration()
    config.server_variables["site"] = site
    config.api_key["apiKeyAuth"] = api_key
    if app_key:
        config.api_key["appKeyAuth"] = app_key
    return config, site


@app.command()
def verify():
    """Confirm credentials work; list dashboards visible to this user."""
    from datadog_api_client import ApiClient
    from datadog_api_client.v1.api.authentication_api import AuthenticationApi
    from datadog_api_client.v1.api.dashboards_api import DashboardsApi

    config, site = _configuration()
    console.rule(f"[bold]Datadog verify @ {site}")

    with ApiClient(config) as client:
        auth = AuthenticationApi(client)
        r = auth.validate()
        ok = bool(getattr(r, "valid", False))
        console.print(f"  API key validate: [{'green' if ok else 'red'}]{'OK' if ok else 'FAIL'}[/]")
        if not ok:
            raise typer.Exit(3)

        try:
            dash = DashboardsApi(client)
            res = dash.list_dashboards()
            dashboards = list(getattr(res, "dashboards", []))
            console.print(f"  dashboards visible: {len(dashboards)}")
            for d in dashboards[:5]:
                console.print(f"    · {d.title} ({d.id})")
        except Exception as e:
            console.print(f"  [yellow]dashboard list failed (app key issue?): {e}[/yellow]")


@app.command()
def push_test():
    """Submit a single test metric so we can confirm end-to-end plumbing."""
    from datadog_api_client import ApiClient
    from datadog_api_client.v2.api.metrics_api import MetricsApi
    from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
    from datadog_api_client.v2.model.metric_payload import MetricPayload
    from datadog_api_client.v2.model.metric_point import MetricPoint
    from datadog_api_client.v2.model.metric_resource import MetricResource
    from datadog_api_client.v2.model.metric_series import MetricSeries

    config, site = _configuration()
    console.rule(f"[bold]Datadog push-test @ {site}")

    body = MetricPayload(
        series=[
            MetricSeries(
                metric="avtri.test.startup",
                type=MetricIntakeType.GAUGE,
                points=[MetricPoint(timestamp=int(time.time()), value=1.0)],
                resources=[MetricResource(name="physical-ai-triage", type="host")],
                tags=["service:physical-ai-triage", "env:demo", "by:cc2_setup"],
            )
        ]
    )

    with ApiClient(config) as client:
        api = MetricsApi(client)
        resp = api.submit_metrics(body=body)
        console.print(f"  submitted: {resp}")
        console.print(
            "  → metric `avtri.test.startup` should appear in Datadog within ~30 sec "
            "at https://{site}/metric/explorer".format(site=site)
        )


@app.command()
def dashboard(
    out: Path = typer.Option(
        Path("docs/datadog_dashboard.txt"), help="Save dashboard URL + JSON here."
    ),
):
    """Create (or replace) the AV-Triage Pipeline dashboard."""
    from datadog_api_client import ApiClient
    from datadog_api_client.v1.api.dashboards_api import DashboardsApi
    from datadog_api_client.v1.model.dashboard import Dashboard
    from datadog_api_client.v1.model.dashboard_layout_type import DashboardLayoutType
    from datadog_api_client.v1.model.widget import Widget
    from datadog_api_client.v1.model.widget_definition import WidgetDefinition
    from datadog_api_client.v1.model.timeseries_widget_definition import (
        TimeseriesWidgetDefinition,
    )
    from datadog_api_client.v1.model.timeseries_widget_definition_type import (
        TimeseriesWidgetDefinitionType,
    )
    from datadog_api_client.v1.model.timeseries_widget_request import TimeseriesWidgetRequest
    from datadog_api_client.v1.model.query_value_widget_definition import (
        QueryValueWidgetDefinition,
    )
    from datadog_api_client.v1.model.query_value_widget_definition_type import (
        QueryValueWidgetDefinitionType,
    )
    from datadog_api_client.v1.model.query_value_widget_request import QueryValueWidgetRequest
    from datadog_api_client.v1.model.widget_layout import WidgetLayout
    from datadog_api_client.v1.model.note_widget_definition import NoteWidgetDefinition
    from datadog_api_client.v1.model.note_widget_definition_type import (
        NoteWidgetDefinitionType,
    )

    config, site = _configuration()
    title = "AV-Triage Pipeline"
    console.rule(f"[bold]Datadog dashboard '{title}' @ {site}")

    def ts_widget(label, query, x, y, w=6, h=2):
        return Widget(
            definition=TimeseriesWidgetDefinition(
                type=TimeseriesWidgetDefinitionType.TIMESERIES,
                title=label,
                requests=[TimeseriesWidgetRequest(q=query, display_type="line")],
            ),
            layout=WidgetLayout(x=x, y=y, width=w, height=h),
        )

    def qv_widget(label, query, x, y, w=3, h=2):
        return Widget(
            definition=QueryValueWidgetDefinition(
                type=QueryValueWidgetDefinitionType.QUERY_VALUE,
                title=label,
                requests=[QueryValueWidgetRequest(q=query)],
                precision=2,
            ),
            layout=WidgetLayout(x=x, y=y, width=w, height=h),
        )

    note = Widget(
        definition=NoteWidgetDefinition(
            type=NoteWidgetDefinitionType.NOTE,
            content=(
                "**AV-Triage Pipeline** · semantic + similarity search over "
                "autonomous-driving data. Metrics emit from the embedding + query "
                "pipeline (see src/integrations/datadog_setup.py). "
                "Code: github.com/rafamaraw/physical-ai-triage"
            ),
            background_color="vivid_purple",
            font_size="14",
            show_tick=False,
        ),
        layout=WidgetLayout(x=0, y=0, width=12, height=1),
    )

    widgets = [
        note,
        qv_widget(
            "Pipeline spend ($)",
            "sum:avtri.pipeline.spend_usd{*}",
            x=0, y=1, w=3, h=2,
        ),
        qv_widget(
            "Index size (MB)",
            "sum:avtri.index.size_bytes{*} / 1000000",
            x=3, y=1, w=3, h=2,
        ),
        qv_widget(
            "Mean recall@10",
            "avg:avtri.eval.recall_at_10{*}",
            x=6, y=1, w=3, h=2,
        ),
        qv_widget(
            "Embed throughput (img/s)",
            "avg:avtri.embed.throughput_imgs_per_sec{*}",
            x=9, y=1, w=3, h=2,
        ),
        ts_widget(
            "Query latency p50/p95/p99 (ms)",
            "p50:avtri.query.latency_ms{*}, p95:avtri.query.latency_ms{*}, p99:avtri.query.latency_ms{*}",
            x=0, y=3, w=6, h=3,
        ),
        ts_widget(
            "Embed throughput by model",
            "avg:avtri.embed.throughput_imgs_per_sec{*} by {model}",
            x=6, y=3, w=6, h=3,
        ),
    ]

    body = Dashboard(
        title=title,
        layout_type=DashboardLayoutType.ORDERED,
        description="Live metrics for the physical-ai-triage semantic search pipeline.",
        widgets=widgets,
    )

    with ApiClient(config) as client:
        api = DashboardsApi(client)

        # Idempotent: delete any prior dashboard with this title
        try:
            existing = api.list_dashboards()
            for d in getattr(existing, "dashboards", []) or []:
                if d.title == title:
                    console.print(f"  removing existing dashboard {d.id}")
                    api.delete_dashboard(d.id)
        except Exception as e:
            console.print(f"  [yellow]could not list/delete prior dashboards: {e}[/yellow]")

        res = api.create_dashboard(body=body)
        url = f"https://{site}/dashboard/{res.id}"
        console.print(f"  ✅ created {res.id}")
        console.print(f"  ✅ url: {url}")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"Dashboard: {title}\nURL: {url}\nID: {res.id}\nCreated: {time.strftime('%Y-%m-%d %H:%M')}\n"
        )


if __name__ == "__main__":
    app()
