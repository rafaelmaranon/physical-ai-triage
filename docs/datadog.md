# Datadog — live integration

The pipeline emits structured metrics to **Datadog** (industry-standard cloud observability). A live dashboard renders embed throughput, query latency, eval recall, and pipeline spend in real time. Point the same hooks at your own org to render the same widgets.

> **Why this is in the architecture.** Pipeline observability is a real PM concern, not a checkbox. Showing live spend + recall@10 + p95 latency in one dashboard lets a stakeholder ask *"what did the last embed pass cost, and did it improve quality?"* and get an answer without opening four tools.

---

## Live demo dashboard

**URL:** _(your Datadog dashboard URL — point the metrics push at your own org)_

**Widgets:**

| Widget | Metric | Where it's pushed from |
|---|---|---|
| Pipeline cost | `avtri.pipeline.cost` | `src/integrations/datadog_push.py spend` reads `eval/cost.json` |
| Index size (MB) | `avtri.index.size_bytes / 1e6` | `src/index/build_hnsw.py` after each LanceDB build (hook TBD) |
| Mean recall@10 | `avg(avtri.eval.recall_at_10)` | `src/eval/run.py` per (model, needle) |
| Embed throughput (img/s) | `avtri.embed.throughput_imgs_per_sec` | `src/embed/{siglip,clip,dinov2}_batch.py` after each batch |
| Query latency p50/p95/p99 (ms) | `percentile(avtri.query.latency_ms)` | `src/query/hybrid.py` at end of `query()` |
| Embed throughput by model | tagged version of throughput metric | same, tagged `model:siglip` etc. |

---

## Three-command setup

```bash
# 1. Credentials live in secrets/datadog.env (sourced, never committed):
set -a; source secrets/datadog.env; set +a

# 2. Confirm auth works
python -m src.integrations.datadog_setup verify

# 3. Create the dashboard (idempotent — replaces any existing "AV-Triage Pipeline")
python -m src.integrations.datadog_setup dashboard
```

Optional first-light test:

```bash
python -m src.integrations.datadog_setup push-test    # sentinel avtri.test.startup=1
python -m src.integrations.datadog_push spend          # push real spend from eval/cost.json
```

---

## Metric schema

All metrics are gauges, all tagged with `service:physical-ai-triage` and `env:demo`. Additional tags per metric:

| Metric | Type | Per-metric tags |
|---|---|---|
| `avtri.pipeline.spend_usd` | gauge | optional `model:<name>`, `phase:<name>` |
| `avtri.pipeline.budget_remaining_usd` | gauge | — |
| `avtri.embed.throughput_imgs_per_sec` | gauge | `model:<name>`, `dataset:<name>`, `gpu_type:<l4\|a100>` |
| `avtri.embed.batch_latency_ms` | histogram | `model:<name>` |
| `avtri.query.latency_ms` | gauge | `k:<int>`, `sql_prefilter:<true\|false>` |
| `avtri.query.candidates_after_prefilter` | gauge | — |
| `avtri.index.size_bytes` | gauge | `table:<lance_table>` |
| `avtri.index.row_count` | gauge | `table:<lance_table>` |
| `avtri.eval.precision_at_k` | gauge | `model:<name>`, `needle:<id>` |
| `avtri.eval.recall_at_10` | gauge | `model:<name>`, `needle:<id>` |
| `avtri.eval.mrr` | gauge | `model:<name>`, `needle:<id>` |

---

## Instrumentation hooks

The pipeline calls four module-level functions from `src/integrations/datadog_push.py`:

```python
from src.integrations.datadog_push import (
    push_spend,
    push_embed_throughput,
    push_query_latency,
    push_eval,
)

# After each embed batch:
push_embed_throughput(model="siglip", dataset="waymo", gpu_type="l4", imgs_per_sec=205.3)

# At the end of hybrid.query():
push_query_latency(latency_ms=elapsed, k=10, sql_filter_used=True)

# Per (model, needle) row in eval/run.py:
push_eval(model="siglip", needle_id="n01_unprotected_left_pedestrian",
          precision_at_k=0.6, recall_at_10=0.4, mrr=0.5)

# After eval/cost.json updates:
push_spend()
```

If `DATADOG_API_KEY` isn't set, the calls no-op gracefully — pipeline never blocks on observability.

---

## Cost

- Custom metric volume is well under cheap-plan thresholds for this workload.
- Configurable per org plan.

---

## Substitution paths

Same metric schema works on any modern observability stack — only the auth + the SDK changes:

| Backend | What to swap |
|---|---|
| **Prometheus + Grafana** | Replace `_submit()` with `prometheus_client.Gauge` writes; scrape via Pushgateway or `/metrics` |
| **OpenTelemetry** | Replace with OTel `Meter.create_gauge()` + OTLP exporter |
| **CloudWatch** | Replace with `boto3.client('cloudwatch').put_metric_data(...)` |
| **SigNoz** | Same OpenTelemetry path — SigNoz is OTel-native |
| **Honeycomb** | OTel + Honeycomb exporter |

All five would render the same widgets (spend, latency, throughput, recall) without changing any pipeline code — only `src/integrations/datadog_push.py` is backend-specific.

---

## Why Datadog (and not the others)?

For this demo:

1. **Industry-standard observability** for the robotics/AV data infra space, sits cleanly alongside AWS + Snowflake.
2. **Free-tier covered the build window.**
3. **Dashboard JSON API** — let me create the entire 6-widget dashboard in one Python call. Most alternatives require clicking through a UI to set up viz.
4. **Native fit with operational stacks** — teams that already pay for Datadog get pipeline metrics into their existing observability surface with zero extra work.

For production, the substitution paths above are all reasonable; the choice is governance + existing-org-spend, not technical.

---

## Refresh

Pipeline runs push metrics live. No periodic job needed — every embed batch, every query, every eval row writes its own gauge.

For backfilling historical cost data:

```bash
python -m src.integrations.datadog_push spend    # idempotent, always pushes current eval/cost.json
```

Re-running this is cheap (~5 metric points) and the dashboard tile will show the latest value.
