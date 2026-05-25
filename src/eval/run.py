"""Eval harness — runs the 12 needles × N models, writes a scoreboard.

INPUTS
  eval/needles.json          12 needle definitions (query_text, sql_filter, expected datasets)
  eval/ground_truth.json     hand-labeled correct frame_ids per needle
  LanceDB tables in data/lance/  discovered dynamically — model = table-name suffix

OUTPUTS
  eval/results.parquet       one row per (model, needle, k) — P@K, R@K, MRR, NDCG@K, coverage
  eval/results.html          rendered scoreboard for the dashboard

TABLE-NAME CONVENTION:
  SigLIP:   <dataset>            (waymo, bdd100k, nuscenes, av2)
  Others:   <dataset>_<model>    (waymo_clip, waymo_dinov2, waymo_cosmos_aug, waymo_lidar_bev)

CLI:
    python -m src.eval.run                 # run all needles × discovered models
    python -m src.eval.run --models siglip,clip --k 10
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.query.hybrid import query as hybrid_query

# Optional Datadog instrumentation (Win 2 Patch 3) — no-op if creds missing
try:
    from src.integrations.datadog_push import push_eval
except ImportError:
    push_eval = None

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


KNOWN_MODELS = ("siglip", "clip", "dinov2", "cosmos_aug", "cosmos_aug_r2", "cosmos_embed1", "lidar_bev")
DATASETS = ("waymo", "bdd100k", "nuscenes", "av2")


@dataclass
class EvalRow:
    model: str
    needle_id: str
    k: int
    precision_at_k: float | None
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    coverage: bool
    n_returned: int
    n_ground_truth: int
    elapsed_ms: float
    notes: str = ""


def _discover_tables(lance_dir: str) -> dict[str, list[str]]:
    """Return {model_name: [lancedb_table_name, ...]} from what's actually present."""
    import lancedb

    if not Path(lance_dir).exists():
        return {m: [] for m in KNOWN_MODELS}

    db = lancedb.connect(lance_dir)
    all_tables = set(db.table_names())
    out: dict[str, list[str]] = {m: [] for m in KNOWN_MODELS}

    for ds in DATASETS:
        # SigLIP = unsuffixed table
        if ds in all_tables:
            out["siglip"].append(ds)
        # Others = suffixed
        for model in KNOWN_MODELS:
            if model == "siglip":
                continue
            suffixed = f"{ds}_{model}"
            if suffixed in all_tables:
                out[model].append(suffixed)
    # cosmos_aug, cosmos_aug_r2, cosmos_embed1 ship as single cross-dataset tables
    # (see cosmos_aug_embed.py + cosmos_embed1_batch.py).
    for single in ("cosmos_aug", "cosmos_aug_r2", "cosmos_embed1"):
        if single in all_tables and not out[single]:
            out[single].append(single)
    return out


def _table_for_query(model: str, lance_tables: list[str]) -> str | None:
    """Pick which LanceDB table to query for this model.

    Strategy: if multiple datasets have tables for the model, hybrid.query()
    handles cross-dataset queries via its `datasets` arg + SQL filter routing.
    Here we just need ONE table that exists, since LanceDB tables ARE per-dataset
    in our convention; the cross-dataset join happens via the metadata parquet.
    Pick the first non-empty one.
    """
    return lance_tables[0] if lance_tables else None


def _precision_recall_at_k(returned: list[str], gt: set[str], k: int) -> tuple[float, float]:
    top_k = returned[:k]
    if not top_k:
        return (0.0, 0.0 if gt else None)
    hits = sum(1 for fid in top_k if fid in gt)
    precision = hits / len(top_k)
    recall = (hits / len(gt)) if gt else None
    return (precision, recall)


def _mrr(returned: list[str], gt: set[str]) -> float:
    for i, fid in enumerate(returned, 1):
        if fid in gt:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(returned: list[str], gt: set[str], k: int) -> float:
    """Binary-relevance NDCG@k. 1 if in gt, 0 otherwise."""
    if not gt:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(i + 2))
        for i, fid in enumerate(returned[:k])
        if fid in gt
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / ideal if ideal > 0 else 0.0


def _run_one(
    needle: dict,
    model: str,
    lance_table: str,
    gt: set[str],
    k: int,
    lance_dir: str,
    datasets_arg: tuple[str, ...],
) -> EvalRow:
    t0 = time.time()
    # cosmos_embed1 has its own corpus (250 Waymo video clips) with frame_ids that
    # are NOT in the Decision-15 metadata parquet. Skip SQL prefilter for it,
    # otherwise the candidate set is empty and we get zero hits regardless of
    # vector quality. This is corpus-isolation, not eval cheating.
    effective_sql = needle.get("sql_filter")
    if lance_table == "cosmos_embed1":
        effective_sql = None
    try:
        hits = hybrid_query(
            text=needle.get("query_text"),
            image_frame_id=needle.get("image_frame_id") if not needle.get("query_text") else None,
            sql_filter=effective_sql,
            table=lance_table,
            k=k,
            lance_dir=lance_dir,
            datasets=datasets_arg,
        )
        notes = "no-sql-filter (corpus disjoint from metadata)" if (effective_sql is None and needle.get("sql_filter")) else ""
    except Exception as e:
        hits = []
        notes = f"query failed: {type(e).__name__}: {e}"
    elapsed_ms = (time.time() - t0) * 1000

    returned_ids = [h.frame_id for h in hits]
    coverage = bool(returned_ids)

    if not gt:
        return EvalRow(
            model=model,
            needle_id=needle["needle_id"],
            k=k,
            precision_at_k=None,
            recall_at_k=None,
            mrr=None,
            ndcg_at_k=None,
            coverage=coverage,
            n_returned=len(returned_ids),
            n_ground_truth=0,
            elapsed_ms=elapsed_ms,
            notes=notes or "no ground truth yet (label this needle)",
        )

    p, r = _precision_recall_at_k(returned_ids, gt, k)
    return EvalRow(
        model=model,
        needle_id=needle["needle_id"],
        k=k,
        precision_at_k=p,
        recall_at_k=r,
        mrr=_mrr(returned_ids, gt),
        ndcg_at_k=_ndcg_at_k(returned_ids, gt, k),
        coverage=coverage,
        n_returned=len(returned_ids),
        n_ground_truth=len(gt),
        elapsed_ms=elapsed_ms,
        notes=notes,
    )


def _summary_table(rows: list[EvalRow]) -> Table:
    """Per-model aggregate scoreboard."""
    t = Table(title="Eval scoreboard — per-model means (NULL = unlabeled needles excluded)")
    t.add_column("model")
    t.add_column("needles", justify="right")
    t.add_column("labeled", justify="right")
    t.add_column("mean P@K", justify="right")
    t.add_column("mean R@K", justify="right")
    t.add_column("mean MRR", justify="right")
    t.add_column("mean NDCG@K", justify="right")
    t.add_column("coverage %", justify="right")
    t.add_column("mean ms", justify="right")

    by_model: dict[str, list[EvalRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    for model, mrows in sorted(by_model.items()):
        labeled = [r for r in mrows if r.precision_at_k is not None]
        n = len(mrows)
        nl = len(labeled)
        if labeled:
            mp = sum(r.precision_at_k for r in labeled) / nl
            mr = sum((r.recall_at_k or 0) for r in labeled) / nl
            mm = sum((r.mrr or 0) for r in labeled) / nl
            mn = sum((r.ndcg_at_k or 0) for r in labeled) / nl
        else:
            mp = mr = mm = mn = None
        cov = sum(1 for r in mrows if r.coverage) / n if n else 0
        avg_ms = sum(r.elapsed_ms for r in mrows) / n if n else 0
        t.add_row(
            model,
            str(n),
            str(nl),
            f"{mp:.3f}" if mp is not None else "—",
            f"{mr:.3f}" if mr is not None else "—",
            f"{mm:.3f}" if mm is not None else "—",
            f"{mn:.3f}" if mn is not None else "—",
            f"{100*cov:.0f}%",
            f"{avg_ms:.0f}",
        )
    return t


def _write_parquet(rows: list[EvalRow], path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([r.__dict__ for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _write_json(rows: list[EvalRow], path: Path):
    """Sidecar JSON for the dashboard (avoids DuckDB-WASM in browser)."""
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "rows": [r.__dict__ for r in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_html(rows: list[EvalRow], path: Path):
    """Minimal HTML scoreboard for the dashboard to embed."""
    by_needle: dict[str, dict[str, EvalRow]] = {}
    models = sorted({r.model for r in rows})
    for r in rows:
        by_needle.setdefault(r.needle_id, {})[r.model] = r

    head = """<!doctype html><meta charset=utf-8><title>Eval scoreboard</title>
<style>
  body { font: 14px/1.5 -apple-system, system-ui, sans-serif; color: #111; max-width: 1200px; margin: 24px auto; padding: 0 16px; }
  h1 { font-size: 22px; margin: 0 0 8px; }
  .meta { color: #666; margin-bottom: 16px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; }
  th { font-weight: 600; background: #f7f7f7; text-align: left; }
  td:first-child, th:first-child { text-align: left; font-family: ui-monospace, monospace; font-size: 13px; }
  .nope { color: #aaa; }
  .strong { color: #0a7; font-weight: 600; }
  .weak { color: #c40; }
</style>
"""
    lines = [head, "<h1>Eval scoreboard</h1>",
             f'<p class=meta>Generated {time.strftime("%Y-%m-%d %H:%M")} · {len(rows)} (model × needle) rows · models: {", ".join(models)}</p>',
             "<table><thead><tr><th>needle</th>"]
    for m in models:
        lines.append(f"<th>{m} P@K</th><th>{m} MRR</th>")
    lines.append("</tr></thead><tbody>")
    for nid in sorted(by_needle):
        lines.append(f"<tr><td>{nid}</td>")
        for m in models:
            r = by_needle[nid].get(m)
            if r is None or r.precision_at_k is None:
                lines.append('<td class=nope>—</td><td class=nope>—</td>')
            else:
                p_cls = "strong" if r.precision_at_k >= 0.5 else ("weak" if r.precision_at_k < 0.2 else "")
                lines.append(f'<td class="{p_cls}">{r.precision_at_k:.2f}</td><td>{(r.mrr or 0):.2f}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


@app.command()
def main(
    needles_path: Path = typer.Option(Path("eval/needles.json"), help="Needles definition."),
    ground_truth_path: Path = typer.Option(Path("eval/ground_truth.json"), help="Hand-labeled correct frame_ids."),
    out_parquet: Path = typer.Option(Path("eval/results.parquet"), help="Output parquet."),
    out_json: Path = typer.Option(Path("eval/results.json"), help="Output JSON sidecar (for dashboard)."),
    out_html: Path = typer.Option(Path("eval/results.html"), help="Output HTML scoreboard."),
    lance_dir: str = typer.Option("data/lance", help="LanceDB directory."),
    k: int = typer.Option(10, help="K for precision@K, recall@K, NDCG@K."),
    models: str = typer.Option(
        "", help="Comma-separated models to evaluate. Empty = all discovered."
    ),
    datasets: str = typer.Option(
        ",".join(DATASETS),
        help="Datasets to include in SQL prefilter routing.",
    ),
):
    """Run the eval harness, write parquet + HTML."""
    load_dotenv()

    needles = json.loads(needles_path.read_text())["needles"]
    gt_doc = json.loads(ground_truth_path.read_text())
    gt_map = {nid: set(fids) for nid, fids in gt_doc["ground_truth"].items()}

    tables_by_model = _discover_tables(lance_dir)
    requested = [m for m in (models.split(",") if models else KNOWN_MODELS) if m]
    active_models = [m for m in requested if tables_by_model.get(m)]
    datasets_tuple = tuple(d.strip() for d in datasets.split(",") if d.strip())

    console.rule("[bold]eval — plan")
    console.print(f"  needles: {len(needles)}")
    console.print(
        f"  labeled: {sum(1 for n in needles if gt_map.get(n['needle_id']))} / {len(needles)}"
    )
    console.print(f"  models requested: {requested}")
    console.print(f"  models with embeddings present: {active_models}")
    if not active_models:
        console.print(
            "[yellow]No LanceDB tables found. Run an embed pass first (e.g. `python -m src.embed.siglip_batch --table waymo`).[/yellow]"
        )
        console.print(
            "[dim]Writing an empty results.parquet + skeleton results.html so the dashboard renders.[/dim]"
        )
        _write_parquet([], out_parquet)
        _write_json([], out_json)
        _write_html([], out_html)
        return

    rows: list[EvalRow] = []
    for model in active_models:
        lance_tables = tables_by_model[model]
        table_name = _table_for_query(model, lance_tables)
        console.print(f"[bold]→ {model}[/bold] (LanceDB table: {table_name})")
        for n in needles:
            gt = gt_map.get(n["needle_id"], set())
            row = _run_one(
                needle=n,
                model=model,
                lance_table=table_name,
                gt=gt,
                k=k,
                lance_dir=lance_dir,
                datasets_arg=datasets_tuple,
            )
            rows.append(row)
            # Win 2 Patch 3 — push per-cell eval to Datadog (only if labeled)
            if push_eval and row.precision_at_k is not None:
                try:
                    push_eval(
                        model=row.model,
                        needle_id=row.needle_id,
                        precision_at_k=row.precision_at_k,
                        recall_at_10=row.recall_at_k,
                        mrr=row.mrr or 0.0,
                    )
                except Exception:
                    pass  # never let observability break the eval

    _write_parquet(rows, out_parquet)
    _write_json(rows, out_json)
    _write_html(rows, out_html)
    console.print(_summary_table(rows))
    console.print(f"\n[green]wrote {out_parquet} + {out_json} + {out_html}[/green]")


if __name__ == "__main__":
    app()
