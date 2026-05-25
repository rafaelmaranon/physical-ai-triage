"""Hybrid query — SQL prefilter on parquet metadata × vector ANN on embeddings.

The product layer. #15, #17, #18:
  1. SQL pre-filter over the Decision-15 metadata parquet (DuckDB + httpfs).
     Filters on ego/perception/scene columns yield a candidate `frame_id` set.
  2. Vector ANN over the candidates in LanceDB (one of: embeddings_rgb,
     embeddings_lidar_bev, or a per-model table from the bake-off).
  3. Top-K returned as (frame_id, score, dataset, device_id, ts_ns, thumbnail_uri).
  4. NULL handling per Decision 17 — predicates on NULL columns auto-exclude rows
     from datasets that don't have that data, so the same query runs across all 4.

CLI:
    python -m src.query.hybrid "jaywalker at night in SF" \\
        --sql "city = 'sf' AND time_of_day = 'night'" \\
        --table embeddings_rgb \\
        --k 10

Programmatic:
    from src.query.hybrid import query
    results = query(
        text="jaywalker at night in SF",
        sql_filter="city = 'sf' AND time_of_day = 'night'",
        table="embeddings_rgb",
        k=10,
    )
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.cloud import duckdb_with_s3, join as uri_join

# Optional Datadog instrumentation — no-op if creds missing (Win 2 Patch 2)
try:
    from src.integrations.datadog_push import push_query_latency
except ImportError:
    push_query_latency = None

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# Dataset → BUCKET_URI prefix env var (mirrors src/embed/siglip_batch.py).
_PREFIX_ENV = {
    "waymo": "WAYMO_PREFIX",
    "bdd100k": "BDD100K_PREFIX",
    "nuscenes": "NUSCENES_PREFIX",
    "av2": "AV2_PREFIX",
}


@dataclass
class Hit:
    frame_id: str
    score: float
    dataset: str
    device_id: str
    ts_ns: int
    camera_name: str
    thumbnail_uri: str
    # Optional context columns that may have helped a rank decision
    city: str | None = None
    time_of_day: str | None = None
    weather: str | None = None
    num_pedestrians: int | None = None
    ego_speed_mps: float | None = None


def _all_metadata_uris(bucket_uri: str, datasets: Iterable[str]) -> list[str]:
    """Return one parquet glob URI per dataset whose metadata exists."""
    uris = []
    for ds in datasets:
        prefix_env = _PREFIX_ENV.get(ds)
        if not prefix_env:
            continue
        prefix = os.environ.get(prefix_env, ds)
        uris.append(uri_join(bucket_uri, prefix, "metadata", "*.parquet"))
    return uris


def _candidate_frame_ids(
    con,
    metadata_uris: list[str],
    sql_filter: str | None,
) -> set[str] | None:
    """Run the SQL prefilter. Returns None if no filter (= full corpus is the candidate set).

    []: filter `metadata_uris` to only those that actually exist on S3.
    Previously hardcoded 4 datasets (waymo/bdd100k/nuscenes/av2) but we only have 2 in v1 —
    UNION ALL BY NAMEfailed on missing nuscenes/av2 paths. Now probe via fsspec first.
    """
    if not sql_filter:
        return None
    # Probe which URIs exist (avoids DuckDB "No files found" error on missing datasets)
    from src.cloud import get_fs
    existing = []
    for uri in metadata_uris:
        try:
            fs, path = get_fs(uri.replace("/*.parquet", "").replace("*.parquet", ""))
            # Strip glob to a directory-level check
            base = path.rstrip("/").rsplit("/", 1)[0] if "*" in path else path
            if fs.exists(base) or any(p.endswith(".parquet") for p in fs.glob(uri.replace(uri.split("/")[0] + "://", ""))):
                existing.append(uri)
        except Exception:
            # Be forgiving — if we can't tell, try it (DuckDB will skip cleanly with a glob)
            existing.append(uri)
    if not existing:
        return set()
    union_sql = " UNION ALL BY NAME ".join(
        f"SELECT * FROM read_parquet('{uri}')" for uri in existing
    )
    sql = (
        f"WITH meta AS ({union_sql}) "
        f"SELECT frame_id FROM meta WHERE {sql_filter}"
    )
    rows = con.execute(sql).fetchall()
    return {r[0] for r in rows}


def _siglip_text_encode(text: str, model_name: str | None = None):
    """Encode `text` with SigLIP text encoder → np.ndarray (768,) float16."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    name = model_name or os.environ.get("EMBED_MODEL", "google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb.squeeze(0).to(torch.float16).cpu().numpy()


def _cosmos_embed1_text_encode(text: str, model_name: str = "nvidia/Cosmos-Embed1-336p"):
    """Encode `text` with Cosmos Embed1 text encoder → np.ndarray (768,) float16.

    Used when querying the `cosmos_embed1` LanceDB table — must use Embed1's
    own text encoder for the cross-modal cosine search to be meaningful.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # fp32 to match cosmos_embed1_batch.py (device_map="auto" leaves layer_norms in fp32)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
    text_inputs = processor(text=[text])
    with torch.no_grad():
        out = model.get_text_embeddings(**text_inputs)
    # Extract tensor from TextEmbedderOutput dataclass (same pattern as video side)
    emb_tensor = None
    for attr in ("text_embeds", "embedding", "embeddings", "last_hidden_state", "pooler_output"):
        if hasattr(out, attr):
            cand = getattr(out, attr)
            if hasattr(cand, "squeeze"):
                emb_tensor = cand
                break
    if emb_tensor is None:
        for v in (out.__dict__.values() if hasattr(out, "__dict__") else []):
            if hasattr(v, "squeeze"):
                emb_tensor = v
                break
    if emb_tensor is None:
        raise RuntimeError(f"Cannot find text embedding tensor in {type(out).__name__}")
    emb = emb_tensor.squeeze(0).to(torch.float16).cpu().numpy()
    n = float(np.linalg.norm(emb)) or 1.0
    return (emb / n).astype(np.float16)


def _text_encoder_for_table(table: str):
    """Pick the right text encoder based on which LanceDB table we're querying."""
    if table.startswith("cosmos_embed1"):
        return _cosmos_embed1_text_encode
    return _siglip_text_encode


def query(
    text: str | None = None,
    image_frame_id: str | None = None,
    sql_filter: str | None = None,
    table: str = "embeddings_rgb",
    k: int = 10,
    lance_dir: str = "data/lance",
    datasets: Iterable[str] = ("waymo", "bdd100k", "nuscenes", "av2"),
    model_name: str | None = None,
) -> list[Hit]:
    """Run a hybrid query and return ranked Hits.

    Exactly one of `text` or `image_frame_id` must be provided.
    `image_frame_id` does an image→image search by reusing the stored embedding.
    """
    _t0 = time.time()  # Win 2 Patch 2 — time the whole query for Datadog push
    load_dotenv()
    if (text is None) == (image_frame_id is None):
        raise ValueError("Provide exactly one of `text` or `image_frame_id`.")

    import duckdb
    import lancedb

    bucket_uri = os.environ.get("BUCKET_URI")
    if not bucket_uri:
        raise RuntimeError("BUCKET_URI not set — copy .env.example to .env first.")

    metadata_uris = _all_metadata_uris(bucket_uri, datasets)
    con = duckdb_with_s3(duckdb.connect())

    # 1. SQL prefilter
    candidates = _candidate_frame_ids(con, metadata_uris, sql_filter)

    # 2. Vector ANN
    db = lancedb.connect(lance_dir)
    # Check existence via filesystem — db.table_names() is deprecated and returns stale cache
    table_path = Path(lance_dir) / f"{table}.lance"
    if not table_path.is_dir():
        raise RuntimeError(
            f"LanceDB table {table!r} missing in {lance_dir}. "
            f"Run an embed pass first (e.g. `python -m src.embed.siglip_batch --table waymo`)."
        )
    tbl = db.open_table(table)

    if text is not None:
        encoder = _text_encoder_for_table(table)
        query_vec = encoder(text) if encoder is _cosmos_embed1_text_encode else encoder(text, model_name=model_name)
    else:
        row = tbl.search().where(f"frame_id = '{image_frame_id}'").limit(1).to_list()
        if not row:
            raise RuntimeError(f"frame_id {image_frame_id!r} not found in {table}.")
        query_vec = row[0]["embedding"]

    search = tbl.search(query_vec)
    if candidates is not None:
        if not candidates:
            return []
        # LanceDB SQL-like WHERE supports IN; for large candidate sets, prefilter
        # by sampling 10K (LanceDB's WHERE clause has a practical size limit).
        sample = list(candidates)[:10000]
        in_list = ",".join(f"'{f}'" for f in sample)
        search = search.where(f"frame_id IN ({in_list})")
    raw = search.limit(k).to_list()

    # 3. Optional: enrich top-K with context columns from the metadata parquet
    if raw:
        fid_list = ",".join(f"'{r['frame_id']}'" for r in raw)
        union_sql = " UNION ALL BY NAME ".join(
            f"SELECT frame_id, city, time_of_day, weather, num_pedestrians, ego_speed_mps "
            f"FROM read_parquet('{uri}')"
            for uri in metadata_uris
        )
        enrich_sql = (
            f"WITH meta AS ({union_sql}) "
            f"SELECT * FROM meta WHERE frame_id IN ({fid_list})"
        )
        try:
            enrich = {
                row[0]: dict(
                    zip(
                        ("city", "time_of_day", "weather", "num_pedestrians", "ego_speed_mps"),
                        row[1:],
                        strict=True,
                    )
                )
                for row in con.execute(enrich_sql).fetchall()
            }
        except Exception:
            # Metadata parquet may not have all columns yet (e.g. while topic_joins is unwired)
            enrich = {}
    else:
        enrich = {}

    hits = []
    for r in raw:
        meta = enrich.get(r["frame_id"], {})
        # LanceDB default metric is L2-squared (not cosine) when table is created without
        # metric="cosine". For un-normalized 768d/1024d embeddings the raw distance is in
        # [0, ~1000]. Map to similarity in (0, 1] via 1/(1+d) so screenshots show clean scores.
        # 
        _dist = float(r.get("_distance", 0))
        score = 1.0 / (1.0 + _dist) if _dist >= 0 else 1.0
        hits.append(
            Hit(
                frame_id=r["frame_id"],
                score=score,
                dataset=r["dataset"],
                device_id=r["device_id"],
                ts_ns=int(r["ts_ns"]),
                camera_name=r["camera_name"],
                thumbnail_uri=r["thumbnail_uri"],
                city=meta.get("city"),
                time_of_day=meta.get("time_of_day"),
                weather=meta.get("weather"),
                num_pedestrians=meta.get("num_pedestrians"),
                ego_speed_mps=meta.get("ego_speed_mps"),
            )
        )

    # Win 2 Patch 2 — emit query latency to Datadog (no-op if creds absent)
    if push_query_latency:
        try:
            push_query_latency(
                latency_ms=(time.time() - _t0) * 1000,
                k=k,
                sql_filter_used=bool(sql_filter),
            )
        except Exception:
            pass  # never let observability break a query

    return hits


@app.command()
def main(
    text: str = typer.Argument(None, help="Text query (e.g. 'jaywalker at night in SF')."),
    image: str = typer.Option(None, "--image", help="Image→image: frame_id whose stored embedding to query with."),
    sql: str = typer.Option(None, "--sql", help="SQL prefilter, e.g. \"city='sf' AND time_of_day='night'\"."),
    table: str = typer.Option("embeddings_rgb", help="LanceDB table to search."),
    k: int = typer.Option(10, help="Top-K to return."),
    lance_dir: str = typer.Option("data/lance", help="Local LanceDB directory."),
):
    """CLI: run one hybrid query, print top-K to terminal."""
    if not text and not image:
        console.print("[red]Provide either a TEXT query or --image FRAME_ID.[/red]")
        raise typer.Exit(2)

    t0 = time.time()
    hits = query(
        text=text,
        image_frame_id=image,
        sql_filter=sql,
        table=table,
        k=k,
        lance_dir=lance_dir,
    )
    elapsed_ms = (time.time() - t0) * 1000

    tag = "text" if text else "image"
    console.rule(f"[bold]{tag}-query → {table} (k={k}, {elapsed_ms:.0f} ms)")
    if sql:
        console.print(f"[dim]SQL prefilter: {sql}[/dim]")
    if not hits:
        console.print("[yellow]No hits.[/yellow]")
        return

    t = Table()
    t.add_column("rank", justify="right")
    t.add_column("score", justify="right")
    t.add_column("dataset")
    t.add_column("city")
    t.add_column("tod")
    t.add_column("weather")
    t.add_column("peds", justify="right")
    t.add_column("device_id (head)")
    t.add_column("ts_ns")
    for i, h in enumerate(hits, 1):
        t.add_row(
            str(i),
            f"{h.score:.3f}",
            h.dataset,
            h.city or "-",
            h.time_of_day or "-",
            h.weather or "-",
            str(h.num_pedestrians) if h.num_pedestrians is not None else "-",
            (h.device_id or "")[:16],
            str(h.ts_ns),
        )
    console.print(t)


if __name__ == "__main__":
    app()
