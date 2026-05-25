"""SigLIP batch embedder — reads parquet metadata + thumbnails from BUCKET_URI.

#6, #7:
  - Reads frame metadata from `{BUCKET_URI}/{dataset_prefix}/metadata/*.parquet` via DuckDB.
  - Pulls thumbnails from `{BUCKET_URI}/{THUMBNAILS_PREFIX}/<dataset>/<frame_id>.jpg` via fsspec.
  - Writes float16 embeddings to LanceDB on the LOCAL Mac (Decision 7).
  - Resumable: skips frame_ids already present in the LanceDB table.

Designed for the Brev A100 host. Smoke-test on a laptop with
`EMBED_DEVICE=cpu` and `--limit 50` to confirm wiring before paying for GPU time.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from src.cloud import duckdb_with_s3, get_fs, join as uri_join

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _lazy_imports():
    try:
        import duckdb  # noqa: F401
        import lancedb  # noqa: F401
        import numpy as np  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        from transformers import AutoModel, AutoProcessor  # noqa: F401
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e.name}[/red]. Run `uv sync`.")
        raise typer.Exit(1) from e


def _pick_device(requested: str) -> str:
    import torch
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cpu":
        return "cpu"
    console.print(f"[yellow]EMBED_DEVICE={requested} unavailable, falling back to cpu[/yellow]")
    return "cpu"


# Map a logical table name → BUCKET_URI prefix env var.
_PREFIX_ENV = {
    "waymo": "WAYMO_PREFIX",
    "bdd100k": "BDD100K_PREFIX",
    "nuscenes": "NUSCENES_PREFIX",
    "av2": "AV2_PREFIX",
}


@app.command()
def main(
    table: str = typer.Option(..., help="LanceDB table name (waymo, bdd100k, nuscenes, av2)."),
    metadata_uri: str = typer.Option(
        None,
        help="Override parquet URI. Default: {BUCKET_URI}/{<table>_PREFIX}/metadata/*.parquet.",
    ),
    limit: int = typer.Option(0, help="Stop after this many frames (0 = no limit)."),
    batch: int = typer.Option(64, help="Batch size."),
    model_name: str = typer.Option(None, help="Override EMBED_MODEL."),
    device: str = typer.Option(None, help="Override EMBED_DEVICE."),
    lance_dir: str = typer.Option("data/lance", help="Local LanceDB directory."),
):
    """Embed frames into the LanceDB table named `--table`."""
    load_dotenv()
    _lazy_imports()

    import duckdb
    import lancedb
    import pyarrow as pa
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    bucket_uri = os.environ.get("BUCKET_URI")
    if not bucket_uri and not metadata_uri:
        console.print("[red]BUCKET_URI not set and --metadata-uri not provided.[/red]")
        raise typer.Exit(2)

    if metadata_uri is None:
        prefix_env = _PREFIX_ENV.get(table)
        if not prefix_env:
            console.print(f"[red]Unknown table '{table}' and no --metadata-uri given.[/red]")
            raise typer.Exit(3)
        prefix = os.environ.get(prefix_env, table)
        metadata_uri = uri_join(bucket_uri, prefix, "metadata", "*.parquet")

    model_name = model_name or os.environ.get("EMBED_MODEL", "google/siglip-base-patch16-224")
    device = _pick_device(device or os.environ.get("EMBED_DEVICE", "cuda"))
    dtype = torch.float16 if device == "cuda" else torch.float32

    console.rule(f"[bold]embed → {table}")
    console.print(f"  model:        {model_name}")
    console.print(f"  device:       {device}")
    console.print(f"  batch:        {batch}")
    console.print(f"  metadata_uri: {metadata_uri}")
    console.print(f"  lance:        {lance_dir}/{table}")

    # 1. Load model
    console.print("[dim]loading model...[/dim]")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.to(dtype=dtype)
    embed_dim = model.config.vision_config.hidden_size

    # 2. Open LanceDB + figure out which frame_ids are already done
    Path(lance_dir).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lance_dir)
    # Schema columns mirror Decision 15 metadata keys + the float16 embedding.
    schema = pa.schema(
        [
            pa.field("frame_id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("device_id", pa.string()),
            pa.field("ts_ns", pa.int64()),
            pa.field("camera_name", pa.string()),
            pa.field("thumbnail_uri", pa.string()),
            pa.field("embedding", pa.list_(pa.float16(), embed_dim)),
        ]
    )
    if table in db.table_names():
        tbl = db.open_table(table)
        existing = {
            r["frame_id"]
            for r in tbl.search().select(["frame_id"]).limit(10**9).to_list()
        }
        console.print(f"  resuming: {len(existing):,} frames already embedded")
    else:
        tbl = db.create_table(table, schema=schema, mode="create")
        existing = set()

    # 3. Pull metadata rows via DuckDB (reads s3:// directly via httpfs).
    # Columns per Decision 15 schema.
    con = duckdb_with_s3(duckdb.connect())
    sql = (
        "SELECT frame_id, dataset, device_id, ts_ns, camera_name, thumbnail_uri "
        f"FROM read_parquet('{metadata_uri}')"
    )
    if limit:
        sql += f" LIMIT {limit}"
    rows = con.execute(sql).fetchall()
    cols = ("frame_id", "dataset", "device_id", "ts_ns", "camera_name", "thumbnail_uri")
    rows = [dict(zip(cols, r, strict=True)) for r in rows]

    todo = [r for r in rows if r["frame_id"] not in existing]
    console.print(f"  pending:  {len(todo):,} frames")

    if not todo:
        console.print("[green]Nothing to do.[/green]")
        return

    # 4. Frame loader — fsspec picks backend by URI scheme
    def load_image(row) -> Image.Image | None:
        uri = row["thumbnail_uri"]
        if not uri:
            return None
        try:
            fs, path = get_fs(uri)
            with fs.open(path, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception as e:
            console.print(f"[yellow]skip {row['frame_id']}: {e}[/yellow]")
            return None

    # 5. Batch loop
    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    committed = 0
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("embedding", total=len(todo))
        for chunk in batched(todo, batch):
            pairs = [(r, load_image(r)) for r in chunk]
            valid = [(r, im) for r, im in pairs if im is not None]
            if not valid:
                progress.advance(task, len(chunk))
                continue
            inputs = processor(images=[im for _, im in valid], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            emb = emb.to(torch.float16).cpu().numpy()
            recs = []
            for (r, _), vec in zip(valid, emb, strict=True):
                recs.append(
                    {
                        "frame_id": r["frame_id"],
                        "dataset": r["dataset"],
                        "device_id": r["device_id"],
                        "ts_ns": int(r["ts_ns"]),
                        "camera_name": r["camera_name"],
                        "thumbnail_uri": r["thumbnail_uri"],
                        "embedding": vec.tolist(),
                    }
                )
            tbl.add(recs)
            committed += len(recs)
            progress.advance(task, len(chunk))

    # 6. Progress sidecar for tail -f during long runs
    (Path(lance_dir) / f"{table}.progress").write_text(
        f"committed={committed}\ntotal_seen={len(rows)}\nremaining={len(todo) - committed}\n"
    )
    console.print(
        f"[green]done — committed {committed:,} new embeddings to {table}[/green]"
    )


if __name__ == "__main__":
    app()
