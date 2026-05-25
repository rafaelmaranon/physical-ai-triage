"""DINOv2-large batch embedder — parallel to siglip_batch.py per Decision 23.

DINOv2 (Meta, 2024) is self-supervised — no caption/text bias. Often beats SigLIP
on image-to-image similarity (the "looks like THIS frame" needle). Good fit for
image→image retrieval over robot data where no captions exist.

Model: facebook/dinov2-large (1024-dim image embeddings).
LanceDB table naming: <dataset>_dinov2 to separate from SigLIP and CLIP.

IMPORTANT API DIFFERENCE from CLIP/SigLIP:
  DINOv2 has no get_image_features method. Use:
    outputs = model(pixel_values=...)
    embedding = outputs.pooler_output  # (B, 1024) — DINOv2 CLS-token pooled
  This is normalized internally; we re-L2-normalize for cosine search.
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
        from transformers import AutoImageProcessor, AutoModel  # noqa: F401
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e.name}[/red]. Run `uv sync`.")
        raise typer.Exit(1) from e


def _pick_device(requested: str) -> str:
    import torch
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_PREFIX_ENV = {
    "waymo": "WAYMO_PREFIX",
    "bdd100k": "BDD100K_PREFIX",
    "nuscenes": "NUSCENES_PREFIX",
    "av2": "AV2_PREFIX",
}


@app.command()
def main(
    table: str = typer.Option(..., help="Dataset name. LanceDB table = <table>_dinov2."),
    metadata_uri: str = typer.Option(None, help="Override parquet URI."),
    limit: int = typer.Option(0),
    batch: int = typer.Option(64),
    model_name: str = typer.Option("facebook/dinov2-large"),
    device: str = typer.Option(None),
    lance_dir: str = typer.Option("data/lance"),
):
    """Embed frames into LanceDB table <table>_dinov2 with DINOv2-large (self-supervised)."""
    load_dotenv()
    _lazy_imports()

    import duckdb
    import lancedb
    import numpy as np
    import pyarrow as pa
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    bucket_uri = os.environ.get("BUCKET_URI")
    if not bucket_uri and not metadata_uri:
        console.print("[red]BUCKET_URI not set and --metadata-uri not provided.[/red]")
        raise typer.Exit(2)

    if metadata_uri is None:
        prefix_env = _PREFIX_ENV.get(table)
        if not prefix_env:
            console.print(f"[red]Unknown table '{table}'[/red]")
            raise typer.Exit(3)
        prefix = os.environ.get(prefix_env, table)
        metadata_uri = uri_join(bucket_uri, prefix, "metadata", "*.parquet")

    device = _pick_device(device or os.environ.get("EMBED_DEVICE", "cuda"))
    dtype = torch.float16 if device == "cuda" else torch.float32
    lance_table = f"{table}_dinov2"

    console.rule(f"[bold]DINOv2 embed → {lance_table}")
    console.print(f"  model:        {model_name}")
    console.print(f"  device:       {device}")
    console.print(f"  metadata_uri: {metadata_uri}")

    console.print("[dim]loading model...[/dim]")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.to(dtype=dtype)
    embed_dim = model.config.hidden_size  # 1024 for dinov2-large

    Path(lance_dir).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lance_dir)
    schema = pa.schema([
        pa.field("frame_id", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("device_id", pa.string()),
        pa.field("ts_ns", pa.int64()),
        pa.field("camera_name", pa.string()),
        pa.field("thumbnail_uri", pa.string()),
        pa.field("embedding", pa.list_(pa.float16(), embed_dim)),
    ])
    if lance_table in db.table_names():
        tbl = db.open_table(lance_table)
        existing = {r["frame_id"] for r in tbl.search().select(["frame_id"]).limit(10**9).to_list()}
        console.print(f"  resuming: {len(existing):,} frames already embedded")
    else:
        tbl = db.create_table(lance_table, schema=schema, mode="create")
        existing = set()

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

    def load_image(row):
        try:
            fs, path = get_fs(row["thumbnail_uri"])
            with fs.open(path, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception as e:
            console.print(f"[yellow]skip {row['frame_id']}: {e}[/yellow]")
            return None

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    committed = 0
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
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
                outputs = model(**inputs)
                # DINOv2 returns last_hidden_state + pooler_output;
                # pooler_output is the CLS token after layernorm — use that.
                emb = outputs.pooler_output  # (B, 1024)
                # L2-normalize so cosine search via dot product works
                emb = F.normalize(emb, p=2, dim=-1)
            emb = emb.to(torch.float16).cpu().numpy()
            recs = []
            for (r, _), vec in zip(valid, emb, strict=True):
                recs.append({
                    "frame_id": r["frame_id"],
                    "dataset": r["dataset"],
                    "device_id": r["device_id"],
                    "ts_ns": int(r["ts_ns"]),
                    "camera_name": r["camera_name"],
                    "thumbnail_uri": r["thumbnail_uri"],
                    "embedding": vec.tolist(),
                })
            tbl.add(recs)
            committed += len(recs)
            progress.advance(task, len(chunk))

    console.print(f"[green]done — committed {committed:,} new DINOv2 embeddings to {lance_table}[/green]")


if __name__ == "__main__":
    app()
