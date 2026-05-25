"""Cosmos-augmented embeddings — embed Cosmos-Reason1 descriptions with SigLIP text encoder.

Per Decision 23: this is the "augmented" embedding for Cosmos-Reason1-7B.
Pipeline:
  1. cosmos_reason_describe.py → eval/cosmos_descriptions.parquet (frame_id, description)
  2. THIS SCRIPT → embed each description's text with SigLIP text encoder → LanceDB
  3. Search: text query → SigLIP text encode → cosine search in this table

So both queries AND indexed items use the same text encoder — apples to apples comparison
with the SigLIP visual-encoder pipeline.

LanceDB table: cosmos_aug (single table for all datasets; smaller scale ~500 rows).
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _lazy_imports():
    try:
        import lancedb  # noqa: F401
        import numpy as np  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e.name}[/red]. Run `uv sync`.")
        raise typer.Exit(1) from e


@app.command()
def main(
    descriptions_path: str = typer.Option("eval/cosmos_descriptions.parquet"),
    model_name: str = typer.Option("google/siglip-base-patch16-224"),
    lance_dir: str = typer.Option("data/lance"),
    table_name: str = typer.Option("cosmos_aug"),
    batch: int = typer.Option(32),
):
    """Embed Cosmos descriptions with SigLIP text encoder → LanceDB table."""
    load_dotenv()
    _lazy_imports()

    import lancedb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    desc_path = Path(descriptions_path)
    if not desc_path.exists():
        console.print(f"[red]{desc_path} not found. Run cosmos_reason_describe.py first.[/red]")
        raise typer.Exit(2)

    table = pq.read_table(desc_path)
    rows = table.to_pylist()
    console.print(f"loaded {len(rows)} descriptions from {desc_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.to(dtype=torch.float16)

    # SigLIP text encoder output dim = vision dim (shared 768)
    embed_dim = model.config.text_config.hidden_size

    # LanceDB table
    Path(lance_dir).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lance_dir)
    schema = pa.schema([
        pa.field("frame_id", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("device_id", pa.string()),
        pa.field("ts_ns", pa.int64()),
        pa.field("camera_name", pa.string()),
        pa.field("thumbnail_uri", pa.string()),
        pa.field("description", pa.string()),
        pa.field("embedding", pa.list_(pa.float16(), embed_dim)),
    ])
    if table_name in db.table_names():
        tbl = db.open_table(table_name)
        existing = {r["frame_id"] for r in tbl.search().select(["frame_id"]).limit(10**9).to_list()}
    else:
        tbl = db.create_table(table_name, schema=schema, mode="create")
        existing = set()

    todo = [r for r in rows if r["frame_id"] not in existing and r.get("description")]
    console.print(f"pending: {len(todo)}")
    if not todo:
        console.print("[green]done — nothing new[/green]")
        return

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    committed = 0
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("embedding", total=len(todo))
        for chunk in batched(todo, batch):
            texts = [r["description"] for r in chunk]
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_text_features(**inputs)
                emb = F.normalize(emb, p=2, dim=-1)
            emb = emb.to(torch.float16).cpu().numpy()
            recs = []
            for r, vec in zip(chunk, emb, strict=True):
                recs.append({
                    "frame_id": r["frame_id"],
                    "dataset": r["dataset"],
                    "device_id": r["device_id"],
                    "ts_ns": int(r["ts_ns"]),
                    "camera_name": r["camera_name"],
                    "thumbnail_uri": r["thumbnail_uri"],
                    "description": r["description"],
                    "embedding": vec.tolist(),
                })
            tbl.add(recs)
            committed += len(recs)
            progress.advance(task, len(chunk))

    console.print(f"\n[green]done — committed {committed:,} cosmos_aug embeddings to {table_name}[/green]")


if __name__ == "__main__":
    app()
