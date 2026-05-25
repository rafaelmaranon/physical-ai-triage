"""Embed local MCAP frames with SigLIP/CLIP/DINOv2 → LanceDB tables.

Reads data/mcap_frames/metadata.parquet + corresponding JPEGs, embeds each
frame with the requested model(s), writes to data/lance/waymo_mcap_<model>.

CLI:
    uv run python -m src.embed.embed_mcap_frames                 # all 3 models
    uv run python -m src.embed.embed_mcap_frames --model siglip  # one model
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


MODELS = {
    "siglip":  {"hf_id": "google/siglip-base-patch16-224", "dim": 768},
    "clip":    {"hf_id": "openai/clip-vit-large-patch14", "dim": 768},
    "dinov2":  {"hf_id": "facebook/dinov2-large",         "dim": 1024},
}


def _load_model(model_name: str, device: str = "cpu"):
    import torch
    from transformers import AutoModel, AutoProcessor
    cfg = MODELS[model_name]
    processor = AutoProcessor.from_pretrained(cfg["hf_id"])
    model = AutoModel.from_pretrained(cfg["hf_id"]).to(device).eval()
    return processor, model, cfg["dim"]


def _embed_one(model_name, processor, model, img, device="cpu"):
    """Return a (dim,) float16 numpy vector."""
    import numpy as np
    import torch
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        if model_name == "dinov2":
            outputs = model(**inputs)
            vec = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
        else:
            vec = model.get_image_features(**inputs)
    v = vec.squeeze(0).cpu().numpy().astype(np.float32)
    # L2 normalize for cosine search
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).astype(np.float16)


@app.command()
def main(
    meta_parquet: str = typer.Option("data/mcap_frames/metadata.parquet"),
    model: str = typer.Option("all", help="siglip | clip | dinov2 | all"),
    lance_dir: str = typer.Option("data/lance"),
):
    import lancedb
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(meta_parquet)
    console.print(f"loaded {len(df)} frame records from {meta_parquet}")
    if len(df) == 0:
        console.print("[red]empty parquet — run extract_mcap_frames first[/red]")
        raise typer.Exit(1)

    models_to_run = list(MODELS.keys()) if model == "all" else [model]
    Path(lance_dir).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lance_dir)

    for m in models_to_run:
        console.rule(f"[bold]embed with {m}")
        t0 = time.time()
        processor, model_obj, dim = _load_model(m)
        console.print(f"  loaded in {time.time()-t0:.1f}s, dim={dim}")

        rows = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), console=console) as prog:
            task = prog.add_task(f"embed_{m}", total=len(df))
            for _, r in df.iterrows():
                local_path = r["thumbnail_uri"].replace("file://", "")
                try:
                    img = Image.open(local_path).convert("RGB")
                except Exception as e:
                    console.print(f"  [yellow]skip {r['frame_id']}: {e}[/yellow]")
                    prog.advance(task)
                    continue
                emb = _embed_one(m, processor, model_obj, img)
                rows.append({
                    "frame_id": r["frame_id"],
                    "dataset": r["dataset"],
                    "device_id": r["device_id"],
                    "ts_ns": int(r["ts_ns"]),
                    "camera_name": r["camera_name"],
                    "thumbnail_uri": r["thumbnail_uri"],
                    "mcap_path": r["mcap_path"],
                    "topic": r["topic"],
                    "embedding": emb.tolist(),
                })
                prog.advance(task)

        table_name = f"waymo_mcap_{m}"
        if table_name in db.table_names():
            db.drop_table(table_name)
        db.create_table(table_name, data=rows)
        console.print(f"  ✓ wrote {len(rows)} → {table_name} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    app()
