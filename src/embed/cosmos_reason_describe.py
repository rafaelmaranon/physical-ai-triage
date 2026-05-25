"""Cosmos-Reason1-7B scene description on a 500-frame sample.

Per Decision 23 (4-model bake-off): NVIDIA Cosmos-Reason1-7B is a vision-language model
trained on driving + physical-AI data. Used here as a SCENE DESCRIBER:
  - Input: camera thumbnail (256x256 RGB) + text prompt
  - Output: natural-language description ("Dense urban intersection at night, pedestrians
    crossing left-to-right, vehicle approaching from oncoming lane")
  - Downstream: cosmos_aug_embed.py embeds those descriptions with SigLIP text encoder
    → "Cosmos-augmented" embeddings for comparison vs. raw SigLIP/CLIP/DINOv2

WHY 500 SAMPLE: VLMs are slow (~3 frames/sec on L4). Full 85K would take 8h+. 500 frames
is enough for honest A/B comparison on the 12 needle queries (~40 frames per query budget).
Sample is stratified: 250 Waymo + 250 BDD100K.

MEMORY: Cosmos-Reason1-7B in FP16 = ~14 GB VRAM. Fits on L4 24GB.

OUTPUT: writes eval/cosmos_descriptions.parquet with (frame_id, description) rows.
Downstream `cosmos_aug_embed.py` reads this + creates LanceDB table.
"""

from __future__ import annotations

import io
import os
import random
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

from src.cloud import duckdb_with_s3, get_fs, join as uri_join

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

DESCRIBE_PROMPT = (
    "Describe this driving scene in 2-3 sentences. Focus on: "
    "scene type (urban intersection, highway, residential), "
    "time of day and weather, "
    "actors (pedestrians, cyclists, vehicles) and their positions, "
    "any unusual or noteworthy elements."
)


def _lazy_imports():
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: F401
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e.name}[/red]. Run `uv sync`.")
        raise typer.Exit(1) from e


@app.command()
def main(
    sample_size: int = typer.Option(500, help="Total frames to describe (stratified across datasets)"),
    seed: int = typer.Option(42),
    model_name: str = typer.Option("nvidia/Cosmos-Reason1-7B"),
    batch: int = typer.Option(4, help="Batch size — keep low, VLM generation is memory-hungry"),
    max_new_tokens: int = typer.Option(120, help="Max tokens per description"),
    out_path: str = typer.Option("eval/cosmos_descriptions.parquet", help="Output parquet (use distinct paths to bench multiple model versions)"),
):
    """Generate Cosmos descriptions for a sample of frames."""
    load_dotenv()
    _lazy_imports()

    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    bucket_uri = os.environ["BUCKET_URI"]

    # 1. Sample stratified across datasets from metadata parquets
    console.rule("[bold]1/3 sample frames")
    con = duckdb_with_s3(duckdb.connect())
    random.seed(seed)

    samples = []
    for ds, prefix_env in [("waymo", "WAYMO_PREFIX"), ("bdd100k", "BDD100K_PREFIX")]:
        prefix = os.environ.get(prefix_env, ds)
        meta_uri = uri_join(bucket_uri, prefix, "metadata", "thumbnails_index.parquet")
        try:
            rows = con.execute(
                f"SELECT frame_id, dataset, device_id, ts_ns, camera_name, thumbnail_uri "
                f"FROM read_parquet('{meta_uri}') USING SAMPLE reservoir({sample_size // 2} ROWS) REPEATABLE ({seed})"
            ).fetchall()
            samples.extend(rows)
            console.print(f"  {ds}: {len(rows)} samples")
        except Exception as e:
            console.print(f"  [yellow]{ds}: skip — {e}[/yellow]")
    if not samples:
        console.print("[red]No metadata found — extract step must complete first.[/red]")
        raise typer.Exit(2)

    cols = ("frame_id", "dataset", "device_id", "ts_ns", "camera_name", "thumbnail_uri")

    # 2. Load Cosmos-Reason1-7B
    console.rule("[bold]2/3 load model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        console.print("[red]No CUDA — Cosmos-Reason1-7B is unusable on CPU. Aborting.[/red]")
        raise typer.Exit(3)
    console.print(f"  device: {device}")
    console.print("[dim]downloading + loading (5-10 min first time, cached after)...[/dim]")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # device_map="auto" + low_cpu_mem_usage streams weights straight to GPU,
    # avoiding the host-RAM OOM on 16GB g6.xlarge (model is 14GB in FP16).
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # 3. Iterate batches, generate descriptions
    console.rule("[bold]3/3 generate")
    results = []

    def load_thumb(uri: str) -> Image.Image | None:
        try:
            fs, path = get_fs(uri)
            with fs.open(path, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception:
            return None

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("cosmos", total=len(samples))
        for chunk in batched(samples, batch):
            recs = [dict(zip(cols, r)) for r in chunk]
            imgs = [load_thumb(r["thumbnail_uri"]) for r in recs]
            pairs = [(r, im) for r, im in zip(recs, imgs) if im is not None]
            if not pairs:
                prog.advance(task, len(chunk))
                continue

            # Cosmos-Reason1 follows a chat-template style
            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": DESCRIBE_PROMPT},
                        ],
                    }
                ]
                for _ in pairs
            ]
            prompts = [
                processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in messages_batch
            ]
            inputs = processor(
                text=prompts,
                images=[im for _, im in pairs],
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                )

            decoded = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for (r, _), desc in zip(pairs, decoded):
                results.append({
                    "frame_id": r["frame_id"],
                    "dataset": r["dataset"],
                    "device_id": r["device_id"],
                    "ts_ns": int(r["ts_ns"]),
                    "camera_name": r["camera_name"],
                    "thumbnail_uri": r["thumbnail_uri"],
                    "description": desc.strip(),
                })
            prog.advance(task, len(chunk))

    # 4. Persist
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    table = pa.Table.from_pylist(results)
    pq.write_table(table, out_path, compression="zstd")
    console.print(f"\n[green]✓ wrote {len(results):,} descriptions to {out_path}[/green]")
    console.print(f"  next: python -m src.embed.cosmos_aug_embed")


if __name__ == "__main__":
    app()
