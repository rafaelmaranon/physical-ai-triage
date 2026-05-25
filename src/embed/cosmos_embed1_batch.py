"""NVIDIA Cosmos-Embed1 batch — 8-frame video clips → 768-d temporal embeddings.

Per Phase-2 plan (see dashboard/architecture.html section 4):
  - Cosmos Embed1 is NVIDIA's purpose-built video embedder (used by their
    video-dataset-search community example). It captures *temporal* signal that
    single-frame embedders (SigLIP/CLIP/DINOv2/Cosmos-Reason-caption) miss.
  - Requires 8-frame clips at 1-2 FPS. Variants: 224p (256-d), 336p (768-d),
    448p (768-d). We use 336p for parity with SigLIP/CLIP dims (768).

This script combines clip extraction + Embed1 inference in one pass to avoid
persisting intermediate clips. Reads Waymo camera_image parquets, samples N
segments, takes 8 frames at 1 FPS per segment+camera, embeds, writes LanceDB.

LanceDB table: `cosmos_embed1` (single cross-dataset table, like cosmos_aug).
Also writes a "middle frame" thumbnail per clip to S3 for the viewer.

USAGE:
    python -m src.embed.cosmos_embed1_batch --max-clips 250
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

from src.cloud import get_fs, join as uri_join

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

# Cosmos Embed1 expects 8 frames at 1-2 FPS. Waymo is 10 Hz native, so stride=10 = 1 FPS.
CLIP_FRAMES = 8
NATIVE_HZ = 10
TARGET_FPS = 1
STRIDE = NATIVE_HZ // TARGET_FPS  # = 10

# Camera name enum (matches extract_thumbnails.py)
_CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}


def _lazy_imports():
    try:
        import torch  # noqa: F401
        import numpy as np  # noqa: F401
        from PIL import Image  # noqa: F401
        from transformers import AutoModel, AutoProcessor  # noqa: F401
    except ImportError as e:
        console.print(f"[red]Missing dep: {e.name}[/red]. Run `uv sync`.")
        raise typer.Exit(1) from e


@app.command()
def main(
    max_clips: int = typer.Option(250, help="How many clips to sample (per dataset that has video)"),
    seed: int = typer.Option(42),
    model_name: str = typer.Option("nvidia/Cosmos-Embed1-336p"),
    table_name: str = typer.Option("cosmos_embed1"),
    lance_dir: str = typer.Option("data/lance"),
    push_to_s3: bool = typer.Option(True, help="Also tar the LanceDB table + push to S3"),
):
    """Sample Waymo video clips, embed with Cosmos Embed1, write LanceDB."""
    load_dotenv()
    _lazy_imports()

    import lancedb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    bucket_uri = os.environ["BUCKET_URI"]
    waymo_root = os.environ.get("WAYMO_SOURCE_URI", "").rstrip("/")
    if not waymo_root:
        raise RuntimeError("WAYMO_SOURCE_URI not set. Copy .env.example to .env and set your Waymo source bucket URI.")
    waymo_src = f"{waymo_root}/camera_image"
    bucket_name = bucket_uri.replace("s3://", "").split("/")[0]
    thumb_s3_prefix = "thumbnails/cosmos_embed1_middle"

    src_fs, src_path = get_fs(waymo_src)
    import boto3
    s3_client = boto3.client("s3")

    console.rule("[bold]1/3 enumerate Waymo clip candidates (lazy)")
    parquets = sorted(p for p in src_fs.ls(src_path) if p.endswith(".parquet"))
    console.print(f"  {len(parquets)} camera_image parquets at {waymo_src}")

    # Build a LAZY list of (parquet_path, segment, camera, frame_indices) tuples.
    # We do NOT decode frames here — that happens one clip at a time during embedding.
    # This keeps host RAM flat regardless of clip count.
    rng = random.Random(seed)
    rng_files = parquets[:]
    rng.shuffle(rng_files)

    # Resize frames to model resolution at decode time — Cosmos Embed1 expects 336×336
    # (for 336p variant). Native Waymo FRONT is 1920×1280 → ~7MB/frame raw vs ~330KB resized.
    # 21x reduction in host RAM footprint per frame.
    RESIZE_TO = 336

    clip_specs = []  # list of (parquet_path, segment, camera_int, [8 frame indices], ts_us)
    seg_col = "key.segment_context_name"
    cam_col = "key.camera_name"
    ts_col = "key.frame_timestamp_micros"
    img_col = "[CameraImageComponent].image"
    target = max_clips

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("scan parquets", total=target)
        for pq_path in rng_files:
            if len(clip_specs) >= target:
                break
            try:
                with src_fs.open(pq_path, "rb") as f:
                    tbl = pq.read_table(io.BytesIO(f.read()), columns=[seg_col, cam_col, ts_col])
            except Exception as e:
                console.print(f"  [yellow]skip {pq_path}: {e}[/yellow]")
                continue
            df = tbl.to_pandas()
            if not all(c in df.columns for c in (seg_col, cam_col, ts_col)):
                continue
            for (seg, cam), grp in df.groupby([seg_col, cam_col]):
                if len(clip_specs) >= target:
                    break
                grp = grp.sort_values(ts_col).reset_index()  # keep original row index
                needed = (CLIP_FRAMES - 1) * STRIDE + 1
                if len(grp) < needed:
                    continue
                max_start = len(grp) - needed
                start = rng.randint(0, max_start)
                row_indices = grp.iloc[[start + i * STRIDE for i in range(CLIP_FRAMES)]]["index"].tolist()
                clip_specs.append({
                    "pq_path": pq_path,
                    "segment": seg,
                    "camera_int": int(cam),
                    "row_indices": row_indices,
                    "start_ts_us": int(grp.iloc[start][ts_col]),
                })
                prog.advance(task)

    console.print(f"  scheduled {len(clip_specs)} clip specs (lazy — frames decoded one clip at a time)")
    if not clip_specs:
        console.print("[red]no clips scheduled[/red]")
        raise typer.Exit(2)
    # Sort by parquet path so the decode loop hits each parquet once (cache-friendly)
    clip_specs.sort(key=lambda c: c["pq_path"])

    console.rule("[bold]2/3 load Cosmos Embed1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        console.print("[red]No CUDA — Cosmos Embed1 needs a GPU. Aborting.[/red]")
        raise typer.Exit(3)
    console.print(f"  device: {device} · model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # Drop bfloat16 — device_map="auto" leaves layer_norms in fp32 which mismatches
    # bf16 inputs. Cosmos Embed1-336p is small (~200MB FP32) so fp32 fits trivially
    # on L4 22GB VRAM. Slower per clip but always works.
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    console.print("  model loaded (fp32)")

    console.rule("[bold]3/3 stream-decode + embed clips (one at a time)")
    results = []
    # Cache the parquet we're currently reading from (avoid re-downloading per clip)
    cached_pq_path = None
    cached_df = None
    from src.frame_id import frame_id as make_fid

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("embed_clips", total=len(clip_specs))
        for i, spec in enumerate(clip_specs):
            # Load parquet (re-use cache if same file)
            if spec["pq_path"] != cached_pq_path:
                try:
                    with src_fs.open(spec["pq_path"], "rb") as f:
                        cached_df = pq.read_table(io.BytesIO(f.read())).to_pandas()
                    cached_pq_path = spec["pq_path"]
                except Exception as e:
                    console.print(f"  [yellow]skip {spec['pq_path']}: {e}[/yellow]")
                    prog.advance(task)
                    continue

            # Decode 8 frames, resize on the fly to RESIZE_TO×RESIZE_TO
            try:
                frames = []
                for idx in spec["row_indices"]:
                    img = Image.open(io.BytesIO(cached_df.iloc[idx][img_col])).convert("RGB")
                    img = img.resize((RESIZE_TO, RESIZE_TO), Image.BILINEAR)
                    frames.append(np.asarray(img))
                middle = Image.open(io.BytesIO(cached_df.iloc[spec["row_indices"][4]][img_col])).convert("RGB")
            except Exception as e:
                console.print(f"  [yellow]decode err: {e}[/yellow]")
                prog.advance(task)
                continue

            # BTCHW: batch=1, time=8, channel=3, H, W
            arr = np.stack(frames, axis=0)          # T H W C
            arr = np.transpose(arr, (0, 3, 1, 2))   # T C H W
            arr = np.expand_dims(arr, 0)            # 1 T C H W
            video_inputs = processor(videos=arr).to(device)
            with torch.no_grad():
                out = model.get_video_embeddings(**video_inputs)
            # Extract the tensor from VideoEmbedderOutput dataclass.
            # Try common attribute names; fall back to the first tensor we find.
            emb_tensor = None
            for attr in ("video_embeds", "embedding", "embeddings", "last_hidden_state", "pooler_output"):
                if hasattr(out, attr):
                    cand = getattr(out, attr)
                    if hasattr(cand, "squeeze"):
                        emb_tensor = cand
                        break
            if emb_tensor is None:
                # Last resort: iterate dataclass fields
                for v in (out.__dict__.values() if hasattr(out, "__dict__") else []):
                    if hasattr(v, "squeeze"):
                        emb_tensor = v
                        break
            if emb_tensor is None:
                raise RuntimeError(f"Cannot find embedding tensor in {type(out).__name__}: {dir(out)}")
            emb = emb_tensor.squeeze(0).to(torch.float16).cpu().numpy()
            n = float(np.linalg.norm(emb)) or 1.0
            emb = (emb / n).astype(np.float16)

            cam_name = _CAM.get(spec["camera_int"], f"CAM_{spec['camera_int']}")
            ts_ns = spec["start_ts_us"] * 1000
            fid = make_fid("waymo", spec["segment"], ts_ns, cam_name)

            # Middle-frame thumbnail for viewer (256×256, separate from model input)
            middle.thumbnail((256, 256))
            buf = io.BytesIO()
            middle.save(buf, format="JPEG", quality=80)
            thumb_key = f"{thumb_s3_prefix}/{fid}.jpg"
            s3_client.put_object(Bucket=bucket_name, Key=thumb_key, Body=buf.getvalue(),
                                  ContentType="image/jpeg")
            thumb_uri = f"s3://{bucket_name}/{thumb_key}"

            results.append({
                "frame_id": fid,
                "dataset": "waymo",
                "device_id": spec["segment"],
                "ts_ns": ts_ns,
                "camera_name": cam_name,
                "thumbnail_uri": thumb_uri,
                "embedding": emb.tolist(),
                "clip_frames": CLIP_FRAMES,
                "clip_fps": TARGET_FPS,
            })
            # Free decoded frames immediately
            del frames, arr, video_inputs, out, emb
            prog.advance(task)

    console.rule("[bold]write LanceDB")
    Path(lance_dir).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lance_dir)
    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, data=results)
    console.print(f"  [green]✓ {len(results)} embeddings → {lance_dir}/{table_name}[/green]")

    if push_to_s3:
        console.rule("[bold]push LanceDB + descriptions to S3")
        import tarfile, boto3
        b = bucket_uri.replace("s3://", "").split("/")[0]
        s3 = boto3.client("s3")
        ld = Path(lance_dir) / f"{table_name}.lance"
        if ld.exists():
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(str(ld), arcname=f"{table_name}.lance")
            buf.seek(0)
            s3.put_object(Bucket=b, Key=f"cosmos/{table_name}.lance.tar.gz", Body=buf.getvalue())
            console.print(f"  uploaded s3://{b}/cosmos/{table_name}.lance.tar.gz ({len(buf.getvalue())//1024} KB)")

    console.print(f"\n[green]DONE — {len(results)} Cosmos Embed1 clips embedded[/green]")


if __name__ == "__main__":
    app()
