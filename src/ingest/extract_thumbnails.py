"""Extract thumbnails from Waymo camera_image parquets + BDD100K images.zip.

Per Decision 25 (Open Work #2 from Decision 16):
  - Reads camera_image parquets from {BUCKET_URI for Waymo}/wods_sf/camera_image/*.parquet
    (each row has JPEG bytes in column `[CameraImageComponent].image`)
  - Reads BDD100K labels.zip + images.zip from S3, filters by Decision 19 density:
    time_of_day in (night, dawn) OR weather != clear OR has ped/cyclist
  - Resizes to 256x256, writes thumbnail JPEGs to {BUCKET_URI}/thumbnails/{dataset}/{frame_id}.jpg
  - Also writes Decision-15 metadata parquet (incl. thumbnail_uri) per dataset

Designed to run on the Brev L4 GPU host (uses CPU for decode + I/O; GPU idle here).
Pre-step before any embedding job.

USAGE:
    python -m src.ingest.extract_thumbnails waymo   # ~50K Waymo frames
    python -m src.ingest.extract_thumbnails bdd     # ~35K BDD100K dense frames
    python -m src.ingest.extract_thumbnails all     # both, sequentially
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

from src.cloud import get_fs, join as uri_join
from src.frame_id import frame_id

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

THUMB_SIZE = (256, 256)
JPEG_QUALITY = 80


# ----------------------------------------------------------------------------
# WAYMO
# ----------------------------------------------------------------------------

def _waymo_camera_name(idx: int) -> str:
    """Waymo v2 camera_image.key.camera_name is int enum: 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT."""
    return {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}.get(int(idx), f"CAM_{idx}")


def _stride_for(sample_hz: float, native_hz: float = 10.0) -> int:
    """How many native frames to skip between samples. 5 Hz from 10 Hz native = stride 2."""
    stride = max(1, int(round(native_hz / sample_hz)))
    return stride


def process_waymo(bucket_uri: str, sample_hz: float, max_segments: int | None = None) -> dict:
    """Iterate Waymo camera_image parquets, decode JPEGs, resize, push thumbnails to S3.

    Returns summary dict for logging.
    """
    from PIL import Image
    import pyarrow.parquet as pq

    # WAYMO_SOURCE_URI in .env points to the SEGMENT ROOT (e.g. gs://.../wods_sf).
    # We append /camera_image to read the JPEG-bearing topic parquets.
    waymo_root = os.environ.get("WAYMO_SOURCE_URI", "").rstrip("/")
    if not waymo_root:
        raise RuntimeError("WAYMO_SOURCE_URI not set. Copy .env.example to .env and set your Waymo source bucket URI.")
    waymo_src = f"{waymo_root}/camera_image"
    thumb_prefix = uri_join(bucket_uri, "thumbnails", "waymo")

    src_fs, src_path = get_fs(waymo_src)
    thumb_fs, thumb_path = get_fs(thumb_prefix)

    # List parquet files
    parquets = [p for p in src_fs.ls(src_path) if p.endswith(".parquet")]
    if max_segments:
        parquets = parquets[:max_segments]
    console.print(f"[blue]Waymo[/blue]: {len(parquets)} segment parquets to process")

    stride = _stride_for(sample_hz)
    metadata_rows = []
    thumb_count = 0
    err_count = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total} segs"), TimeElapsedColumn(), console=console) as prog:
        seg_task = prog.add_task("waymo", total=len(parquets))

        for pq_path in parquets:
            seg = Path(pq_path).stem  # segment_context_name
            try:
                with src_fs.open(pq_path, "rb") as f:
                    table = pq.read_table(f)
            except Exception as e:
                console.print(f"  [red]read fail {seg}: {e}[/red]")
                err_count += 1
                prog.update(seg_task, advance=1)
                continue

            # Get the JPEG bytes column; column names use the [CameraImageComponent] prefix
            # in Waymo v2 parquet schema
            img_col = None
            for name in table.schema.names:
                if "CameraImageComponent" in name and name.endswith(".image"):
                    img_col = name
                    break
            if img_col is None:
                console.print(f"  [yellow]skip {seg}: no image column[/yellow]")
                prog.update(seg_task, advance=1)
                continue

            ts_col = "key.frame_timestamp_micros"
            cam_col = "key.camera_name"
            seg_col = "key.segment_context_name"

            # Index by (ts_micros, cam_name) to apply stride per-camera
            # Group by camera, sort by ts, take every Nth
            df = table.select([seg_col, ts_col, cam_col, img_col]).to_pandas()
            df.columns = ["seg", "ts_us", "cam_idx", "img"]

            for cam_idx, cam_group in df.groupby("cam_idx"):
                cam_group = cam_group.sort_values("ts_us").reset_index(drop=True)
                # Apply stride
                sampled = cam_group.iloc[::stride]
                cam_name = _waymo_camera_name(cam_idx)

                for _, row in sampled.iterrows():
                    ts_ns = int(row["ts_us"]) * 1000
                    fid = frame_id("waymo", row["seg"], ts_ns, cam_name)
                    try:
                        img = Image.open(io.BytesIO(row["img"])).convert("RGB")
                        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
                        # Write to S3 thumbnail prefix
                        dst = f"{thumb_path}/{fid}.jpg"
                        with thumb_fs.open(dst, "wb") as f:
                            f.write(buf.getvalue())
                        metadata_rows.append({
                            "frame_id": fid,
                            "dataset": "waymo",
                            "device_id": row["seg"],
                            "ts_ns": ts_ns,
                            "camera_name": cam_name,
                            "thumbnail_uri": f"{thumb_prefix}/{fid}.jpg",
                        })
                        thumb_count += 1
                    except Exception as e:
                        err_count += 1
                        if err_count < 5:
                            console.print(f"  [red]frame fail: {e}[/red]")

            prog.update(seg_task, advance=1)

    # Write rolling metadata as parquet sidecar (just keys + thumbnail_uri — topic cols filled by separate step)
    if metadata_rows:
        import pyarrow as pa
        meta_uri = uri_join(bucket_uri, os.environ.get("WAYMO_PREFIX", "waymo"), "metadata", "thumbnails_index.parquet")
        meta_fs, meta_path = get_fs(meta_uri)
        try:
            meta_fs.makedirs(meta_path.rsplit("/", 1)[0], exist_ok=True)
        except (AttributeError, NotImplementedError):
            pass
        table = pa.Table.from_pylist(metadata_rows)
        with meta_fs.open(meta_path, "wb") as f:
            pq.write_table(table, f, compression="zstd")
        console.print(f"  → metadata index: {meta_uri}")

    return {"dataset": "waymo", "thumbs_written": thumb_count, "errors": err_count, "metadata_uri": meta_uri if metadata_rows else None}


# ----------------------------------------------------------------------------
# BDD100K
# ----------------------------------------------------------------------------

def _bdd_density_filter(labels_obj: dict) -> bool:
    """Decision 19 density filter for BDD100K.

    Keep if: time_of_day in (night, dawn/dusk) OR weather in (rainy, snowy, foggy, overcast)
             OR has ≥1 pedestrian OR has ≥1 cyclist.
    Drop the boring clear-day-highway-no-pedestrian frames.
    """
    attrs = labels_obj.get("attributes", {})
    tod = (attrs.get("timeofday") or "").lower()
    wx = (attrs.get("weather") or "").lower()

    if tod in {"night", "dawn/dusk"}:
        return True
    if wx in {"rainy", "snowy", "foggy", "overcast"}:
        return True

    # Object density check
    labels = labels_obj.get("labels") or []
    n_ped = sum(1 for L in labels if L.get("category") == "person")
    n_cyc = sum(1 for L in labels if L.get("category") in {"bike", "rider", "motor"})
    if n_ped >= 1 or n_cyc >= 1:
        return True

    return False


def process_bdd100k(bucket_uri: str, max_frames: int | None = None) -> dict:
    """Pull BDD100K labels.zip + images.zip from S3, filter by density, push thumbnails.

    Reads zips directly from S3 (no local extraction). Filters at decode time.
    """
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq

    bdd_prefix = os.environ.get("BDD100K_PREFIX", "bdd100k")
    labels_uri = uri_join(bucket_uri, bdd_prefix, "raw", "bdd100k_labels.zip")
    images_uri = uri_join(bucket_uri, bdd_prefix, "raw", "bdd100k_images.zip")
    thumb_prefix = uri_join(bucket_uri, "thumbnails", "bdd100k")

    src_fs, _ = get_fs(labels_uri)
    thumb_fs, thumb_path = get_fs(thumb_prefix)

    console.print(f"[orange3]BDD100K[/orange3]: loading labels from {labels_uri}")
    # Stream labels zip from S3 to memory (~147 MB, fits)
    labels_path = labels_uri.replace("s3://", "").replace("gs://", "")
    with src_fs.open(labels_path, "rb") as f:
        labels_bytes = f.read()
    labels_zip = zipfile.ZipFile(io.BytesIO(labels_bytes))

    # BDD100K labels.zip can come in TWO formats depending on the source:
    #   Format A — Berkeley official: one consolidated JSON file
    #              `bdd100k_labels_images_train.json` (list of ~70K label objects)
    #   Format B — archive.org mirror: per-image JSON files
    #              `bdd100k/labels/100k/train/<image_id>.json` (one dict per image)
    # Detect which format we have, then load accordingly.
    label_candidates = labels_zip.namelist()
    big_label_files = [
        n for n in label_candidates
        if n.endswith(".json")
        and ("bdd100k_labels_images_train" in n or "labels_images_train" in n or "/det_train.json" in n)
    ]

    if big_label_files:
        # Format A — single consolidated file
        chosen = big_label_files[0]
        console.print(f"  format A (consolidated): {chosen}")
        with labels_zip.open(chosen) as f:
            all_labels = json.load(f)
        if not isinstance(all_labels, list):
            console.print(f"[red]Expected list of label objects, got {type(all_labels).__name__}[/red]")
            raise typer.Exit(5)
    else:
        # Format B — per-image JSONs. Iterate all of them.
        per_image_jsons = [n for n in label_candidates if n.endswith(".json") and "/train/" in n]
        console.print(f"  format B (per-image): {len(per_image_jsons)} JSON files to load")
        all_labels = []
        load_errors = 0
        for n in per_image_jsons:
            try:
                with labels_zip.open(n) as f:
                    obj = json.load(f)
                # Normalize: ensure obj["name"] exists (the matching key for image filenames)
                if isinstance(obj, dict):
                    # Per-image JSONs may not include the image extension in `name`;
                    # derive it from the zip path so the downstream matcher works.
                    if "name" not in obj:
                        obj["name"] = Path(n).stem + ".jpg"
                    elif not obj["name"].endswith((".jpg", ".jpeg", ".png")):
                        obj["name"] = obj["name"] + ".jpg"
                    all_labels.append(obj)
            except Exception:
                load_errors += 1
        if load_errors:
            console.print(f"  [yellow]{load_errors} per-image JSON load errors (skipped)[/yellow]")
    console.print(f"  loaded {len(all_labels)} label entries")

    # Filter by Decision 19 density
    kept = [L for L in all_labels if _bdd_density_filter(L)]
    console.print(f"  after density filter: {len(kept)} kept of {len(all_labels)} ({100*len(kept)/len(all_labels):.1f}%)")

    if max_frames:
        kept = kept[:max_frames]
        console.print(f"  truncated to {len(kept)} for run")

    # Now stream images zip and pull only the kept frames
    console.print(f"  opening images zip stream from {images_uri}")
    images_path = images_uri.replace("s3://", "").replace("gs://", "")

    # Build a map of kept frame names
    kept_names = {L["name"]: L for L in kept}
    # BDD100K image paths in zip are like: bdd100k/images/100k/train/0000f77c-6257be58.jpg
    name_to_zippath = {}

    with src_fs.open(images_path, "rb") as f:
        img_zip = zipfile.ZipFile(f, allowZip64=True)
        for n in img_zip.namelist():
            base = Path(n).name
            if base in kept_names:
                name_to_zippath[base] = n
        console.print(f"  matched {len(name_to_zippath)} kept names to zip entries")

        metadata_rows = []
        err_count = 0

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn(), console=console) as prog:
            t = prog.add_task("bdd100k", total=len(name_to_zippath))
            for name, zippath in name_to_zippath.items():
                try:
                    with img_zip.open(zippath) as zf:
                        img = Image.open(zf).convert("RGB")
                        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=JPEG_QUALITY)

                    fid = frame_id("bdd100k", "bdd100k_train", hash(name) & 0x7FFFFFFFFFFFFFFF, "FRONT")
                    dst = f"{thumb_path}/{fid}.jpg"
                    with thumb_fs.open(dst, "wb") as f2:
                        f2.write(buf.getvalue())

                    L = kept_names[name]
                    attrs = L.get("attributes", {})
                    labels = L.get("labels") or []
                    metadata_rows.append({
                        "frame_id": fid,
                        "dataset": "bdd100k",
                        "device_id": "bdd100k_train",
                        "ts_ns": hash(name) & 0x7FFFFFFFFFFFFFFF,
                        "camera_name": "FRONT",
                        "thumbnail_uri": f"{thumb_prefix}/{fid}.jpg",
                        "source_name": name,
                        "time_of_day": attrs.get("timeofday"),
                        "weather": attrs.get("weather"),
                        "scene": attrs.get("scene"),
                        "num_pedestrians": sum(1 for L in labels if L.get("category") == "person"),
                        "num_vehicles": sum(1 for L in labels if L.get("category") in {"car", "truck", "bus"}),
                        "num_cyclists": sum(1 for L in labels if L.get("category") in {"bike", "rider", "motor"}),
                    })
                except Exception as e:
                    err_count += 1
                    if err_count < 5:
                        console.print(f"  [red]bdd frame fail {name}: {e}[/red]")
                prog.update(t, advance=1)

    # Write BDD metadata parquet
    if metadata_rows:
        meta_uri = uri_join(bucket_uri, bdd_prefix, "metadata", "thumbnails_index.parquet")
        meta_fs, meta_path = get_fs(meta_uri)
        try:
            meta_fs.makedirs(meta_path.rsplit("/", 1)[0], exist_ok=True)
        except (AttributeError, NotImplementedError):
            pass
        table = pa.Table.from_pylist(metadata_rows)
        with meta_fs.open(meta_path, "wb") as f:
            pq.write_table(table, f, compression="zstd")
        console.print(f"  → BDD metadata: {meta_uri}")

    return {"dataset": "bdd100k", "thumbs_written": len(metadata_rows), "errors": err_count}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

@app.command()
def waymo(
    sample_hz: float = typer.Option(5.0, help="Sampling Hz per camera (Decision 19 default)"),
    max_segments: int = typer.Option(None, help="Cap segments (for smoke test)"),
):
    """Extract thumbnails for Waymo (from {WAYMO_SOURCE_URI}/camera_image/*.parquet)."""
    load_dotenv()
    bucket = os.environ.get("BUCKET_URI")
    if not bucket:
        console.print("[red]BUCKET_URI not set in .env[/red]")
        raise typer.Exit(2)
    summary = process_waymo(bucket, sample_hz, max_segments)
    console.print(f"\n[green]✓ {summary}[/green]")


@app.command()
def bdd(max_frames: int = typer.Option(None, help="Cap frames (for smoke test)")):
    """Extract thumbnails for BDD100K (density-filtered per Decision 19)."""
    load_dotenv()
    bucket = os.environ.get("BUCKET_URI")
    if not bucket:
        console.print("[red]BUCKET_URI not set in .env[/red]")
        raise typer.Exit(2)
    summary = process_bdd100k(bucket, max_frames)
    console.print(f"\n[green]✓ {summary}[/green]")


@app.command()
def all(
    sample_hz: float = typer.Option(5.0),
    max_waymo_segments: int = typer.Option(None),
    max_bdd_frames: int = typer.Option(None),
):
    """Extract thumbnails for both datasets."""
    load_dotenv()
    bucket = os.environ.get("BUCKET_URI")
    if not bucket:
        console.print("[red]BUCKET_URI not set in .env[/red]")
        raise typer.Exit(2)
    s1 = process_waymo(bucket, sample_hz, max_waymo_segments)
    s2 = process_bdd100k(bucket, max_bdd_frames)
    console.print(f"\n[green]✓ Waymo: {s1}[/green]")
    console.print(f"[green]✓ BDD100K: {s2}[/green]")


if __name__ == "__main__":
    app()
