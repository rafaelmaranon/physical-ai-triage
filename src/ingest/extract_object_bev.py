"""Extract object-BEV thumbnails from Waymo lidar_box parquets.

Second embedding modality — LiDAR-derived top-down view to complement RGB.

ARCHITECTURE NOTE — this is OBJECT-density BEV, not raw-point-cloud BEV.
  Full point-cloud BEV requires decoding Waymo v2's range_image format which is
  ~3-4h of code (range_image → unproject via lidar_calibration → xyz → BEV).
  Instead we project the LiDAR-detected 3D bounding boxes (already in lidar_box
  parquets) to a top-down 256x256 image: each box becomes a filled rectangle
  with intensity by object class (pedestrian=255, cyclist=200, vehicle=120, sign=80).

  This captures: where objects are relative to ego, density, layout, class mix.
  It does NOT capture: free-space points, road geometry, fine-grained shapes.

  Story: "object-density BEV — captures scene graph, sufficient for the 'find scenes
  with similar object layout' demo. Phase 2 = point-cloud BEV from range images."

OUTPUT:
  - thumbnails written to {BUCKET_URI}/thumbnails/object_bev/{frame_id}.jpg (256x256 grayscale)
  - metadata parquet to {BUCKET_URI}/waymo/metadata/object_bev_index.parquet
  - frame_id uses "object_bev" as camera_name (to disambiguate from RGB cameras)

Same embedder downstream: `python -m src.embed.siglip_batch --table object_bev \\
  --metadata-uri s3://.../waymo/metadata/object_bev_index.parquet`
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

from src.cloud import get_fs, join as uri_join
from src.frame_id import frame_id

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

# BEV parameters
BEV_RANGE_M = 50.0          # render ±50 m around ego
BEV_SIZE = 256              # 256x256 grayscale image
BEV_PIXELS_PER_METER = BEV_SIZE / (2 * BEV_RANGE_M)  # 2.56 px/m

# Object class → intensity (Waymo lidar_box type enum: 1=VEH, 2=PED, 3=SIGN, 4=CYC)
CLASS_INTENSITY = {
    1: 120,   # vehicle (dim — many of them)
    2: 255,   # pedestrian (brightest)
    3: 80,    # sign (background)
    4: 200,   # cyclist (visible)
}


def _draw_box(arr, cx_m: float, cy_m: float, sx_m: float, sy_m: float, intensity: int):
    """Draw a filled rectangle on the BEV image. ego is at center, +x forward, +y left."""
    # Convert vehicle frame (ego at origin, x forward, y left) → image frame (origin top-left).
    # In image: ego is at (BEV_SIZE/2, BEV_SIZE/2). +x_m (forward) → -y_pix (up).
    # +y_m (left) → -x_pix (left).
    px = int(BEV_SIZE / 2 - cy_m * BEV_PIXELS_PER_METER)
    py = int(BEV_SIZE / 2 - cx_m * BEV_PIXELS_PER_METER)
    half_w = max(1, int(sy_m * BEV_PIXELS_PER_METER / 2))
    half_h = max(1, int(sx_m * BEV_PIXELS_PER_METER / 2))
    x0, x1 = max(0, px - half_w), min(BEV_SIZE, px + half_w + 1)
    y0, y1 = max(0, py - half_h), min(BEV_SIZE, py + half_h + 1)
    if x0 < x1 and y0 < y1:
        # additive blend so overlapping boxes accumulate intensity
        arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1].clip(0, 255 - intensity) + intensity


@app.command()
def main(
    sample_hz: float = typer.Option(5.0, help="Sampling Hz (matches Waymo RGB extract)"),
    max_segments: int = typer.Option(None, help="Cap segments for smoke test"),
):
    """Generate object-density BEV thumbnails for each Waymo frame."""
    load_dotenv()

    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    bucket_uri = os.environ.get("BUCKET_URI")
    if not bucket_uri:
        console.print("[red]BUCKET_URI not set[/red]")
        raise typer.Exit(2)

    waymo_root = os.environ.get("WAYMO_SOURCE_URI", "").rstrip("/")
    if not waymo_root:
        raise RuntimeError("WAYMO_SOURCE_URI not set. Copy .env.example to .env and set your Waymo source bucket URI.")
    src_uri = f"{waymo_root}/lidar_box"
    thumb_prefix = uri_join(bucket_uri, "thumbnails", "object_bev")

    src_fs, src_path = get_fs(src_uri)
    thumb_fs, thumb_path = get_fs(thumb_prefix)

    parquets = sorted([p for p in src_fs.ls(src_path) if p.endswith(".parquet")])
    if max_segments:
        parquets = parquets[:max_segments]
    console.print(f"[purple]Object-BEV[/purple]: {len(parquets)} segments to process")

    stride_us = int(round(1_000_000 / sample_hz))  # micros between samples
    metadata_rows = []
    err_count = 0
    bev_count = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total} segs"), TimeElapsedColumn(), console=console) as prog:
        seg_task = prog.add_task("object_bev", total=len(parquets))

        for pq_path in parquets:
            seg = Path(pq_path).stem
            try:
                with src_fs.open(pq_path, "rb") as f:
                    table = pq.read_table(f)
            except Exception as e:
                console.print(f"  [red]read fail {seg}: {e}[/red]")
                err_count += 1
                prog.update(seg_task, advance=1)
                continue

            # Group rows by (segment, ts_micros). Each row is one object detection.
            # Schema (Waymo v2): key.segment_context_name, key.frame_timestamp_micros,
            #   [LiDARBoxComponent].type, [LiDARBoxComponent].box.center.x/y/z, .size.x/y/z
            cols_needed = ["key.frame_timestamp_micros"]
            type_col = None
            cx_col = cy_col = sx_col = sy_col = None
            for n in table.schema.names:
                if n.endswith(".type") and "LiDARBoxComponent" in n:
                    type_col = n
                elif n.endswith(".center.x") and "LiDARBoxComponent" in n:
                    cx_col = n
                elif n.endswith(".center.y") and "LiDARBoxComponent" in n:
                    cy_col = n
                elif n.endswith(".size.x") and "LiDARBoxComponent" in n:
                    sx_col = n
                elif n.endswith(".size.y") and "LiDARBoxComponent" in n:
                    sy_col = n
            if not all([type_col, cx_col, cy_col, sx_col, sy_col]):
                console.print(f"  [yellow]skip {seg}: missing box columns. names={table.schema.names[:5]}...[/yellow]")
                prog.update(seg_task, advance=1)
                continue

            df = table.select(["key.frame_timestamp_micros", type_col, cx_col, cy_col, sx_col, sy_col]).to_pandas()
            df.columns = ["ts_us", "type", "cx", "cy", "sx", "sy"]

            # Find unique frames in this segment, then sample at stride
            unique_ts = sorted(df["ts_us"].unique())
            if not unique_ts:
                prog.update(seg_task, advance=1)
                continue
            base_ts = unique_ts[0]
            sampled_ts = {ts for ts in unique_ts if (ts - base_ts) % stride_us == 0}

            for ts in sampled_ts:
                frame_df = df[df["ts_us"] == ts]
                if frame_df.empty:
                    continue
                # Build BEV image
                bev = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)
                # Ego marker: small bright cross at center
                bev[BEV_SIZE // 2 - 2 : BEV_SIZE // 2 + 3, BEV_SIZE // 2] = 255
                bev[BEV_SIZE // 2, BEV_SIZE // 2 - 2 : BEV_SIZE // 2 + 3] = 255
                # Boxes
                for _, row in frame_df.iterrows():
                    intensity = CLASS_INTENSITY.get(int(row["type"]), 60)
                    try:
                        _draw_box(bev, float(row["cx"]), float(row["cy"]), float(row["sx"]), float(row["sy"]), intensity)
                    except Exception:
                        pass

                ts_ns = int(ts) * 1000
                fid = frame_id("waymo", seg, ts_ns, "object_bev")
                try:
                    img = Image.fromarray(bev, mode="L").convert("RGB")  # SigLIP wants RGB
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    dst = f"{thumb_path}/{fid}.jpg"
                    with thumb_fs.open(dst, "wb") as f:
                        f.write(buf.getvalue())
                    metadata_rows.append({
                        "frame_id": fid,
                        "dataset": "waymo",
                        "device_id": seg,
                        "ts_ns": ts_ns,
                        "camera_name": "object_bev",
                        "thumbnail_uri": f"{thumb_prefix}/{fid}.jpg",
                    })
                    bev_count += 1
                except Exception as e:
                    err_count += 1
                    if err_count < 5:
                        console.print(f"  [red]bev fail: {e}[/red]")

            prog.update(seg_task, advance=1)

    # Write metadata parquet sidecar
    if metadata_rows:
        import pyarrow as pa
        meta_uri = uri_join(bucket_uri, os.environ.get("WAYMO_PREFIX", "waymo"), "metadata", "object_bev_index.parquet")
        meta_fs, meta_path = get_fs(meta_uri)
        try:
            meta_fs.makedirs(meta_path.rsplit("/", 1)[0], exist_ok=True)
        except (AttributeError, NotImplementedError):
            pass
        table = pa.Table.from_pylist(metadata_rows)
        with meta_fs.open(meta_path, "wb") as f:
            pq.write_table(table, f, compression="zstd")
        console.print(f"  → metadata: {meta_uri}")

    console.print(f"\n[green]✓ object_bev: {bev_count} thumbs, {err_count} errors[/green]")


if __name__ == "__main__":
    app()
