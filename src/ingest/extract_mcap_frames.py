"""Extract frames from local Waymo MCAPs → JPEGs + per-frame metadata parquet.

Sister script to extract_thumbnails.py, but reads MCAPs the user already has
in Foxglove locally (not the GCS Waymo parquet source).

For each MCAP file, samples N frames per camera at TARGET_FPS, decodes the
compressed image, resizes to THUMB_SIZE, writes a JPEG per frame. Builds a
metadata parquet with: frame_id, mcap_path, segment_name, camera_name, ts_ns,
thumbnail_uri (local file path).

This enables searches that link back to Foxglove via deeplinks at the exact
timestamp, closing the end-to-end loop.

CLI:
    uv run python -m src.ingest.extract_mcap_frames \\
        --mcap-glob 'data/mcap_local/*.mcap' \\
        --target-fps 1 --thumb-size 256
"""
from __future__ import annotations

import glob
import io
import json
from pathlib import Path
from typing import Iterable

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.frame_id import frame_id as make_fid

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

# Topic name patterns for camera images — Waymo MCAPs use varying conventions.
# We match any topic containing "camera" + "compressed_image" or "image" + a camera role.
_CAMERA_TOPICS = {
    # seg1_v* MCAPs use these
    "/camera/front":          "FRONT",
    "/camera/front_left":     "FRONT_LEFT",
    "/camera/front_right":    "FRONT_RIGHT",
    "/camera/side_left":      "SIDE_LEFT",
    "/camera/side_right":     "SIDE_RIGHT",
    # official_waymo_2026.mcap uses these
    "/CAMERA_FRONT/compressed_image":          "FRONT",
    "/CAMERA_FRONT_LEFT/compressed_image":     "FRONT_LEFT",
    "/CAMERA_FRONT_RIGHT/compressed_image":    "FRONT_RIGHT",
    "/CAMERA_SIDE_LEFT/compressed_image":      "SIDE_LEFT",
    "/CAMERA_SIDE_RIGHT/compressed_image":     "SIDE_RIGHT",
}


def _decode_image(msg_data: bytes, schema_name: str) -> bytes | None:
    """Extract raw JPEG/PNG bytes from a Foxglove CompressedImage message.
    Foxglove schema CompressedImage has fields: timestamp, frame_id, data, format.
    We use a minimal CDR-like decoder by finding the data field after the header.
    """
    # The simplest path: try the foxglove_schemas_protobuf package if available
    try:
        from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
        img = CompressedImage()
        img.ParseFromString(msg_data)
        return bytes(img.data)
    except Exception:
        pass
    # Fallback: look for JPEG/PNG magic bytes in the message body
    jpeg_start = msg_data.find(b"\xff\xd8\xff")
    if jpeg_start >= 0:
        return msg_data[jpeg_start:]
    png_start = msg_data.find(b"\x89PNG\r\n\x1a\n")
    if png_start >= 0:
        return msg_data[png_start:]
    return None


@app.command()
def main(
    mcap_glob: str = typer.Option("data/mcap_local/*.mcap", help="glob pattern for MCAP files (default: data/mcap_local/)"),
    out_dir: str = typer.Option("data/mcap_frames"),
    target_fps: float = typer.Option(1.0, help="frames per second per camera"),
    thumb_size: int = typer.Option(256, help="thumbnail max edge in pixels"),
    max_per_mcap: int = typer.Option(50, help="cap on total frames per MCAP (across cameras)"),
):
    """Extract frames from local MCAPs + write metadata parquet."""
    from mcap.reader import make_reader
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq

    mcap_files = sorted(glob.glob(mcap_glob))
    # Only the substantial Waymo MCAPs (skip the tiny robot tests + the v2 stub)
    mcap_files = [f for f in mcap_files
                  if Path(f).stat().st_size > 100_000_000
                  and "robot" not in Path(f).name.lower()]
    console.print(f"[bold]{len(mcap_files)} large Waymo MCAPs to process:[/bold]")
    for f in mcap_files:
        console.print(f"  {f}  ({Path(f).stat().st_size//1024//1024} MB)")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_records = []

    for mcap_path in mcap_files:
        mcap_name = Path(mcap_path).stem
        console.rule(f"[bold]{mcap_name}")
        try:
            with open(mcap_path, "rb") as f:
                r = make_reader(f)
                summary = r.get_summary()
                start_ns = summary.statistics.message_start_time
                end_ns = summary.statistics.message_end_time
                duration_s = (end_ns - start_ns) / 1e9

                # Find camera channels in this MCAP
                camera_channels = {}
                for ch_id, ch in summary.channels.items():
                    if ch.topic in _CAMERA_TOPICS:
                        camera_channels[ch_id] = _CAMERA_TOPICS[ch.topic]
                console.print(f"  duration {duration_s:.1f}s · {len(camera_channels)} cameras")
                if not camera_channels:
                    console.print(f"  [yellow]no recognized camera topics — skipping[/yellow]")
                    continue

                # Per-camera, sample at target_fps
                stride_ns = int(1e9 / target_fps)
                last_kept_ts = {ch_id: 0 for ch_id in camera_channels}
                kept = 0

                # Re-open for streaming iteration
                pass
            # Stream messages, decode + write per stride
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages(
                    topics=list(_CAMERA_TOPICS.keys())
                ):
                    if kept >= max_per_mcap:
                        break
                    if channel.id not in camera_channels:
                        continue
                    if message.log_time - last_kept_ts[channel.id] < stride_ns:
                        continue
                    last_kept_ts[channel.id] = message.log_time

                    jpeg_bytes = _decode_image(message.data, schema.name)
                    if jpeg_bytes is None:
                        continue
                    try:
                        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                    except Exception as e:
                        continue
                    img.thumbnail((thumb_size, thumb_size))

                    cam_name = camera_channels[channel.id]
                    fid = make_fid("waymo_mcap", mcap_name, message.log_time, cam_name)
                    jpeg_out = out_path / f"{fid}.jpg"
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    jpeg_out.write_bytes(buf.getvalue())

                    all_records.append({
                        "frame_id": fid,
                        "dataset": "waymo_mcap",
                        "mcap_path": str(mcap_path),
                        "device_id": mcap_name,  # segment-equivalent
                        "ts_ns": int(message.log_time),
                        "camera_name": cam_name,
                        "thumbnail_uri": f"file://{jpeg_out.absolute()}",
                        "topic": channel.topic,
                    })
                    kept += 1
                console.print(f"  ✓ extracted {kept} frames")
        except Exception as e:
            console.print(f"  [red]error: {e}[/red]")
            continue

    if not all_records:
        console.print("[red]no frames extracted[/red]")
        raise typer.Exit(2)

    # Write metadata parquet
    meta_path = out_path / "metadata.parquet"
    table = pa.Table.from_pylist(all_records)
    pq.write_table(table, str(meta_path), compression="zstd")
    console.print(f"\n[green]✓ {len(all_records)} total frames → {out_path}[/green]")
    console.print(f"  metadata: {meta_path}")

    # Print a summary by mcap + camera
    by_mcap = {}
    for r in all_records:
        key = (r["device_id"], r["camera_name"])
        by_mcap[key] = by_mcap.get(key, 0) + 1
    console.print("\nframes per (mcap, camera):")
    for (mcap, cam), n in sorted(by_mcap.items()):
        console.print(f"  {mcap:35s} {cam:14s} {n:>3}")


if __name__ == "__main__":
    app()
