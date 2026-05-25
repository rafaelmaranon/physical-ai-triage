"""BDD100K labels → shared parquet schema (per ).

BDD100K's labels live as JSON files inside `bdd100k_labels.zip` — one file per
split (train/val/test), with per-image entries that contain:
  - name (image filename, derived from a 40s video at 30fps; 10th-second key frame)
  - attributes: weather, scene, timeofday  ← maps to `weather`, `time_of_day`
  - labels[]: per-object boxes with categories ← counted into num_pedestrians/vehicles/cyclists
  - videoName / timestamp (10000 ms = 10s = key frame moment)

We map each labelled image into one row of the cross-dataset parquet schema.
BDD has no ego pose / IMU / GPS / 3D box ranges, so those columns stay NULL.

Run AFTER bdd100k_download.py has placed `bdd100k_labels.zip` in S3.
Reads the zip from S3, parses in-memory, writes parquet back to
`{BUCKET_URI}/{BDD100K_PREFIX}/metadata/bdd100k_metadata.parquet`.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.cloud import get_fs, join as uri_join
from src.frame_id import frame_id

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# BDD object class → category bucket used by the shared schema's count columns.
PEDESTRIAN_CLASSES = {"person", "pedestrian", "rider"}
VEHICLE_CLASSES = {"car", "truck", "bus", "train"}
CYCLIST_CLASSES = {"bike", "motor", "bicycle", "motorcycle"}


# BDD scene/timeofday/weather attribute strings → shared enum values.
# Most BDD strings already match the schema; this mapping documents the contract.
TIME_OF_DAY_MAP = {
    "daytime": "day",
    "night": "night",
    "dawn/dusk": "dawn_dusk",
    "undefined": None,
}
WEATHER_MAP = {
    "clear": "clear",
    "rainy": "rain",
    "snowy": "snow",
    "overcast": "overcast",
    "partly cloudy": "partly_cloudy",
    "foggy": "fog",
    "undefined": None,
}


def _count_objects(labels: list[dict]) -> tuple[int, int, int]:
    """Return (num_pedestrians, num_vehicles, num_cyclists)."""
    p = v = c = 0
    for obj in labels or []:
        cat = (obj.get("category") or "").lower()
        if cat in PEDESTRIAN_CLASSES:
            p += 1
        elif cat in VEHICLE_CLASSES:
            v += 1
        elif cat in CYCLIST_CLASSES:
            c += 1
    return p, v, c


def _frame_record(entry: dict, thumb_base: str) -> dict:
    """Map one BDD label entry → one row in the shared schema."""
    name = entry.get("name", "")
    device_id = entry.get("videoName") or name.rsplit(".", 1)[0]
    ts_ns = int(entry.get("timestamp", 0)) * 1_000_000  # BDD timestamps are ms

    attrs = entry.get("attributes") or {}
    weather_raw = (attrs.get("weather") or "undefined").lower()
    tod_raw = (attrs.get("timeofday") or "undefined").lower()
    p, v, c = _count_objects(entry.get("labels", []))

    fid = frame_id("bdd100k", device_id, ts_ns, "FRONT")
    return {
        "frame_id": fid,
        "dataset": "bdd100k",
        "device_id": device_id,
        "ts_ns": ts_ns,
        "camera_name": "FRONT",
        "thumbnail_uri": f"{thumb_base}/{fid}.jpg",
        "city": None,  # BDD scene attr has "city street" / "highway" — not city name
        # BDD has no ego pose / IMU / GPS / 3D ranges
        "ego_speed_mps": None,
        "ego_heading_deg": None,
        "ego_accel_x_mps2": None,
        "ego_accel_y_mps2": None,
        "ego_accel_z_mps2": None,
        "gps_lat": None,
        "gps_lon": None,
        # 2D box counts only
        "num_pedestrians": p,
        "num_vehicles": v,
        "num_cyclists": c,
        "closest_ped_distance_m": None,  # no 3D depth in BDD
        "time_of_day": TIME_OF_DAY_MAP.get(tod_raw),
        "weather": WEATHER_MAP.get(weather_raw),
    }


def _arrow_schema():
    import pyarrow as pa
    return pa.schema(
        [
            pa.field("frame_id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("device_id", pa.string()),
            pa.field("ts_ns", pa.int64()),
            pa.field("camera_name", pa.string()),
            pa.field("thumbnail_uri", pa.string()),
            pa.field("city", pa.string()),
            pa.field("ego_speed_mps", pa.float32()),
            pa.field("ego_heading_deg", pa.float32()),
            pa.field("ego_accel_x_mps2", pa.float32()),
            pa.field("ego_accel_y_mps2", pa.float32()),
            pa.field("ego_accel_z_mps2", pa.float32()),
            pa.field("gps_lat", pa.float64()),
            pa.field("gps_lon", pa.float64()),
            pa.field("num_pedestrians", pa.int32()),
            pa.field("num_vehicles", pa.int32()),
            pa.field("num_cyclists", pa.int32()),
            pa.field("closest_ped_distance_m", pa.float32()),
            pa.field("time_of_day", pa.string()),
            pa.field("weather", pa.string()),
        ]
    )


@app.command()
def main(
    splits: str = typer.Option(
        "train,val", help="Which BDD splits to ingest (comma-separated)."
    ),
    limit: int = typer.Option(0, help="Cap at N rows for smoke tests (0 = no cap)."),
    dry_run: bool = typer.Option(False, help="Print plan, don't read or write."),
):
    """Convert BDD100K labels to the shared parquet schema in S3."""
    load_dotenv()

    bucket_uri = os.environ.get("BUCKET_URI")
    prefix = os.environ.get("BDD100K_PREFIX", "bdd100k")
    thumbnails_prefix = os.environ.get("THUMBNAILS_PREFIX", "thumbnails")
    if not bucket_uri:
        console.print("[red]BUCKET_URI not set.[/red]")
        raise typer.Exit(2)

    labels_uri = uri_join(bucket_uri, prefix, "raw", "bdd100k_labels.zip")
    metadata_uri = uri_join(bucket_uri, prefix, "metadata", "bdd100k_metadata.parquet")
    thumb_base = uri_join(bucket_uri, thumbnails_prefix, "bdd100k")
    split_list = [s.strip() for s in splits.split(",")]

    console.rule("[bold]BDD100K labels → parquet")
    console.print(f"  labels zip:   {labels_uri}")
    console.print(f"  parquet sink: {metadata_uri}")
    console.print(f"  thumb base:   {thumb_base}")
    console.print(f"  splits:       {split_list}")

    if dry_run:
        console.print("[yellow]dry-run[/yellow]")
        return

    # 1. Read the labels zip from S3 into memory (it's ~147 MB).
    fs, path = get_fs(labels_uri)
    with fs.open(path, "rb") as f:
        zip_bytes = f.read()

    records: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for split in split_list:
            # BDD label files: `bdd100k_labels_images_train.json` etc.
            matches = [n for n in names if split in n and n.endswith(".json")]
            if not matches:
                console.print(f"[yellow]no label file for split {split!r} in zip[/yellow]")
                continue
            for member in matches:
                console.print(f"  [dim]parsing {member}...[/dim]")
                with zf.open(member) as jf:
                    entries = json.load(jf)
                for entry in entries:
                    records.append(_frame_record(entry, thumb_base))
                    if limit and len(records) >= limit:
                        break
                if limit and len(records) >= limit:
                    break
            if limit and len(records) >= limit:
                break

    if not records:
        console.print("[red]No records produced — check splits + zip contents.[/red]")
        raise typer.Exit(4)

    # 2. Write parquet to S3.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(records, schema=_arrow_schema())
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)

    out_fs, out_path = get_fs(metadata_uri)
    try:
        out_fs.makedirs(out_path.rsplit("/", 1)[0], exist_ok=True)
    except (AttributeError, NotImplementedError):
        pass
    with out_fs.open(out_path, "wb") as f:
        f.write(buf.getvalue())

    t = Table(title="BDD100K labels summary")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("rows", f"{len(records):,}")
    t.add_row("parquet", metadata_uri)
    t.add_row("splits", ", ".join(split_list))
    console.print(t)


if __name__ == "__main__":
    app()
