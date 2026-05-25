"""Waymo Open Perception v2 ingest — BigQuery source → S3 parquet sink.

#6, #15:
  - SOURCE: BigQuery public dataset `bigquery-public-data.waymo_open_dataset_v_2_0_0`.
  - SINK:   parquet to {BUCKET_URI}/{WAYMO_PREFIX}/metadata/waymo_metadata.parquet.
  - SCHEMA: per Decision 15 — frame_id + keys + topic-derived columns (ego/IMU/GPS/perception).
            All 14 topic-derived columns share names across datasets so DuckDB queries
            stay dataset-agnostic. NULL is acceptable where a column doesn't apply.

This step writes ONLY metadata. Frame images are mirrored to S3 in a downstream
step run from the GPU host.

OPEN WORK (TODO Decision 15):
  - Cross-table BQ joins for ego_pose / imu / gps / perception are stubbed as NULL.
  - Populating them needs joins across
    bigquery-public-data.waymo_open_dataset_v_2_0_0.{vehicle_pose, vehicle_imu, vehicle_gps,
    lidar_box, camera_box} by (segment_context_name, frame_timestamp_micros).
  - Schema is correct already — fill values in place when the cross-table joins land.
"""

from __future__ import annotations

import io
import os

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.cloud import get_fs, join as uri_join
from src.frame_id import frame_id

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# Waymo location code → human-readable city slug (used for `city` column).
WAYMO_LOCATION_TO_CITY = {
    "location_sf": "sf",
    "location_phx": "phx",
    "location_other": "other",
}


def _require_bq():
    """Lazy-import google-cloud-bigquery so this module loads on any laptop."""
    try:
        from google.cloud import bigquery
    except ImportError as e:
        console.print(
            "[red]google-cloud-bigquery is not installed.[/red] Run `uv sync` first."
        )
        raise typer.Exit(1) from e
    return bigquery


@app.command()
def main(
    segments: int = typer.Option(None, help="Number of Waymo segments (env: WAYMO_SEGMENTS_COUNT)."),
    sample_hz: float = typer.Option(None, help="Sampling rate per camera (env: WAYMO_SAMPLING_HZ)."),
    cameras: str = typer.Option(
        "FRONT,FRONT_LEFT,FRONT_RIGHT,SIDE_LEFT,SIDE_RIGHT",
        help="Comma-separated camera names.",
    ),
    locations: str = typer.Option(
        "location_sf,location_phx",
        help="Comma-separated Waymo location codes.",
    ),
    dry_run: bool = typer.Option(False, help="Print plan but don't run BQ jobs or write to S3."),
):
    """Stage `--segments` Waymo segments as parquet at `{BUCKET_URI}/{WAYMO_PREFIX}/metadata/`."""
    load_dotenv()

    segments = segments or int(os.environ.get("WAYMO_SEGMENTS_COUNT", 150))
    sample_hz = sample_hz or float(os.environ.get("WAYMO_SAMPLING_HZ", 1.0))
    bucket_uri = os.environ.get("BUCKET_URI")
    waymo_prefix = os.environ.get("WAYMO_PREFIX", "waymo")
    thumbnails_prefix = os.environ.get("THUMBNAILS_PREFIX", "thumbnails")
    waymo_ds = os.environ.get(
        "WAYMO_BQ_DATASET", "bigquery-public-data.waymo_open_dataset_v_2_0_0"
    )

    if not bucket_uri:
        console.print("[red]BUCKET_URI not set.[/red] Copy .env.example to .env and fill in.")
        raise typer.Exit(2)

    location_list = [s.strip() for s in locations.split(",")]
    cam_list = [c.strip() for c in cameras.split(",")]

    metadata_uri = uri_join(bucket_uri, waymo_prefix, "metadata", "waymo_metadata.parquet")

    console.rule("[bold]1/3 plan")
    console.print(f"  BQ source:    {waymo_ds}")
    console.print(f"  locations:    {location_list}")
    console.print(f"  segments:     {segments}")
    console.print(f"  sample_hz:    {sample_hz}")
    console.print(f"  cameras:      {cam_list}")
    console.print(f"  parquet sink: {metadata_uri}")

    if dry_run:
        console.print("[yellow]dry-run: skipping all BQ + S3 calls[/yellow]")
        return

    bq = _require_bq()
    client = bq.Client()

    # 1. Pick segments — keep location for the `city` column on each frame row.
    console.rule("[bold]2/3 segment selection")
    segment_sql = f"""
        SELECT DISTINCT segment_context_name, location
        FROM `{waymo_ds}.camera_image`
        WHERE location IN UNNEST(@locations)
        ORDER BY segment_context_name
        LIMIT @n
    """
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter("locations", "STRING", location_list),
            bq.ScalarQueryParameter("n", "INT64", segments),
        ]
    )
    seg_rows = list(client.query(segment_sql, job_config=job_config).result())
    if not seg_rows:
        console.print(
            "[red]No segments returned — check ADC project and locations.[/red]"
        )
        raise typer.Exit(3)
    chosen = [r.segment_context_name for r in seg_rows]
    seg_to_city = {
        r.segment_context_name: WAYMO_LOCATION_TO_CITY.get(r.location, "unknown")
        for r in seg_rows
    }
    console.print(f"  → selected {len(chosen)} segments across {len(set(seg_to_city.values()))} cities")

    # 2. Frame-level metadata extract (1 Hz × N cameras).
    # TODO Decision 15: JOIN onto vehicle_pose / vehicle_imu / vehicle_gps / lidar_box
    # to populate ego_*, gps_*, num_*, closest_ped_distance_m columns.
    console.rule("[bold]3/3 frame metadata extract")
    metadata_sql = f"""
        SELECT
          ci.key.segment_context_name AS device_id,
          ci.key.frame_timestamp_micros * 1000 AS ts_ns,
          ci.key.camera_name AS camera_name
        FROM `{waymo_ds}.camera_image` AS ci
        WHERE ci.key.segment_context_name IN UNNEST(@segments)
          AND ci.key.camera_name IN UNNEST(@cameras)
          AND MOD(
            CAST(ci.key.frame_timestamp_micros / 1000000 AS INT64),
            CAST(1.0 / @hz AS INT64)
          ) = 0
    """
    job = client.query(
        metadata_sql,
        job_config=bq.QueryJobConfig(
            query_parameters=[
                bq.ArrayQueryParameter("segments", "STRING", chosen),
                bq.ArrayQueryParameter("cameras", "STRING", cam_list),
                bq.ScalarQueryParameter("hz", "FLOAT64", sample_hz),
            ]
        ),
    )
    frame_rows = list(job.result())
    if not frame_rows:
        console.print(
            "[yellow]No frames returned — check camera names against Waymo v2 enum.[/yellow]"
        )
        raise typer.Exit(4)
    console.print(f"  → {len(frame_rows):,} frame rows extracted")

    # 3. Build records — schema per Decision 15. Topic-derived columns NULL for now (TODO).
    thumb_base = uri_join(bucket_uri, thumbnails_prefix, "waymo")
    records = []
    for r in frame_rows:
        fid = frame_id("waymo", r.device_id, int(r.ts_ns), r.camera_name)
        records.append(
            {
                # Keys
                "frame_id": fid,
                "dataset": "waymo",
                "device_id": r.device_id,
                "ts_ns": int(r.ts_ns),
                "camera_name": r.camera_name,
                # Derived
                "thumbnail_uri": f"{thumb_base}/{fid}.jpg",
                "city": seg_to_city.get(r.device_id, "unknown"),
                # TODO Decision 15: populate from BQ topic joins
                "ego_speed_mps": None,
                "ego_heading_deg": None,
                "ego_accel_x_mps2": None,
                "ego_accel_y_mps2": None,
                "ego_accel_z_mps2": None,
                "gps_lat": None,
                "gps_lon": None,
                "num_pedestrians": None,
                "num_vehicles": None,
                "num_cyclists": None,
                "closest_ped_distance_m": None,
                "time_of_day": None,
                "weather": None,
            }
        )

    # 4. Write parquet to S3 via fsspec.
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_schema = pa.schema(
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
    table = pa.Table.from_pylist(records, schema=arrow_schema)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)

    fs, path = get_fs(metadata_uri)
    parent = path.rsplit("/", 1)[0]
    try:
        fs.makedirs(parent, exist_ok=True)
    except (AttributeError, NotImplementedError):
        pass  # S3 has no real dirs
    with fs.open(path, "wb") as f:
        f.write(buf.getvalue())

    # Also drop the segment manifest as a sidecar
    seg_manifest_uri = uri_join(bucket_uri, waymo_prefix, "metadata", "segments.txt")
    seg_fs, seg_path = get_fs(seg_manifest_uri)
    with seg_fs.open(seg_path, "w") as f:
        f.write("\n".join(chosen) + "\n")

    # Summary
    t = Table(title="Waymo ingest summary")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("segments selected", str(len(chosen)))
    t.add_row("frames staged", f"{len(records):,}")
    t.add_row("metadata parquet", metadata_uri)
    t.add_row("segment manifest", seg_manifest_uri)
    t.add_row("thumbnail prefix", thumb_base)
    t.add_row("schema version", "Decision 15 (topic columns NULL pending BQ joins)")
    console.print(t)


if __name__ == "__main__":
    app()
