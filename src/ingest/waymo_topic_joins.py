"""Waymo BigQuery cross-table topic joins → fills the 13 NULL columns in Decision 15.

Per Decision 16 open-work item #1. This module owns the SQL + Python aggregation
that pulls scene attributes (time_of_day, weather), perception counts/distances
(from lidar_box), and ego kinematics (from vehicle_pose) into one keyed dict
indexed by (segment_context_name, frame_timestamp_micros).

USAGE (from waymo_segments.py, after frame keys are fetched):

    from src.ingest.waymo_topic_joins import fetch_topic_data
    topic_lookup = fetch_topic_data(client, bq, waymo_ds, chosen_segments)
    # then for each frame:
    topic_row = topic_lookup.get((segment, ts_micros), {})
    record.update(topic_row)

DESIGN NOTES — what's fillable from Waymo Open Dataset v2 (honest):

  ✅ time_of_day, weather              from stats table
  ✅ num_pedestrians, num_vehicles,
     num_cyclists, closest_ped_distance_m  from lidar_box (LiDAR detections aggregated per frame)
  ⚠️ ego_speed_mps, ego_heading_deg    derived from vehicle_pose (window function over consecutive transforms)
  ❌ ego_accel_x/y/z_mps2              NOT in Waymo v2 (no IMU table — left NULL)
  ❌ gps_lat, gps_lon                  NOT in Waymo v2 (transform is local frame, no published WGS84 — left NULL)

Waymo v2 publicly drops IMU + GPS that production fleets typically capture. Most
production robot/AV MCAPs include these; Waymo's public release is a subset. The
remaining 4 NULL columns are a dataset limitation, not a code issue — they will
populate cleanly on full-sensor data.

SCHEMA ASSUMPTIONS (verify with `verify_waymo_schema.py` before relying):

  - `stats` table has `stats.time_of_day` (str enum: DAY/NIGHT/DAWN_DUSK) and
    `stats.weather` (str enum: SUNNY/RAIN/etc), keyed by segment + frame_ts_micros.
  - `lidar_box` table has repeated `lidar_box` STRUCT with `type` (int enum),
    `center_x/y/z`, etc. — one row per detection.
  - `vehicle_pose` table has `vehicle_pose.transform` as repeated FLOAT64 (16 elements,
    row-major 4x4 transform matrix), keyed by segment + frame_ts_micros.

If any of those assumptions is wrong, the per-table CTE will fail with a clear column-
not-found error, and the right fix is to inspect the actual schema via
INFORMATION_SCHEMA.COLUMNS and adjust the CTE.
"""

from __future__ import annotations

from typing import Any

# Waymo LiDAR box type enum (per Waymo Open Dataset v2 proto)
WAYMO_LIDAR_TYPE_VEHICLE = 1
WAYMO_LIDAR_TYPE_PEDESTRIAN = 2
WAYMO_LIDAR_TYPE_SIGN = 3
WAYMO_LIDAR_TYPE_CYCLIST = 4


def topic_join_sql(waymo_ds: str) -> str:
    """Returns the parameterized SQL that joins scene stats + lidar perception + ego pose.

    Parameters expected at execution:
      @segments : ARRAY<STRING> — segment_context_name list

    Returns rows keyed by (device_id, ts_micros) with all topic-derived columns.
    """
    return f"""
    WITH
      -- 1. Scene attributes per frame
      scene_stats AS (
        SELECT
          s.key.segment_context_name AS device_id,
          s.key.frame_timestamp_micros AS ts_micros,
          LOWER(IFNULL(s.stats.time_of_day, '')) AS time_of_day,
          LOWER(IFNULL(s.stats.weather, ''))     AS weather
        FROM `{waymo_ds}.stats` AS s
        WHERE s.key.segment_context_name IN UNNEST(@segments)
      ),

      -- 2. LiDAR perception aggregated per frame
      --    `lidar_box` is per-detection; unnest implicitly by row, group by frame.
      lidar_objects AS (
        SELECT
          lb.key.segment_context_name AS device_id,
          lb.key.frame_timestamp_micros AS ts_micros,
          COUNTIF(lb.type = {WAYMO_LIDAR_TYPE_PEDESTRIAN}) AS num_pedestrians,
          COUNTIF(lb.type = {WAYMO_LIDAR_TYPE_VEHICLE})    AS num_vehicles,
          COUNTIF(lb.type = {WAYMO_LIDAR_TYPE_CYCLIST})    AS num_cyclists,
          MIN(
            IF(
              lb.type = {WAYMO_LIDAR_TYPE_PEDESTRIAN},
              SQRT(POW(lb.box.center_x, 2) + POW(lb.box.center_y, 2) + POW(lb.box.center_z, 2)),
              CAST(NULL AS FLOAT64)
            )
          ) AS closest_ped_distance_m
        FROM `{waymo_ds}.lidar_box` AS lb
        WHERE lb.key.segment_context_name IN UNNEST(@segments)
        GROUP BY device_id, ts_micros
      ),

      -- 3. Ego pose + LAG to compute speed and heading
      --    transform[OFFSET(3)] = tx, [OFFSET(7)] = ty, [OFFSET(11)] = tz (translation column)
      --    yaw from atan2(R[1,0], R[0,0]) → atan2(T[OFFSET(4)], T[OFFSET(0)])
      pose_window AS (
        SELECT
          vp.key.segment_context_name AS device_id,
          vp.key.frame_timestamp_micros AS ts_micros,
          vp.vehicle_pose.transform AS T,
          LAG(vp.vehicle_pose.transform) OVER w AS T_prev,
          LAG(vp.key.frame_timestamp_micros) OVER w AS prev_ts_micros
        FROM `{waymo_ds}.vehicle_pose` AS vp
        WHERE vp.key.segment_context_name IN UNNEST(@segments)
        WINDOW w AS (
          PARTITION BY vp.key.segment_context_name
          ORDER BY vp.key.frame_timestamp_micros
        )
      ),
      ego_kinematics AS (
        SELECT
          device_id,
          ts_micros,
          ATAN2(T[OFFSET(4)], T[OFFSET(0)]) * 180.0 / ACOS(-1) AS ego_heading_deg,
          CASE
            WHEN T_prev IS NULL OR prev_ts_micros IS NULL THEN CAST(NULL AS FLOAT64)
            ELSE SAFE_DIVIDE(
              SQRT(
                POW(T[OFFSET(3)]  - T_prev[OFFSET(3)],  2) +
                POW(T[OFFSET(7)]  - T_prev[OFFSET(7)],  2) +
                POW(T[OFFSET(11)] - T_prev[OFFSET(11)], 2)
              ),
              (ts_micros - prev_ts_micros) / 1e6
            )
          END AS ego_speed_mps
        FROM pose_window
      )

    SELECT
      f.device_id,
      f.ts_micros,
      ek.ego_speed_mps,
      ek.ego_heading_deg,
      lo.num_pedestrians,
      lo.num_vehicles,
      lo.num_cyclists,
      lo.closest_ped_distance_m,
      NULLIF(ss.time_of_day, '') AS time_of_day,
      NULLIF(ss.weather, '')     AS weather
    FROM (
      SELECT DISTINCT
        s.key.segment_context_name AS device_id,
        s.key.frame_timestamp_micros AS ts_micros
      FROM `{waymo_ds}.stats` AS s
      WHERE s.key.segment_context_name IN UNNEST(@segments)
    ) AS f
    LEFT JOIN scene_stats    AS ss USING (device_id, ts_micros)
    LEFT JOIN lidar_objects  AS lo USING (device_id, ts_micros)
    LEFT JOIN ego_kinematics AS ek USING (device_id, ts_micros)
    """


def fetch_topic_data(client, bq, waymo_ds: str, segments: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    """Run the JOIN and return a lookup keyed by (device_id, ts_micros).

    Args:
        client: instantiated BigQuery client
        bq: the google.cloud.bigquery module (for QueryJobConfig + parameter types)
        waymo_ds: dotted path to the Waymo dataset
        segments: list of segment_context_name strings to limit the scan

    Returns:
        {(device_id, ts_micros): {col_name: value, ...}, ...}
        - Returns empty dict if no rows.
        - Always returns 8 fillable columns; the remaining 4 (ego_accel_*, gps_*)
          are NULL and should be set in the caller's record assembly.
    """
    sql = topic_join_sql(waymo_ds)
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter("segments", "STRING", segments),
        ]
    )
    job = client.query(sql, job_config=job_config)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in job.result():
        out[(row.device_id, int(row.ts_micros))] = {
            "ego_speed_mps": row.ego_speed_mps,
            "ego_heading_deg": row.ego_heading_deg,
            "num_pedestrians": int(row.num_pedestrians) if row.num_pedestrians is not None else None,
            "num_vehicles": int(row.num_vehicles) if row.num_vehicles is not None else None,
            "num_cyclists": int(row.num_cyclists) if row.num_cyclists is not None else None,
            "closest_ped_distance_m": row.closest_ped_distance_m,
            "time_of_day": row.time_of_day,
            "weather": row.weather,
        }
    return out


# Columns this module fills (vs leaves NULL — see module docstring)
COLUMNS_FILLED = [
    "ego_speed_mps",
    "ego_heading_deg",
    "num_pedestrians",
    "num_vehicles",
    "num_cyclists",
    "closest_ped_distance_m",
    "time_of_day",
    "weather",
]

COLUMNS_LEFT_NULL = [
    "ego_accel_x_mps2",  # no IMU in Waymo v2
    "ego_accel_y_mps2",
    "ego_accel_z_mps2",
    "gps_lat",            # no GPS in Waymo v2 (transform is local frame)
    "gps_lon",
]
