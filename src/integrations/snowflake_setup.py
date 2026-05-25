"""Snowflake live integration — read av-triage parquet directly from S3 via external tables.

(upgraded from stub to live, 2026-05-21).

ARCHITECTURE
  Our metadata parquet at s3://YOUR_BUCKET/{dataset}/metadata/*.parquet
  is read by Snowflake EXTERNAL TABLES via a Storage Integration — zero data
  movement, zero ETL, fully customer-side.

THREE-COMMAND FLOW (idempotent; safe to re-run):

  1. python -m src.integrations.snowflake_setup init
     - Creates DB/schema/storage-integration
     - Prints STORAGE_AWS_IAM_USER_ARN + STORAGE_AWS_EXTERNAL_ID
     - Prints the AWS IAM JSON to paste (trust policy + read policy)

  2. (Manual ~2 min in AWS console using the JSON from step 1)

  3. python -m src.integrations.snowflake_setup finalize --iam-role-arn <ARN>
     - Updates the integration with the role ARN
     - Creates STAGE + EXTERNAL TABLES for waymo + bdd100k
     - Runs a smoke SELECT to verify

  4. python -m src.integrations.snowflake_setup demo
     - Cross-dataset query: 'most dense pedestrian frames across Waymo + BDD'
     - Prints results + saves screenshot-ready output to docs/snowflake_demo.txt

CREDENTIALS (set in .env; NEVER commit)
  SNOWFLAKE_ACCOUNT    (e.g., ABC12345-XY67890 — your Snowflake account identifier)
  SNOWFLAKE_USER
  SNOWFLAKE_PASSWORD
  SNOWFLAKE_ROLE       (default: ACCOUNTADMIN — required for CREATE STORAGE INTEGRATION)
  SNOWFLAKE_WAREHOUSE  (default: COMPUTE_WH — Snowflake creates this on trial accounts)

The full credentials reference is at secrets/accounts/snowflake.md (gitignored).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# Names used inside Snowflake. Idempotent — re-running CREATE OR REPLACE is safe.
DB_NAME = "AV_TRIAGE"
SCHEMA_NAME = "PUBLIC"
INTEGRATION_NAME = "AV_TRIAGE_S3_INT"
STAGE_NAME = "AV_TRIAGE_S3_STAGE"
WAREHOUSE_NAME_DEFAULT = "COMPUTE_WH"


def _connect():
    """Open a Snowflake connection from env credentials. Lazy-imports the driver."""
    try:
        import snowflake.connector  # noqa: F401
    except ImportError:
        console.print("[red]snowflake-connector-python not installed.[/red] Run `uv sync`.")
        raise typer.Exit(1)

    import snowflake.connector

    load_dotenv()
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    role = os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", WAREHOUSE_NAME_DEFAULT)

    missing = [k for k, v in {
        "SNOWFLAKE_ACCOUNT": account,
        "SNOWFLAKE_USER": user,
        "SNOWFLAKE_PASSWORD": password,
    }.items() if not v]
    if missing:
        console.print(
            f"[red]Missing env vars: {', '.join(missing)}[/red] "
            "— add them to .env (see secrets/accounts/snowflake.md template)."
        )
        raise typer.Exit(2)

    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        role=role,
        warehouse=warehouse,
    )


def _exec(con, sql: str, params: tuple = ()):
    """Run SQL, return list of dict rows."""
    cur = con.cursor()
    try:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        return [dict(zip(cols, r, strict=True)) for r in rows]
    finally:
        cur.close()


def _exec_many(con, statements: list[str]):
    """Run multiple statements sequentially (Snowflake driver doesn't accept multi-stmt by default)."""
    for s in statements:
        s = s.strip()
        if not s:
            continue
        _exec(con, s)


@app.command()
def init():
    """Step 1: create DB/schema/storage-integration, print AWS IAM ingredients."""
    bucket = os.environ.get("BUCKET_URI", "s3://YOUR_BUCKET").replace("s3://", "")

    con = _connect()
    try:
        console.rule("[bold]1/3 create database + schema + warehouse")
        _exec_many(con, [
            f"CREATE WAREHOUSE IF NOT EXISTS {WAREHOUSE_NAME_DEFAULT} "
            f"WITH WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE",
            f"CREATE DATABASE IF NOT EXISTS {DB_NAME}",
            f"USE DATABASE {DB_NAME}",
            f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}",
            f"USE SCHEMA {SCHEMA_NAME}",
        ])
        console.print(f"  → {DB_NAME}.{SCHEMA_NAME} ready")

        console.rule("[bold]2/3 create storage integration")
        _exec(con, f"""
            CREATE OR REPLACE STORAGE INTEGRATION {INTEGRATION_NAME}
              TYPE = EXTERNAL_STAGE
              STORAGE_PROVIDER = 'S3'
              ENABLED = TRUE
              STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::000000000000:role/PLACEHOLDER'
              STORAGE_ALLOWED_LOCATIONS = ('s3://{bucket}/')
        """)
        console.print(f"  → integration {INTEGRATION_NAME} created (with placeholder role)")

        rows = _exec(con, f"DESC INTEGRATION {INTEGRATION_NAME}")
        info = {r["property"]: r["property_value"] for r in rows}
        iam_user_arn = info.get("STORAGE_AWS_IAM_USER_ARN")
        external_id = info.get("STORAGE_AWS_EXTERNAL_ID")

        console.rule("[bold]3/3 NEXT: create AWS IAM role")
        console.print("\nPaste this trust policy into a new AWS IAM role:\n")
        trust = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": iam_user_arn},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"sts:ExternalId": external_id}}
            }]
        }
        console.print(json.dumps(trust, indent=2))

        console.print("\nAnd this inline policy on that role:\n")
        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Resource": f"arn:aws:s3:::{bucket}/*/metadata/*"
            }, {
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": ["*/metadata/*"]}}
            }]
        }
        console.print(json.dumps(policy, indent=2))

        console.print(
            f"\n[bold]Then:[/bold] python -m src.integrations.snowflake_setup finalize "
            f"--iam-role-arn arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:role/<your-role-name>\n"
        )
    finally:
        con.close()


@app.command()
def finalize(iam_role_arn: str = typer.Option(..., help="ARN of the role you just created in AWS.")):
    """Step 3: bind the role + create stage + external tables + smoke SELECT."""
    bucket = os.environ.get("BUCKET_URI", "s3://YOUR_BUCKET").replace("s3://", "")

    con = _connect()
    try:
        _exec(con, f"USE DATABASE {DB_NAME}")
        _exec(con, f"USE SCHEMA {SCHEMA_NAME}")
        _exec(con, f"USE WAREHOUSE {WAREHOUSE_NAME_DEFAULT}")

        console.rule("[bold]1/4 bind IAM role to integration")
        _exec(con, f"""
            ALTER STORAGE INTEGRATION {INTEGRATION_NAME}
            SET STORAGE_AWS_ROLE_ARN = '{iam_role_arn}'
        """)
        console.print(f"  → bound {iam_role_arn}")

        console.rule("[bold]2/4 create stage")
        _exec(con, f"""
            CREATE OR REPLACE STAGE {STAGE_NAME}
              STORAGE_INTEGRATION = {INTEGRATION_NAME}
              URL = 's3://{bucket}/'
              FILE_FORMAT = (TYPE = PARQUET)
        """)
        console.print(f"  → stage {STAGE_NAME} on s3://{bucket}/")

        console.rule("[bold]3/4 create external tables (one per dataset)")
        # Explicit column definitions match the Decision-15 parquet schema.
        # Without these, Snowflake exposes only a `VALUE` VARIANT and queries
        # require VALUE:col::TYPE notation — ugly for a demo.
        decision_15_cols = """
                frame_id              STRING  AS (VALUE:frame_id::STRING),
                dataset               STRING  AS (VALUE:dataset::STRING),
                device_id             STRING  AS (VALUE:device_id::STRING),
                ts_ns                 BIGINT  AS (VALUE:ts_ns::BIGINT),
                camera_name           STRING  AS (VALUE:camera_name::STRING),
                thumbnail_uri         STRING  AS (VALUE:thumbnail_uri::STRING),
                city                  STRING  AS (VALUE:city::STRING),
                ego_speed_mps         DOUBLE  AS (VALUE:ego_speed_mps::DOUBLE),
                ego_heading_deg       DOUBLE  AS (VALUE:ego_heading_deg::DOUBLE),
                ego_accel_x_mps2      DOUBLE  AS (VALUE:ego_accel_x_mps2::DOUBLE),
                ego_accel_y_mps2      DOUBLE  AS (VALUE:ego_accel_y_mps2::DOUBLE),
                ego_accel_z_mps2      DOUBLE  AS (VALUE:ego_accel_z_mps2::DOUBLE),
                gps_lat               DOUBLE  AS (VALUE:gps_lat::DOUBLE),
                gps_lon               DOUBLE  AS (VALUE:gps_lon::DOUBLE),
                num_pedestrians       INTEGER AS (VALUE:num_pedestrians::INTEGER),
                num_vehicles          INTEGER AS (VALUE:num_vehicles::INTEGER),
                num_cyclists          INTEGER AS (VALUE:num_cyclists::INTEGER),
                closest_ped_distance_m DOUBLE AS (VALUE:closest_ped_distance_m::DOUBLE),
                time_of_day           STRING  AS (VALUE:time_of_day::STRING),
                weather               STRING  AS (VALUE:weather::STRING)
        """
        for ds, prefix in (("waymo", "waymo"), ("bdd100k", "bdd100k")):
            table = f"{ds.upper()}_METADATA"
            _exec(con, f"""
                CREATE OR REPLACE EXTERNAL TABLE {table} (
                {decision_15_cols}
                )
                  WITH LOCATION = @{STAGE_NAME}/{prefix}/metadata/
                  AUTO_REFRESH = FALSE
                  FILE_FORMAT = (TYPE = PARQUET)
            """)
            console.print(f"  → {table}")

        console.rule("[bold]4/4 smoke SELECT (counts per dataset)")
        try:
            rows = _exec(con, """
                SELECT 'waymo' AS dataset, COUNT(*) AS n FROM WAYMO_METADATA
                UNION ALL
                SELECT 'bdd100k', COUNT(*) FROM BDD100K_METADATA
            """)
        except Exception as e:
            console.print(f"[yellow]Smoke query failed (may be no parquet yet): {e}[/yellow]")
            return

        t = Table(title="Snowflake external table row counts")
        t.add_column("dataset")
        t.add_column("rows", justify="right")
        for r in rows:
            t.add_row(str(r["DATASET"]), f"{int(r['N']):,}")
        console.print(t)
        console.print("\n[green]Snowflake integration LIVE.[/green]")
    finally:
        con.close()


@app.command()
def demo(out: Path = typer.Option(Path("docs/snowflake_demo.txt"), help="Demo output path.")):
    """Step 4: cross-dataset demo query. Saves a paste-ready transcript."""
    con = _connect()
    try:
        _exec(con, f"USE DATABASE {DB_NAME}")
        _exec(con, f"USE SCHEMA {SCHEMA_NAME}")
        _exec(con, f"USE WAREHOUSE {WAREHOUSE_NAME_DEFAULT}")

        demos = [
            (
                "Cross-dataset cheap aggregate",
                "How many frames per dataset?",
                """
                SELECT dataset, COUNT(*) AS n
                FROM (
                  SELECT * FROM WAYMO_METADATA
                  UNION ALL SELECT * FROM BDD100K_METADATA
                )
                GROUP BY dataset
                ORDER BY n DESC
                """,
            ),
            (
                "Topic-derived filter (Decision 17 hybrid story)",
                "Dense pedestrian frames at night across BDD100K",
                """
                SELECT dataset, time_of_day, num_pedestrians, frame_id, thumbnail_uri
                FROM BDD100K_METADATA
                WHERE time_of_day = 'night' AND num_pedestrians >= 3
                ORDER BY num_pedestrians DESC
                LIMIT 10
                """,
            ),
            (
                "Cross-dataset unified schema (Decision 15)",
                "Top weather distribution across both datasets",
                """
                SELECT dataset, weather, COUNT(*) AS n
                FROM (
                  SELECT dataset, weather FROM WAYMO_METADATA
                  UNION ALL SELECT dataset, weather FROM BDD100K_METADATA
                )
                WHERE weather IS NOT NULL
                GROUP BY dataset, weather
                ORDER BY dataset, n DESC
                """,
            ),
        ]

        transcript = []
        for title, q_desc, sql in demos:
            console.rule(f"[bold]{title}")
            console.print(f"[dim]{q_desc}[/dim]")
            console.print(sql.strip())
            try:
                rows = _exec(con, sql)
                tbl = Table()
                if rows:
                    for c in rows[0].keys():
                        tbl.add_column(c)
                    for r in rows[:20]:
                        tbl.add_row(*[str(v) for v in r.values()])
                console.print(tbl)
                transcript.append({"title": title, "question": q_desc, "sql": sql.strip(), "rows": rows[:20]})
            except Exception as e:
                console.print(f"[yellow]demo skipped: {e}[/yellow]")
                transcript.append({"title": title, "error": str(e)})

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(transcript, indent=2, default=str))
        console.print(f"\n[green]Wrote demo transcript → {out}[/green]")
    finally:
        con.close()


if __name__ == "__main__":
    app()
