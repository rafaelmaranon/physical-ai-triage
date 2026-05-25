# Snowflake — live integration

This project's metadata parquet on S3 is queryable from Snowflake via **External Tables** + a **Storage Integration**. Zero data movement: Snowflake reads the parquet directly. Re-runnable schema migrations are NOT needed — `ALTER EXTERNAL TABLE REFRESH` picks up new files as the ingest pipeline writes them.

> **Why this is in the architecture.** Data infra customers don't want to ETL parquet a second time just to query it. Snowflake (and Databricks, Athena, Trino) all read S3 parquet natively. The metadata layout below — one parquet per dataset under `metadata/`, deterministic `frame_id` join key, shared Decision-15 schema — is built so any of them works out of the box.

---

## TL;DR

```bash
# One-time setup (~5 min total):
python -m src.integrations.snowflake_setup init
# → prints AWS IAM trust + permission JSON; paste into AWS console (~2 min)
python -m src.integrations.snowflake_setup finalize --iam-role-arn arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:role/<your-role>
python -m src.integrations.snowflake_setup demo
```

After this, anyone with read access to the bucket can query the data from Snowflake:

```sql
SELECT dataset, weather, COUNT(*)
FROM (
  SELECT dataset, weather FROM WAYMO_METADATA
  UNION ALL SELECT dataset, weather FROM BDD100K_METADATA
)
WHERE weather IS NOT NULL
GROUP BY dataset, weather
ORDER BY dataset;
```

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│ Pipeline writes metadata parquet (Decision-15 schema)  │
│ → s3://YOUR_BUCKET/waymo/metadata/*.parquet     │
│ → s3://YOUR_BUCKET/bdd100k/metadata/*.parquet   │
└────────────────────────────────────────────────────────┘
                              │
                              │  (Snowflake EXTERNAL TABLE — no copy, no ETL)
                              ▼
┌────────────────────────────────────────────────────────┐
│  Snowflake account <your-account> / us-east-2           │
│  DB AV_TRIAGE / SCHEMA PUBLIC                            │
│  • WAYMO_METADATA   (external table)                     │
│  • BDD100K_METADATA (external table)                     │
└────────────────────────────────────────────────────────┘
                              │
                              │  ANY Snowflake client (BI tool, JupyterSQL, dbt, customer dashboards)
                              ▼
                       Customer SQL
```

Security: Snowflake assumes an AWS IAM role (configured below) — no long-lived keys leave Snowflake's environment, and the role's S3 permission is scoped to `*/metadata/*` (no thumbnails, no embeddings).

---

## Setup walkthrough

### 1. Credentials (one-time)

Save Snowflake credentials at `secrets/snowflake.md` (gitignored). Mirror into `.env`:

```
SNOWFLAKE_ACCOUNT=<your-snowflake-account>     # e.g. ABC12345-XY67890
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ROLE=ACCOUNTADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
```

### 2. `init` — create database + storage integration

```bash
python -m src.integrations.snowflake_setup init
```

This creates `AV_TRIAGE` DB, schema, warehouse, and a `STORAGE INTEGRATION` pointed at `s3://YOUR_BUCKET/`. It then prints two JSON blobs:

- A **trust policy** — Snowflake's IAM user that will assume your role
- A **permission policy** — what that role is allowed to do (read S3 metadata parquet only)

### 3. AWS console — create the role (~2 min)

Go to AWS Console → IAM → Roles → Create role → "Custom trust policy" → paste the trust policy from step 2 → on the next screen, add an inline policy with the permission JSON. Name the role `snowflake-av-triage-reader`. Copy the role ARN.

### 4. `finalize --iam-role-arn ...`

```bash
python -m src.integrations.snowflake_setup finalize \
  --iam-role-arn arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:role/snowflake-av-triage-reader
```

This binds the role to the integration, creates the stage, creates external tables for `WAYMO_METADATA` and `BDD100K_METADATA`, and runs a smoke `SELECT COUNT(*)` to verify.

### 5. `demo` — run three sample queries

```bash
python -m src.integrations.snowflake_setup demo
```

Runs three queries that exercise the cross-dataset story:

1. Per-dataset row counts
2. Topic-derived filter (`time_of_day = 'night' AND num_pedestrians >= 3`) — Decision 17 hybrid story
3. Cross-dataset weather distribution

Saves the transcript to [`docs/snowflake_demo.txt`](snowflake_demo.txt) for screenshot/post use.

---

## Refresh after new ingest

External tables don't auto-poll S3 by default. After a new parquet drop:

```sql
ALTER EXTERNAL TABLE WAYMO_METADATA REFRESH;
ALTER EXTERNAL TABLE BDD100K_METADATA REFRESH;
```

For production, enable SQS-based auto-refresh on the integration (one-line ALTER + an S3 event notification). Not required for the demo.

---

## Cost

- A `COUNT(*)` on the demo data uses <0.01 credit. Full eval queries: still <1 credit total.
- Costs depend on your org's Snowflake plan; the workload here is tiny.

---

## Why not Athena / Trino / Databricks SQL?

All four work the same way for this pipeline — the parquet is the contract, not the query engine. Snowflake is the chosen demo because:

1. Most "data stack" audiences default to it
2. Common in the robotics/AV data infra stack (alongside AWS + Datadog)
3. The trial credit makes it free to demonstrate end-to-end

A line in the README + this doc explicitly note the substitution path: *"Same external-table pattern works on Athena (`CREATE EXTERNAL TABLE` from Glue catalog) or Trino (`CREATE TABLE ... WITH (external_location = 's3://...')`)."*
