"""Cloud-agnostic storage helper (per #6, #11).

Single entry point: `get_fs(uri)` returns an fsspec filesystem plus the path
stripped of its scheme. Use the same code path for s3://, gs://, and local
file:// targets — backend chosen by URI scheme.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import fsspec


def get_fs(uri: str) -> tuple[fsspec.AbstractFileSystem, str]:
    """Return `(filesystem, path_without_scheme)` for any storage URI.

    Examples:
        >>> fs, path = get_fs("s3://YOUR_BUCKET/waymo/segments.parquet")
        # fs is s3fs.S3FileSystem; path is "YOUR_BUCKET/waymo/segments.parquet"

        >>> fs, path = get_fs("gs://bucket/key")
        # fs is gcsfs.GCSFileSystem; path is "bucket/key"

        >>> fs, path = get_fs("/tmp/local/file.parquet")
        # fs is local; path is "/tmp/local/file.parquet"
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"

    if scheme == "file":
        return fsspec.filesystem("file"), uri

    fs = fsspec.filesystem(scheme)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return fs, f"{bucket}/{key}" if key else bucket


def join(base_uri: str, *parts: str) -> str:
    """Join a base URI with one or more path parts, preserving the scheme."""
    return base_uri.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)


def duckdb_with_s3(con):
    """Configure a DuckDB connection to read s3:// paths via httpfs.

    Picks creds from the standard AWS env / ~/.aws/credentials chain via boto3.
    No-op if AWS_REGION is unset (i.e. user only uses GCS).
    """
    con.execute("INSTALL httpfs; LOAD httpfs;")
    region = os.environ.get("AWS_REGION")
    if not region:
        return con

    try:
        import boto3
    except ImportError:
        return con

    creds = boto3.Session().get_credentials()
    if creds is None:
        return con
    frozen = creds.get_frozen_credentials()
    con.execute(f"SET s3_region='{region}';")
    con.execute(f"SET s3_access_key_id='{frozen.access_key}';")
    con.execute(f"SET s3_secret_access_key='{frozen.secret_key}';")
    if frozen.token:
        con.execute(f"SET s3_session_token='{frozen.token}';")
    return con
