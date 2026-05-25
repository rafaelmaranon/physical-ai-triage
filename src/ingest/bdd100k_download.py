"""BDD100K ingest — archive.org mirror → S3 raw, then 1Hz frame extraction.

:
  - Skip the Berkeley academic portal (unreachable + requires registration).
  - Source: archive.org mirror, two zips:
      * https://archive.org/download/bdd100k/bdd100k_images.zip   (~6.5 GB, key-frame JPEGs)
      * https://archive.org/download/bdd100k/bdd100k_labels.zip   (~147 MB, JSON labels)
  - Stream both directly to s3://{BUCKET_URI}/{BDD100K_PREFIX}/raw/ — no local-disk hop.

This module ONLY mirrors the raw zips to S3. Label parsing into the shared parquet
schema (per Decision 15) lives in `src/ingest/bdd100k_labels.py`. Frame extraction
from the images zip is a separate Brev step.
"""

from __future__ import annotations

import os
import sys
import urllib.request

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.cloud import get_fs, join as uri_join

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


ARTIFACTS = {
    "images": "https://archive.org/download/bdd100k/bdd100k_images.zip",
    "labels": "https://archive.org/download/bdd100k/bdd100k_labels.zip",
}


def _stream_to_s3(url: str, dest_uri: str, chunk: int = 8 * 1024 * 1024) -> int:
    """Stream a URL to an fsspec destination URI without buffering to disk."""
    fs, path = get_fs(dest_uri)
    total = 0
    with urllib.request.urlopen(url) as src, fs.open(path, "wb") as dst:
        while True:
            buf = src.read(chunk)
            if not buf:
                break
            dst.write(buf)
            total += len(buf)
            print(f"  {total / 1e9:.2f} GB", end="\r", file=sys.stderr)
    return total


@app.command()
def main(
    artifact: str = typer.Option(
        "both", help="Which artifact to mirror: images, labels, or both."
    ),
    dry_run: bool = typer.Option(False, help="Print the plan, don't transfer."),
):
    """Mirror BDD100K zips from archive.org to `{BUCKET_URI}/{BDD100K_PREFIX}/raw/`."""
    load_dotenv()

    bucket_uri = os.environ.get("BUCKET_URI")
    prefix = os.environ.get("BDD100K_PREFIX", "bdd100k")
    if not bucket_uri:
        console.print("[red]BUCKET_URI not set.[/red] Copy .env.example to .env.")
        raise typer.Exit(2)

    targets = list(ARTIFACTS.keys()) if artifact == "both" else [artifact]
    if any(a not in ARTIFACTS for a in targets):
        console.print(f"[red]Unknown artifact {artifact!r}; choose: images, labels, both.[/red]")
        raise typer.Exit(3)

    console.rule("[bold]BDD100K archive.org → S3")
    for name in targets:
        url = ARTIFACTS[name]
        dest = uri_join(bucket_uri, prefix, "raw", f"bdd100k_{name}.zip")
        console.print(f"  [cyan]{name}[/cyan]: {url}\n        → {dest}")

    if dry_run:
        console.print("[yellow]dry-run: skipping transfers[/yellow]")
        return

    for name in targets:
        url = ARTIFACTS[name]
        dest = uri_join(bucket_uri, prefix, "raw", f"bdd100k_{name}.zip")
        console.print(f"[bold]streaming {name}...[/bold]")
        size = _stream_to_s3(url, dest)
        console.print(f"  done — {size / 1e9:.2f} GB at {dest}")


if __name__ == "__main__":
    app()
