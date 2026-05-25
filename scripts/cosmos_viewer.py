"""Cosmos-Reason1 descriptions viewer — thumbnail + caption grid.

Reads cosmos_descriptions.parquet (frame_id, dataset, camera_name, thumbnail_uri, description),
fetches each thumbnail from S3, base64-embeds it, and writes a self-contained HTML page.

Open in a browser:  open dashboard/cosmos_viewer.html

CLI:
    python -m scripts.cosmos_viewer                              # all 500
    python -m scripts.cosmos_viewer --limit 50                   # first 50
    python -m scripts.cosmos_viewer --parquet s3://.../foo.parquet
"""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.cloud import get_fs

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    parquet: str = typer.Option("eval/cosmos_descriptions.parquet", help="Local path or s3:// URI"),
    out: str = typer.Option("dashboard/cosmos_viewer.html"),
    limit: int = typer.Option(0, help="0 = all rows"),
):
    load_dotenv()
    import duckdb

    if parquet.startswith("s3://"):
        from src.cloud import duckdb_with_s3
        con = duckdb_with_s3(duckdb.connect())
        src = f"read_parquet('{parquet}')"
    else:
        con = duckdb.connect()
        src = f"read_parquet('{parquet}')"

    sql = f"SELECT frame_id, dataset, camera_name, thumbnail_uri, description FROM {src}"
    if limit:
        sql += f" LIMIT {limit}"
    rows = con.execute(sql).fetchall()
    console.print(f"loaded {len(rows)} rows from {parquet}")

    cards = []
    for i, (fid, ds, cam, thumb_uri, desc) in enumerate(rows, 1):
        try:
            fs, path = get_fs(thumb_uri)
            with fs.open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img = f'<img src="data:image/jpeg;base64,{b64}" alt="">'
        except Exception as e:
            img = f'<div class="err">image load failed: {html.escape(str(e))}</div>'
        cards.append(
            f'<div class="card">'
            f'  <div class="img">{img}</div>'
            f'  <div class="meta"><span class="ds">{html.escape(ds)}</span> · '
            f'<span class="cam">{html.escape(cam)}</span></div>'
            f'  <div class="fid">{html.escape(fid)}</div>'
            f'  <div class="desc">{html.escape(desc)}</div>'
            f"</div>"
        )
        if i % 25 == 0:
            console.print(f"  {i}/{len(rows)}")

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cosmos-Reason1 captions ({len(rows)})</title>
<style>
  body {{ font:14px/1.5 -apple-system,system-ui,sans-serif; margin:0; padding:24px; background:#fafbfc; color:#14181f; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .sub {{ color:#6b7280; margin-bottom:20px; font-size:13px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:18px; }}
  .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }}
  .img img {{ width:100%; height:auto; border-radius:4px; background:#000; }}
  .meta {{ font:11px ui-monospace,monospace; color:#6b7280; margin-top:8px; }}
  .ds {{ font-weight:600; color:#0a7; }}
  .fid {{ font:11px ui-monospace,monospace; color:#6b7280; margin-top:2px; word-break:break-all; }}
  .desc {{ margin-top:10px; line-height:1.45; }}
  .err {{ color:#c40; font:12px ui-monospace,monospace; padding:20px; background:#fff5f0; }}
</style></head><body>
<h1>Cosmos-Reason1-7B scene captions</h1>
<div class="sub">{len(rows)} thumbnails captioned by NVIDIA Cosmos-Reason1-7B (vision-language model trained on physical-AI data). Each caption was generated from the thumbnail + the prompt "Describe this driving scene in 2-3 sentences...".</div>
<div class="grid">
{chr(10).join(cards)}
</div>
</body></html>"""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    console.print(f"[green]✓ wrote {out_path} ({out_path.stat().st_size//1024} KB)[/green]")
    console.print(f"  open {out_path.absolute()}")


if __name__ == "__main__":
    app()
