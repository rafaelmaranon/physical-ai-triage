"""Cosmos-Reason1 vs Cosmos-Reason2 side-by-side caption viewer.

Both models captioned the SAME 500 frames (same seed). This page renders the same
thumbnail with R1 caption on the left, R2 caption on the right. Lets a human
compare qualitative caption quality even though both scored 25% on coverage.

Reads:
    eval/cosmos_descriptions.parquet      (Cosmos-Reason1-7B)
    eval/cosmos_descriptions_r2.parquet   (Cosmos-Reason2-8B)

Outputs:
    dashboard/cosmos_compare.html         (self-contained, base64 thumbnails)

CLI:
    python -m scripts.cosmos_compare_viewer                    # default 30 pairs
    python -m scripts.cosmos_compare_viewer --limit 60
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.cloud import get_fs

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    r1_parquet: str = typer.Option("eval/cosmos_descriptions.parquet"),
    r2_parquet: str = typer.Option("eval/cosmos_descriptions_r2.parquet"),
    out: str = typer.Option("dashboard/cosmos_compare.html"),
    limit: int = typer.Option(30, help="Number of pairs to render (0 = all 500)"),
):
    load_dotenv()
    import duckdb

    con = duckdb.connect()
    sql = f"""
        SELECT r1.frame_id, r1.dataset, r1.camera_name, r1.thumbnail_uri,
               r1.description AS desc_r1, r2.description AS desc_r2
        FROM read_parquet('{r1_parquet}') r1
        JOIN read_parquet('{r2_parquet}') r2 USING (frame_id)
    """
    if limit:
        sql += f" LIMIT {limit}"
    rows = con.execute(sql).fetchall()
    console.print(f"loaded {len(rows)} paired captions")

    pairs = []
    for i, (fid, ds, cam, thumb_uri, d1, d2) in enumerate(rows, 1):
        try:
            fs, path = get_fs(thumb_uri)
            with fs.open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img = f'<img src="data:image/jpeg;base64,{b64}" alt="">'
        except Exception as e:
            img = f'<div class="err">image load failed: {html.escape(str(e))}</div>'
        # Length-based diff hint — R2 tends to write more structured/longer descriptions
        len_diff = len(d2) - len(d1)
        delta = f"R2 is {abs(len_diff)} chars {'longer' if len_diff > 0 else 'shorter'}"
        pairs.append(
            f'<div class="pair">'
            f'  <div class="header"><span class="rank">#{i}</span> '
            f'<span class="ds">{html.escape(ds)} · {html.escape(cam)}</span> '
            f'<span class="delta">{delta}</span></div>'
            f'  <div class="grid">'
            f'    <div class="thumb">{img}<div class="fid">{html.escape(fid)}</div></div>'
            f'    <div class="cap r1"><div class="cap-h">Cosmos-Reason1-7B (prior gen)</div>'
            f'      <div class="cap-b">{html.escape(d1)}</div></div>'
            f'    <div class="cap r2"><div class="cap-h">Cosmos-Reason2-8B (current gen)</div>'
            f'      <div class="cap-b">{html.escape(d2)}</div></div>'
            f'  </div>'
            f'</div>'
        )
        if i % 10 == 0:
            console.print(f"  {i}/{len(rows)}")

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cosmos-Reason1 vs Reason2 · same frames</title>
<style>
  body {{ font:14px/1.55 -apple-system,system-ui,sans-serif; margin:0; padding:24px; background:#fafbfc; color:#14181f; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:#6b7280; font-size:13px; margin-bottom:24px; max-width:880px; }}
  .pair {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:16px 18px; margin-bottom:18px; }}
  .header {{ display:flex; gap:12px; align-items:center; margin-bottom:12px; font:12px ui-monospace,monospace; color:#6b7280; }}
  .rank {{ background:#3730a3; color:#fff; padding:2px 8px; border-radius:4px; font-weight:700; }}
  .ds {{ color:#0a7; font-weight:600; }}
  .delta {{ margin-left:auto; color:#b45309; font-style:italic; }}
  .grid {{ display:grid; grid-template-columns:320px 1fr 1fr; gap:14px; }}
  @media (max-width:1000px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .thumb img {{ width:100%; height:auto; border-radius:4px; background:#000; }}
  .fid {{ font:10px ui-monospace,monospace; color:#6b7280; margin-top:6px; word-break:break-all; }}
  .cap {{ background:#fcfdff; border:1px solid #e5e7eb; border-radius:6px; padding:12px; font-size:13.5px; line-height:1.5; }}
  .cap.r1 {{ border-left:3px solid #5b50d6; }}
  .cap.r2 {{ border-left:3px solid #0a7; }}
  .cap-h {{ font:11px ui-monospace,monospace; color:#6b7280; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-bottom:8px; }}
  .err {{ color:#c40; font:12px ui-monospace,monospace; padding:20px; background:#fff5f0; border-radius:4px; }}
</style></head><body>
<h1>Cosmos-Reason1-7B vs Cosmos-Reason2-8B — same frames, same prompt, different captions</h1>
<div class="sub">Both models captioned the EXACT SAME 500 frames (seed=42) with the prompt "Describe this driving scene in 2-3 sentences. Focus on: scene type, time of day and weather, actors and their positions, any unusual or noteworthy elements." Coverage on the 12-needle eval was identical (25% each), but the qualitative differences are visible per-frame. R2 tends to write more structured, numeric descriptions (lane counts, divider mentions, specific actor positions); R1 tends toward atmospheric/vibes language ("bustling," "soft glow").</div>
{chr(10).join(pairs)}
</body></html>"""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    console.print(f"[green]✓ wrote {out_path} ({out_path.stat().st_size//1024} KB)[/green]")
    console.print(f"  open {out_path.absolute()}")


if __name__ == "__main__":
    app()
