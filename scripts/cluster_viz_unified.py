"""Unified cluster viz — ONE shared UMAP projection of:
  - 15K random corpus frames (background — semantic space coverage)
  - 7 query TEXT vectors (large stars — where each query "lands" in space)
  - 84 top-K result vectors (colored dots — each query's 12 nearest neighbors)

Unlike the per-query maps, here ALL points share the same projection coordinate
system, so spatial proximity between queries / results is real and comparable.

Output: dashboard/cluster_viz_unified.html (~3-5 MB, scattergl for 15K background)
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

QUERIES = [
    ("school",                                "#10b981"),
    ("jaywalker at night in SF",              "#ef4444"),
    ("unprotected left turn with pedestrian", "#f59e0b"),
    ("school buses",                          "#3b82f6"),
    ("cyclist weaving between cars",          "#8b5cf6"),
    ("wet road reflections at night",         "#06b6d4"),
    ("construction zone cones",               "#ec4899"),
]


def _thumb_b64(uri: str, fs_cache: dict) -> str | None:
    try:
        from src.cloud import get_fs
        from PIL import Image
        scheme = "s3" if uri.startswith("s3://") else "gcs" if uri.startswith("gs://") else None
        if not scheme:
            return None
        fs = fs_cache.get(scheme)
        if fs is None:
            fs, _ = get_fs(uri)
            fs_cache[scheme] = fs
        path = uri.replace("s3://", "").replace("gs://", "")
        with fs.open(path, "rb") as f:
            img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


@app.command()
def main(
    table: str = typer.Option("waymo"),
    k: int = typer.Option(12, help="top-K per query"),
    bg_sample: int = typer.Option(15000, help="corpus background points"),
    out: str = typer.Option("dashboard/cluster_viz_unified.html"),
    lance_dir: str = typer.Option("data/lance"),
):
    load_dotenv()
    import lancedb
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import umap

    from src.query.hybrid import _siglip_text_encode

    console.rule("[bold]1/5 encode 7 query texts → 768d vectors")
    query_vecs = {}  # q_text -> vector
    for q_text, color in QUERIES:
        query_vecs[q_text] = _siglip_text_encode(q_text)
        console.print(f"  '{q_text}' → 768d vector encoded")

    console.rule(f"[bold]2/5 retrieve top-{k} per query")
    db = lancedb.connect(lance_dir)
    tbl = db.open_table(table)
    query_hits = {}
    for q_text, color in QUERIES:
        hits = tbl.search(query_vecs[q_text]).limit(k).to_list()
        query_hits[q_text] = hits
        console.print(f"  '{q_text}' → {len(hits)} top-K hits")

    console.rule(f"[bold]3/5 sample {bg_sample} corpus background")
    all_rows = tbl.search().limit(10**9).to_list()
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(all_rows), size=min(bg_sample, len(all_rows)), replace=False)
    bg_rows = [all_rows[i] for i in bg_indices]
    # Exclude any top-K frames from background
    all_topk_ids = {h["frame_id"] for hits in query_hits.values() for h in hits}
    bg_rows = [r for r in bg_rows if r["frame_id"] not in all_topk_ids]
    console.print(f"  background pool: {len(bg_rows)} unique frames (post-exclusion)")

    console.rule("[bold]4/5 unified UMAP — projecting ALL points to ONE 2d space")
    # Build the unified embedding matrix:
    #   [background ... top-K results ... query vectors]
    all_embeddings = []
    all_meta = []  # parallel list of dicts

    for r in bg_rows:
        all_embeddings.append(np.array(r["embedding"], dtype=np.float32))
        all_meta.append({
            "kind": "background",
            "frame_id": r["frame_id"],
            "camera_name": r["camera_name"],
            "thumbnail_uri": r["thumbnail_uri"],
            "query": None,
            "color": "#cbd5e1",
            "rank": -1,
            "score": 0.0,
        })

    for q_text, color in QUERIES:
        for i, h in enumerate(query_hits[q_text]):
            all_embeddings.append(np.array(h["embedding"], dtype=np.float32))
            raw_score = float(1 - h.get("_distance", 0))
            all_meta.append({
                "kind": "hit",
                "frame_id": h["frame_id"],
                "camera_name": h["camera_name"],
                "thumbnail_uri": h["thumbnail_uri"],
                "query": q_text,
                "color": color,
                "rank": i + 1,
                "score": raw_score if -1.5 < raw_score < 1.5 else float("nan"),
            })

    for q_text, color in QUERIES:
        all_embeddings.append(np.array(query_vecs[q_text], dtype=np.float32))
        all_meta.append({
            "kind": "query_vec",
            "frame_id": "(query text)",
            "camera_name": "—",
            "thumbnail_uri": None,
            "query": q_text,
            "color": color,
            "rank": 0,
            "score": 0.0,
        })

    embeddings = np.stack(all_embeddings, axis=0)
    console.print(f"  total points to project: {len(embeddings)}")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                        metric="cosine", random_state=42, verbose=True)
    xy = reducer.fit_transform(embeddings)

    df = pd.DataFrame(all_meta)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    console.rule("[bold]4b/5 load thumbnails for hit + query points")
    fs_cache = {}
    df["thumb_b64"] = ""
    hit_mask = df["kind"] == "hit"
    loaded = 0
    for idx in df[hit_mask].index:
        thumb = _thumb_b64(df.at[idx, "thumbnail_uri"], fs_cache)
        if thumb:
            df.at[idx, "thumb_b64"] = thumb
            loaded += 1
    console.print(f"  loaded {loaded} hit thumbnails")

    console.rule("[bold]5/5 render unified plotly chart")
    fig = go.Figure()

    # Background — scattergl for performance with 15K points
    bg = df[df["kind"] == "background"]
    fig.add_trace(go.Scattergl(
        x=bg["x"], y=bg["y"],
        mode="markers",
        name=f"background ({len(bg):,} corpus frames)",
        marker=dict(color="#cbd5e1", size=3.5, opacity=0.35,
                    line=dict(width=0)),
        customdata=[[r["frame_id"], r["camera_name"]] for _, r in bg.iterrows()],
        hovertemplate="<b>background frame</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ))

    # Per-query top-K traces (one trace per query for legend)
    for q_text, color in QUERIES:
        hits_df = df[(df["kind"] == "hit") & (df["query"] == q_text)]
        if hits_df.empty:
            continue
        custom = []
        for _, r in hits_df.iterrows():
            score_str = f"{r['score']:.4f}" if not pd.isna(r["score"]) else "n/a"
            custom.append([
                int(r["rank"]), r["frame_id"], r["camera_name"],
                score_str, r["thumb_b64"] or "", q_text,
            ])
        fig.add_trace(go.Scatter(
            x=hits_df["x"], y=hits_df["y"],
            mode="markers",
            name=f"{q_text} (top-{k})",
            marker=dict(color=color, size=10, opacity=0.92,
                        line=dict(width=1.5, color="#fff")),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[5]}</b> · rank #%{customdata[0]}<br>"
                "score: %{customdata[3]}<br>"
                "%{customdata[1]}<br>%{customdata[2]}"
                "<extra></extra>"
            ),
        ))

    # Query text vectors — large stars
    qv = df[df["kind"] == "query_vec"]
    for q_text, color in QUERIES:
        qv_row = qv[qv["query"] == q_text]
        if qv_row.empty:
            continue
        r = qv_row.iloc[0]
        fig.add_trace(go.Scatter(
            x=[r["x"]], y=[r["y"]],
            mode="markers+text",
            name=f"⭐ '{q_text}' query vector",
            text=[f"⭐ {q_text}"],
            textposition="top center",
            textfont=dict(size=11, color=color, family="-apple-system, system-ui, sans-serif"),
            marker=dict(symbol="star", color=color, size=22,
                        line=dict(width=2, color="#fff")),
            customdata=[[q_text, "query text vector"]],
            hovertemplate="<b>QUERY: %{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text="<b>🌌 Unified SigLIP embedding space</b><br>"
                 "<sub>ALL points share the same UMAP projection. ⭐ stars = query text vectors. "
                 "Colored dots = each query's top-12 hits. Gray = corpus background. "
                 "Spatial proximity here is REAL — close points = semantically similar in 768d space.</sub>",
            font=dict(size=15),
        ),
        xaxis=dict(showticklabels=False, title="", zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, title="", zeroline=False, showgrid=False),
        plot_bgcolor="#fafbfc",
        paper_bgcolor="#fff",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=11)),
        margin=dict(l=20, r=260, t=90, b=20),
        height=820,
        font=dict(family="-apple-system, system-ui, sans-serif", size=12),
        hoverlabel=dict(bgcolor="#fff", font_size=12),
    )

    # Inject thumbnail-hover JS overlay
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>physical-ai-triage · unified embedding space</title>
<style>
  body {{ font:14px/1.5 -apple-system,system-ui,sans-serif; margin:0; padding:0; background:#fff; }}
  #thumb-overlay {{
    position: fixed; pointer-events: none; z-index: 9999;
    background: #fff; border: 2px solid #3730a3; border-radius: 8px;
    padding: 6px; box-shadow: 0 6px 20px rgba(0,0,0,0.25); display: none;
  }}
  #thumb-overlay img {{ display: block; width: 256px; height: auto; border-radius: 4px; }}
  #thumb-overlay .label {{ font: 11px ui-monospace,monospace; color: #6b7280; margin-top: 4px; text-align: center; }}
</style>
</head><body>
{fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="unified_plot")}
<div id="thumb-overlay">
  <img id="thumb-img" src=""><div class="label" id="thumb-label"></div>
</div>
<script>
  function attachHover() {{
    const overlay = document.getElementById('thumb-overlay');
    const img = document.getElementById('thumb-img');
    const label = document.getElementById('thumb-label');
    const plot = document.getElementById('unified_plot');
    if (!plot || !plot.on) {{ setTimeout(attachHover, 100); return; }}
    plot.on('plotly_hover', e => {{
      const pt = e.points[0];
      if (!pt || !pt.customdata || pt.customdata.length < 5 || !pt.customdata[4]) return;
      img.src = pt.customdata[4];
      label.textContent = `${{pt.customdata[5]}} · #${{pt.customdata[0]}} · ${{pt.customdata[2]}}`;
      overlay.style.display = 'block';
      const x = e.event.clientX + 16;
      const y = e.event.clientY + 16;
      overlay.style.left = Math.min(x, window.innerWidth - 290) + 'px';
      overlay.style.top = Math.min(y, window.innerHeight - 240) + 'px';
    }});
    plot.on('plotly_unhover', () => {{ overlay.style.display = 'none'; }});
  }}
  attachHover();
</script>
</body></html>"""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    sz_mb = out_path.stat().st_size / 1024 / 1024
    console.print(f"\n[green]✓ unified viz written: {out_path} ({sz_mb:.1f} MB)[/green]")


if __name__ == "__main__":
    app()
