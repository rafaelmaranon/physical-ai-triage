"""HQ cluster viz — one publication-quality map per query, with thumbnail hovers.

Each query gets its own UMAP projection so the cluster structure is clearest. Top-K
points are colored + larger; background is a dense gray cloud of the rest of the
corpus. Hover any point → thumbnail preview + frame_id + score.

Outputs:
    dashboard/cluster_viz_hq.html       (gallery — all queries, switch via dropdown)
    dashboard/cluster_viz_<slug>.html   (one per query, standalone for embedding)

CLI:
    uv run python -m scripts.cluster_viz_hq                   # default 8K bg
    uv run python -m scripts.cluster_viz_hq --bg-sample 15000 # denser background
"""
from __future__ import annotations

import base64
import io
import os
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


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _thumb_b64(uri: str, src_fs_cache: dict) -> str | None:
    """Load a thumbnail from S3 and return data:image/jpeg;base64,... Returns None on fail."""
    try:
        from src.cloud import get_fs
        from PIL import Image
        if uri.startswith("s3://"):
            scheme = "s3"
        elif uri.startswith("gs://"):
            scheme = "gcs"
        else:
            return None
        fs = src_fs_cache.get(scheme)
        if fs is None:
            fs, _ = get_fs(uri)
            src_fs_cache[scheme] = fs
        path = uri.replace("s3://", "").replace("gs://", "")
        with fs.open(path, "rb") as f:
            img_bytes = f.read()
        # Resize to 128px wide for hover preview (keeps the HTML lean)
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
    bg_sample: int = typer.Option(8000, help="background corpus points (richer = more context)"),
    out_dir: str = typer.Option("dashboard"),
    lance_dir: str = typer.Option("data/lance"),
    skip_thumbs: bool = typer.Option(False, help="skip thumbnail hover (faster, smaller HTML)"),
):
    load_dotenv()
    import lancedb
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import umap

    from src.query.hybrid import _siglip_text_encode

    console.rule(f"[bold]1/4 retrieve top-{k} per query against `{table}`")
    db = lancedb.connect(lance_dir)
    tbl = db.open_table(table)

    query_hits: dict[str, list[dict]] = {}
    for q_text, color in QUERIES:
        console.print(f"  query: '{q_text}'")
        qvec = _siglip_text_encode(q_text)
        hits = tbl.search(qvec).limit(k).to_list()
        query_hits[q_text] = [
            {
                "frame_id": h["frame_id"],
                "embedding": h["embedding"],
                "rank": i + 1,
                "score": float(1 - h.get("_distance", 0)),
                "thumbnail_uri": h["thumbnail_uri"],
                "camera_name": h["camera_name"],
            }
            for i, h in enumerate(hits)
        ]

    console.rule(f"[bold]2/4 sample background corpus ({bg_sample} points)")
    all_rows = tbl.search().limit(10**9).to_list()
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(all_rows), size=min(bg_sample, len(all_rows)), replace=False)
    bg_set = {all_rows[i]["frame_id"]: all_rows[i] for i in bg_indices}
    console.print(f"  background pool: {len(bg_set)} unique frames")

    fs_cache = {}
    if not skip_thumbs:
        console.rule("[bold]3a/4 load thumbnails for query hits (hover previews)")
        loaded = 0
        for q_text, hits in query_hits.items():
            for h in hits:
                h["thumb_b64"] = _thumb_b64(h["thumbnail_uri"], fs_cache)
                if h["thumb_b64"]:
                    loaded += 1
        console.print(f"  loaded {loaded} thumbnails for query results")

    console.rule("[bold]3b/4 UMAP project + render per-query maps")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    per_query_figs = {}  # slug → fig

    for q_text, color in QUERIES:
        hits = query_hits[q_text]
        rows = []
        for h in hits:
            rows.append({**h, "category": "top-K hit", "color": color})
        # Background: exclude any frames that are in the query top-K to avoid dupes
        excluded = {h["frame_id"] for h in hits}
        for fid, r in bg_set.items():
            if fid in excluded:
                continue
            rows.append({
                "frame_id": fid,
                "embedding": r["embedding"],
                "rank": -1,
                "score": 0.0,
                "thumbnail_uri": r["thumbnail_uri"],
                "camera_name": r["camera_name"],
                "category": "background",
                "color": "#cbd5e1",
                "thumb_b64": None,
            })

        embeddings = np.array([np.array(r["embedding"], dtype=np.float32) for r in rows])
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2,
                            metric="cosine", random_state=42, verbose=False)
        xy = reducer.fit_transform(embeddings)
        df = pd.DataFrame(rows)
        df["x"], df["y"] = xy[:, 0], xy[:, 1]
        df["thumb_b64"] = df["thumb_b64"].fillna("")

        # Cluster tightness stat: mean pairwise cosine among top-K
        topk = df[df["category"] == "top-K hit"]
        topk_vecs = np.array([np.array(e, dtype=np.float32) for e in topk["embedding"]])
        topk_norms = topk_vecs / (np.linalg.norm(topk_vecs, axis=1, keepdims=True) + 1e-9)
        if len(topk_norms) > 1:
            sims = topk_norms @ topk_norms.T
            np.fill_diagonal(sims, np.nan)
            tightness = float(np.nanmean(sims))
        else:
            tightness = float("nan")

        # Build plotly fig
        fig = go.Figure()

        # Background trace (gray, small)
        bg_df = df[df["category"] == "background"]
        fig.add_trace(go.Scatter(
            x=bg_df["x"], y=bg_df["y"],
            mode="markers",
            name=f"background ({len(bg_df):,})",
            marker=dict(color="#cbd5e1", size=4, opacity=0.35,
                        line=dict(width=0)),
            customdata=[[r["frame_id"], r["camera_name"]] for _, r in bg_df.iterrows()],
            hovertemplate="<b>background frame</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        ))

        # Top-K trace (colored, large, with thumbnails in CUSTOM JS OVERLAY — NOT in hovertemplate)
        tk_df = df[df["category"] == "top-K hit"].copy()
        custom = []
        for _, r in tk_df.iterrows():
            # Normalize score display: if it's a sensible cosine similarity, show 4 decimals;
            # otherwise (raw distance leaking through), show "n/a" rather than -539.xx garbage.
            raw_score = float(r["score"])
            score_str = f"{raw_score:.4f}" if -1.5 < raw_score < 1.5 else "n/a"
            custom.append([
                int(r["rank"]),
                r["frame_id"],
                r["camera_name"],
                score_str,
                r["thumb_b64"] or "",  # consumed by JS overlay, NOT by plotly hover
            ])
        fig.add_trace(go.Scatter(
            x=tk_df["x"], y=tk_df["y"],
            mode="markers+text",
            name=f"top-{k} hits",
            text=[f"#{int(r['rank'])}" for _, r in tk_df.iterrows()],
            textposition="top center",
            textfont=dict(size=10, color=color),
            marker=dict(color=color, size=14, opacity=0.92,
                        line=dict(width=2, color="#fff")),
            customdata=custom,
            # NO <img> here — plotly strips it and leaks the base64 as text.
            # The JS overlay (in gallery HTML) reads customdata[4] and renders the image.
            hovertemplate=(
                "<b>rank #%{customdata[0]}</b> · score %{customdata[3]}<br>"
                "%{customdata[1]}<br>%{customdata[2]}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>SigLIP embedding space · query: \"{q_text}\"</b><br>"
                     f"<sub>top-{k} hits (colored) vs {len(bg_df):,} background corpus frames · "
                     f"768d → 2d via UMAP · cluster tightness "
                     f"<b>{tightness:.3f}</b> (mean pairwise cosine of top-K, 1.0 = identical)</sub>",
                font=dict(size=15),
            ),
            xaxis=dict(showticklabels=False, title="", zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, title="", zeroline=False, showgrid=False),
            plot_bgcolor="#fafbfc",
            paper_bgcolor="#fff",
            margin=dict(l=20, r=20, t=80, b=20),
            height=620,
            font=dict(family="-apple-system, system-ui, sans-serif", size=12),
            hoverlabel=dict(bgcolor="#fff", font_size=12, bordercolor=color),
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05),
        )
        slug = _slug(q_text)
        per_query_figs[slug] = (q_text, fig, tightness, len(tk_df))

        # Write standalone per-query HTML
        single_path = out_path / f"cluster_viz_{slug}.html"
        fig.write_html(str(single_path), include_plotlyjs="cdn", full_html=True)
        console.print(f"  ✓ {slug:40s} tightness={tightness:.3f} → {single_path.name}")

    console.rule("[bold]4/4 build gallery (all queries, switch via dropdown)")
    # Build a gallery HTML: one section per query, all in one scrollable page
    sections = []
    for slug, (q_text, fig, tightness, n_hits) in per_query_figs.items():
        sections.append(
            f'<section id="{slug}">'
            f'  <h2>{q_text}</h2>'
            f'  <div class="meta">slug: <code>{slug}</code> · top-K cluster tightness: '
            f'<b>{tightness:.3f}</b> · <span class="tightness-bar">'
            f'<span style="width:{max(0,min(100, tightness*100)):.0f}%"></span></span></div>'
            f'  {fig.to_html(include_plotlyjs=False, full_html=False, div_id=f"plot_{slug}")}'
            f'</section>'
        )
    nav = '<nav>' + ' · '.join(
        f'<a href="#{slug}">{qt}</a>' for slug, (qt, _, _, _) in per_query_figs.items()
    ) + '</nav>'

    gallery_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>physical-ai-triage · cluster viz gallery</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font:14.5px/1.5 -apple-system,system-ui,sans-serif; margin:0; padding:0; background:#fafbfc; color:#14181f; }}
  header {{ background:#fff; border-bottom:1px solid #e5e7eb; padding:18px 28px; position:sticky; top:0; z-index:10; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .sub {{ color:#6b7280; font-size:13px; max-width:780px; }}
  nav {{ margin-top:10px; font-size:12.5px; }}
  nav a {{ color:#0a7; text-decoration:none; margin-right:6px; }}
  nav a:hover {{ text-decoration:underline; }}
  main {{ max-width:1200px; margin:0 auto; padding:24px; }}
  section {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:20px 24px; margin-bottom:24px; }}
  section h2 {{ margin:0 0 4px; font-size:17px; }}
  .meta {{ color:#6b7280; font-size:12.5px; margin-bottom:12px; display:flex; align-items:center; gap:10px; }}
  .meta code {{ background:#f5f7fa; padding:1px 6px; border-radius:3px; font:11.5px ui-monospace,monospace; color:#1c3a5e; }}
  .tightness-bar {{ display:inline-block; width:120px; height:6px; background:#eee; border-radius:99px; overflow:hidden; }}
  .tightness-bar span {{ display:block; height:100%; background:linear-gradient(to right,#fef3c7,#10b981); }}
  /* Floating thumbnail overlay (positioned on hover via JS) */
  #thumb-overlay {{
    position: fixed; pointer-events: none; z-index: 9999;
    background: #fff; border: 2px solid #3730a3; border-radius: 8px;
    padding: 6px; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    display: none;
  }}
  #thumb-overlay img {{ display: block; width: 256px; height: auto; border-radius: 4px; }}
  #thumb-overlay .label {{ font: 11px ui-monospace,monospace; color: #6b7280; margin-top: 4px; text-align: center; }}
</style>
</head><body>
<header>
  <h1>🌌 SigLIP embedding space · 7 queries · 768d → 2d (UMAP)</h1>
  <div class="sub">Each query gets its own UMAP projection so cluster structure is clearest. <b>Hover any colored dot to see the matching thumbnail.</b> Background = {len(bg_set):,} random corpus frames. <b>Cluster tightness</b> = mean pairwise cosine similarity among top-K hits (higher = the model groups results closer together).</div>
  {nav}
</header>
<main>
{chr(10).join(sections)}
</main>

<div id="thumb-overlay">
  <img id="thumb-img" src="">
  <div class="label" id="thumb-label"></div>
</div>

<script>
  // Wait until plotly has rendered all plots, then attach hover handlers.
  function attachHoverHandlers() {{
    const overlay = document.getElementById('thumb-overlay');
    const img = document.getElementById('thumb-img');
    const label = document.getElementById('thumb-label');
    document.querySelectorAll('.js-plotly-plot').forEach(plotEl => {{
      plotEl.on('plotly_hover', e => {{
        const pt = e.points[0];
        if (!pt || !pt.customdata) return;
        // customdata for top-K trace: [rank, frame_id, camera, score, b64]
        // background trace customdata is [frame_id, camera] — skip those
        if (pt.customdata.length < 5 || !pt.customdata[4]) return;
        img.src = pt.customdata[4];
        label.textContent = `#${{pt.customdata[0]}} · ${{pt.customdata[2]}} · ${{pt.customdata[3]}}`;
        overlay.style.display = 'block';
        // Position next to cursor (offset to avoid covering the hover tooltip)
        const x = e.event.clientX + 16;
        const y = e.event.clientY + 16;
        overlay.style.left = Math.min(x, window.innerWidth - 290) + 'px';
        overlay.style.top = Math.min(y, window.innerHeight - 240) + 'px';
      }});
      plotEl.on('plotly_unhover', () => {{ overlay.style.display = 'none'; }});
    }});
  }}
  // Plotly plots are added asynchronously. Wait for them.
  let attempts = 0;
  const waitForPlots = setInterval(() => {{
    if (document.querySelectorAll('.js-plotly-plot').length >= {len(per_query_figs)} || attempts++ > 100) {{
      clearInterval(waitForPlots);
      attachHoverHandlers();
    }}
  }}, 100);
</script>
</body></html>"""

    gallery_path = out_path / "cluster_viz_hq.html"
    gallery_path.write_text(gallery_html)
    sz_mb = gallery_path.stat().st_size / 1024 / 1024
    console.print(f"\n[green]✓ gallery written: {gallery_path} ({sz_mb:.1f} MB)[/green]")
    console.print(f"  open {gallery_path.absolute()}")


if __name__ == "__main__":
    app()
