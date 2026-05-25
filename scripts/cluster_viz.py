"""Cluster viz — project 768-dim SigLIP embeddings to 2D, color by query.

For each search query in QUERIES, run hybrid retrieval against the `waymo` LanceDB
table, collect the top-K frame_ids + their embeddings, and add a background random
sample of 2000 corpus points for context. Project all to 2D with UMAP, color the
top-K points by query, plot with Plotly (interactive: hover shows thumbnail URL +
frame_id).

Output: dashboard/cluster_viz.html — self-contained HTML, opens in any browser.

CLI:
    uv run python -m scripts.cluster_viz                            # default queries
    uv run python -m scripts.cluster_viz --k 20 --bg-sample 3000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

# Same 6 queries we tested in the demo + 1 extra control
QUERIES = [
    ("school",                              "#10b981"),
    ("jaywalker at night in SF",            "#ef4444"),
    ("unprotected left turn with pedestrian", "#f59e0b"),
    ("school buses",                        "#3b82f6"),
    ("cyclist weaving between cars",        "#8b5cf6"),
    ("wet road reflections at night",       "#06b6d4"),
    ("construction zone cones",             "#ec4899"),
]


@app.command()
def main(
    table: str = typer.Option("waymo"),
    k: int = typer.Option(12, help="top-K per query"),
    bg_sample: int = typer.Option(2000, help="random background points for context"),
    out: str = typer.Option("dashboard/cluster_viz.html"),
    lance_dir: str = typer.Option("data/lance"),
):
    load_dotenv()
    import lancedb
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import umap

    from src.query.hybrid import _siglip_text_encode

    console.rule(f"[bold]1/4 retrieve top-{k} per query")
    db = lancedb.connect(lance_dir)
    tbl = db.open_table(table)

    # Load query encoder once
    rows = []  # list of dicts: query, frame_id, embedding, category
    for q_text, color in QUERIES:
        console.print(f"  query: '{q_text}'")
        qvec = _siglip_text_encode(q_text)
        hits = tbl.search(qvec).limit(k).to_list()
        for i, h in enumerate(hits):
            rows.append({
                "query": q_text,
                "color": color,
                "category": q_text,
                "frame_id": h["frame_id"],
                "embedding": h["embedding"],
                "rank": i + 1,
                "score": float(1 - h.get("_distance", 0)),
                "thumbnail_uri": h["thumbnail_uri"],
                "camera_name": h["camera_name"],
            })

    console.print(f"  total query hits: {len(rows)}")

    console.rule("[bold]2/4 sample background corpus")
    # Get the full corpus (95K) — for background we just grab N random frames
    all_rows = tbl.search().limit(10**9).to_list()
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(all_rows), size=min(bg_sample, len(all_rows)), replace=False)
    for idx in bg_indices:
        r = all_rows[idx]
        if r["frame_id"] in {x["frame_id"] for x in rows}:
            continue
        rows.append({
            "query": "(corpus background)",
            "color": "#cbd5e1",
            "category": "(corpus background)",
            "frame_id": r["frame_id"],
            "embedding": r["embedding"],
            "rank": -1,
            "score": 0.0,
            "thumbnail_uri": r["thumbnail_uri"],
            "camera_name": r["camera_name"],
        })
    console.print(f"  total points after background: {len(rows)}")

    console.rule("[bold]3/4 UMAP project 768d → 2d")
    embeddings = np.array([np.array(r["embedding"], dtype=np.float32) for r in rows])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric="cosine", random_state=42, verbose=False)
    xy = reducer.fit_transform(embeddings)
    console.print(f"  projected {len(rows)} points to 2D")

    df = pd.DataFrame(rows)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    console.rule("[bold]4/4 render plotly HTML")
    # Build a custom hover that shows everything useful
    df["hover"] = df.apply(
        lambda r: f"<b>{r['category']}</b><br>"
                  f"rank: {'(bg)' if r['rank']<0 else f'#{r['rank']}'}<br>"
                  f"frame_id: {r['frame_id']}<br>"
                  f"camera: {r['camera_name']}<br>"
                  f"score: {r['score']:.4f}",
        axis=1,
    )

    fig = px.scatter(
        df, x="x", y="y",
        color="category",
        color_discrete_map={**{q: c for q, c in QUERIES}, "(corpus background)": "#cbd5e1"},
        hover_data={"x": False, "y": False, "category": False, "hover": True},
        opacity=0.7,
        category_orders={"category": [q for q, _ in QUERIES] + ["(corpus background)"]},
    )

    # Make background points smaller and behind everything
    for trace in fig.data:
        if trace.name == "(corpus background)":
            trace.marker.size = 4
            trace.marker.opacity = 0.25
        else:
            trace.marker.size = 11
            trace.marker.line = dict(width=1, color="#fff")

    fig.update_layout(
        title="<b>SigLIP embedding space — top-12 per query, projected from 768d → 2d (UMAP)</b><br>"
              "<sub>Each colored dot is a top-K result for one query. Gray = random corpus background (2000 frames). "
              "Cluster tightness shows how confidently the model groups query-relevant frames.</sub>",
        xaxis=dict(showticklabels=False, title="", zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, title="", zeroline=False, showgrid=False),
        plot_bgcolor="#fafbfc",
        paper_bgcolor="#fff",
        legend=dict(title="<b>Query</b>", orientation="v", x=1.02, y=1),
        margin=dict(l=20, r=200, t=80, b=20),
        height=720,
        font=dict(family="-apple-system, system-ui, sans-serif", size=12),
        hoverlabel=dict(bgcolor="#fff", font_size=12),
    )
    # Replace default hovertemplate with our custom hover field
    for trace in fig.data:
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    console.print(f"  [green]✓ wrote {out_path} ({out_path.stat().st_size//1024} KB)[/green]")
    console.print(f"  open {out_path.absolute()}")


if __name__ == "__main__":
    app()
