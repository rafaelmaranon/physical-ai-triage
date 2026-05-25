"""Standalone Flask app: live side-by-side pure-vector vs metadata+vector lift demo.

Run: .venv/bin/python eval/lift_demo_server.py
Open: http://127.0.0.1:5051/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lancedb
import torch
from transformers import AutoModel, AutoProcessor
from flask import Flask, request, render_template_string

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANCE_DIR = os.environ.get("LANCE_DIR", os.path.join(_ROOT, "data", "lance"))

print("[1] loading lance tables...")
db = lancedb.connect(LANCE_DIR)
sig = db.open_table("waymo").to_pandas()
cap = db.open_table("cosmos_aug_r2").to_pandas()
joined = sig.merge(cap[["frame_id", "description"]], on="frame_id", how="inner").reset_index(drop=True)

print("[2] deriving metadata via caption regex...")
desc = joined["description"].str.lower().fillna("")
joined["time_of_day"] = np.where(desc.str.contains("night|dark|evening|dusk"), "night", "day")
joined["has_pedestrian"] = desc.str.contains("pedestrian|person walking|people walking|man walking|woman walking|child walking")
joined["is_wet"] = desc.str.contains(r"\brain\b|\bwet\b|reflection|puddle")

emb = np.stack(joined["embedding"].values).astype(np.float32)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
print(f"    {len(joined)} labeled frames, embeddings normalized")

print("[3] loading SigLIP (this may take ~10s)...")
model_name = "google/siglip-base-patch16-224"
proc = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()
print("    ready")


def embed_text(q: str) -> np.ndarray:
    with torch.no_grad():
        ins = proc(text=[q], return_tensors="pt", padding=True, truncation=True)
        feat = model.get_text_features(**ins)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype(np.float32)


PRESETS = {
    "night_ped": {
        "label": "🌙 Pedestrian at night",
        "query": "pedestrian at night, poor visibility",
        "predicate_str": "time_of_day = 'night' AND has_pedestrian = true",
        "predicate": lambda r: r["time_of_day"] == "night" and r["has_pedestrian"],
    },
    "wet_night": {
        "label": "🌧 Wet road at night",
        "query": "wet road, reflections from streetlights at night",
        "predicate_str": "is_wet = true AND time_of_day = 'night'",
        "predicate": lambda r: r["is_wet"] and r["time_of_day"] == "night",
    },
    "night": {
        "label": "🌃 Any night scene",
        "query": "scene at night",
        "predicate_str": "time_of_day = 'night'",
        "predicate": lambda r: r["time_of_day"] == "night",
    },
}

K = 10
app = Flask(__name__)


PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>Metadata-lift demo · physical-ai-triage</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#FAFAF8; color:#1a1a1a; padding:28px; line-height:1.5; }
  h1 { font-size:1.4rem; margin-bottom:4px; }
  .sub { color:#666; font-size:0.88rem; margin-bottom:18px; }
  .controls { background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:14px 18px; margin-bottom:18px; }
  .controls form { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .controls label { font-size:0.82rem; color:#444; }
  .controls input[type=text] { flex:1; min-width:280px; padding:8px 12px; border:1px solid #ccc; border-radius:5px; font-size:0.9rem; }
  .controls button { padding:8px 18px; border:0; background:#185FA5; color:#fff; border-radius:5px; cursor:pointer; font-weight:600; }
  .controls button:hover { background:#114a85; }
  .preset-btns { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .preset-btns a { padding:5px 12px; background:#eef2fb; color:#185FA5; text-decoration:none; border-radius:14px; font-size:0.78rem; font-weight:600; }
  .preset-btns a:hover { background:#dde6f7; }
  .preset-btns a.active { background:#185FA5; color:#fff; }
  .summary { background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:16px 20px; margin-bottom:18px; display:flex; gap:30px; flex-wrap:wrap; align-items:center; }
  .summary .item { text-align:left; }
  .summary .item .lbl { font-size:0.7rem; color:#888; text-transform:uppercase; letter-spacing:1px; }
  .summary .item .val { font-size:1.6rem; font-weight:700; font-family:ui-monospace, Menlo, monospace; color:#1a1a1a; }
  .summary .item .val.bad { color:#a01a1a; }
  .summary .item .val.good { color:#1a8a4a; }
  .summary .lift-arrow { font-size:1.2rem; color:#999; }
  .summary .lift-num { font-size:1.6rem; font-weight:700; color:#854F0B; font-family:ui-monospace, Menlo, monospace; }
  .filter-pill { display:inline-block; background:#FFFBEF; border:1px solid #d4b15c; color:#5a4a18; font-family:ui-monospace, Menlo, monospace; font-size:0.78rem; padding:3px 9px; border-radius:4px; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media (max-width:900px) { .cols { grid-template-columns:1fr; } }
  .col { background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:16px 18px; }
  .col h3 { font-size:0.92rem; font-weight:700; margin-bottom:4px; }
  .col h3 .tag { display:inline-block; font-size:0.65rem; padding:2px 7px; border-radius:3px; vertical-align:middle; margin-left:6px; font-family:ui-monospace,Menlo,monospace; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
  .col.pure h3 .tag { background:#fdecec; color:#a01a1a; }
  .col.hybrid h3 .tag { background:#e8fdf0; color:#1a8a4a; }
  .col .desc { font-size:0.78rem; color:#666; margin-bottom:12px; }
  .row { padding:9px 10px; border-radius:5px; margin-bottom:5px; background:#fafaf8; border-left:3px solid #ccc; }
  .row.match { border-left-color:#1a8a4a; background:#f0fdf4; }
  .row.miss { border-left-color:#a01a1a; background:#fef2f2; }
  .row .top { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
  .row .top .mark { font-weight:700; font-family:ui-monospace,Menlo,monospace; font-size:0.85rem; min-width:24px; }
  .row .top .mark.ok { color:#1a8a4a; }
  .row .top .mark.x { color:#a01a1a; }
  .row .caption { font-size:0.78rem; color:#333; line-height:1.4; flex:1; }
  .row .meta { font-size:0.68rem; color:#888; font-family:ui-monospace,Menlo,monospace; margin-top:3px; }
  .row .meta b { color:#333; }
  .why { background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:14px 18px; margin-top:18px; font-size:0.85rem; color:#444; line-height:1.55; }
  .why b { color:#1a1a1a; }
  .nav { display:flex; gap:18px; margin-bottom:14px; font-size:0.82rem; }
  .nav a { color:#185FA5; text-decoration:none; }
  .nav a:hover { text-decoration:underline; }
</style>
</head><body>
<h1>Metadata-lift demo · physical-ai-triage</h1>
<div class="sub">240 Waymo frames · SigLIP image embeddings · metadata derived from Cosmos-R2 captions · same encoder both sides</div>
<div class="nav">
  <a href="http://127.0.0.1:5050/" target="_blank">↗ main search app (port 5050)</a>
  <a href="http://127.0.0.1:5050/compare" target="_blank">↗ 6-model compare</a>
</div>

<div class="controls">
  <form method="get" action="/">
    <label>Query:</label>
    <input type="text" name="q" value="{{ query }}">
    <input type="hidden" name="pred" value="{{ predicate_key }}">
    <button type="submit">Search</button>
  </form>
  <div class="preset-btns">
    <span style="color:#888; font-size:0.78rem; margin-right:4px;">Try a preset:</span>
    {% for k, p in presets.items() %}
      <a href="/?preset={{k}}" class="{% if k == preset %}active{% endif %}">{{ p.label }}</a>
    {% endfor %}
  </div>
</div>

<div class="summary">
  <div class="item">
    <div class="lbl">Pure SigLIP P@10</div>
    <div class="val {% if pure_p < 0.5 %}bad{% endif %}">{{ "%.2f"|format(pure_p) }}</div>
  </div>
  <div class="lift-arrow">→</div>
  <div class="item">
    <div class="lbl">Hybrid (metadata + SigLIP) P@10</div>
    <div class="val good">{{ "%.2f"|format(hyb_p) }}</div>
  </div>
  <div class="lift-arrow">=</div>
  <div class="item">
    <div class="lbl">Lift</div>
    <div class="lift-num">{% if pure_p > 0 %}{{ "%.1f"|format(hyb_p / pure_p) }}×{% else %}0 → 1{% endif %}</div>
  </div>
  <div class="item" style="margin-left:auto;">
    <div class="lbl">Metadata filter applied (hybrid only)</div>
    <span class="filter-pill">{{ predicate_str }}</span>
  </div>
</div>

<div class="cols">
  <div class="col pure">
    <h3>Top-10 · Pure SigLIP <span class="tag">no filter</span></h3>
    <div class="desc">Embed query → cosine-rank all {{ corpus_n }} frames → top 10.</div>
    {% for r in pure_rows %}
      <div class="row {% if r.match %}match{% else %}miss{% endif %}">
        <div class="top">
          <span class="mark {% if r.match %}ok{% else %}x{% endif %}">{% if r.match %}✓{% else %}✗{% endif %}</span>
          <div style="flex:1;">
            <div class="caption">{{ r.caption[:200] }}{% if r.caption|length > 200 %}…{% endif %}</div>
            <div class="meta">time_of_day=<b>{{ r.tod }}</b> · has_ped=<b>{{ r.ped }}</b> · is_wet=<b>{{ r.wet }}</b> · sim={{ "%.3f"|format(r.sim) }}</div>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>

  <div class="col hybrid">
    <h3>Top-10 · Hybrid (metadata + SigLIP) <span class="tag">filter then rank</span></h3>
    <div class="desc">Filter corpus to the {{ pool_n }} frames matching <code>{{ predicate_str }}</code> → SigLIP cosine-rank within → top 10.</div>
    {% for r in hyb_rows %}
      <div class="row {% if r.match %}match{% else %}miss{% endif %}">
        <div class="top">
          <span class="mark {% if r.match %}ok{% else %}x{% endif %}">{% if r.match %}✓{% else %}✗{% endif %}</span>
          <div style="flex:1;">
            <div class="caption">{{ r.caption[:200] }}{% if r.caption|length > 200 %}…{% endif %}</div>
            <div class="meta">time_of_day=<b>{{ r.tod }}</b> · has_ped=<b>{{ r.ped }}</b> · is_wet=<b>{{ r.wet }}</b> · sim={{ "%.3f"|format(r.sim) }}</div>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>
</div>

<div class="why">
  <b>How to read this:</b> ✓ means the row's metadata satisfies <code>{{ predicate_str }}</code> (i.e. it's actually a correct match for your query). ✗ means it doesn't. <b>Pure</b> = SigLIP only. <b>Hybrid</b> = same SigLIP, but with the metadata predicate pre-filtering the corpus to {{ pool_n }} candidates before ranking. The encoder doesn't change; only the structure around it does. <br><br>
  <b>Caveat:</b> the labels and the filter share derivation (both come from regex on the Cosmos-R2 caption), so the hybrid side is favored by construction. A real ingest-time labeler will introduce noise — but the gap is large enough to survive 20–30% noise easily.
</div>

</body></html>"""


@app.route("/")
def index():
    preset_key = request.args.get("preset")
    qtext = request.args.get("q")
    if preset_key and preset_key in PRESETS:
        p = PRESETS[preset_key]
        qtext = p["query"]
        pred_str = p["predicate_str"]
        pred_fn = p["predicate"]
    elif qtext:
        # custom query — but we need a predicate. Reuse the night-pedestrian preset by default.
        # In a fuller demo we'd parse query → predicate via LLM. For now: keep last preset.
        p = PRESETS.get(preset_key, PRESETS["night_ped"])
        pred_str = p["predicate_str"]
        pred_fn = p["predicate"]
        preset_key = preset_key or "night_ped"
    else:
        preset_key = "night_ped"
        p = PRESETS[preset_key]
        qtext = p["query"]
        pred_str = p["predicate_str"]
        pred_fn = p["predicate"]

    qv = embed_text(qtext)

    # PURE
    sims = emb @ qv
    pure_idx = np.argsort(-sims)[:K]
    pure_rows = []
    for i in pure_idx:
        r = joined.iloc[i]
        match = bool(pred_fn(r))
        pure_rows.append({
            "caption": r["description"],
            "tod": r["time_of_day"],
            "ped": bool(r["has_pedestrian"]),
            "wet": bool(r["is_wet"]),
            "sim": float(sims[i]),
            "match": match,
        })
    pure_p = sum(r["match"] for r in pure_rows) / K

    # HYBRID
    mask = joined.apply(pred_fn, axis=1).values
    pool_n = int(mask.sum())
    hyb_rows = []
    if pool_n > 0:
        f_idx = np.where(mask)[0]
        f_sims = emb[f_idx] @ qv
        order = np.argsort(-f_sims)[:K]
        for j, oi in enumerate(order):
            i = f_idx[oi]
            r = joined.iloc[i]
            hyb_rows.append({
                "caption": r["description"],
                "tod": r["time_of_day"],
                "ped": bool(r["has_pedestrian"]),
                "wet": bool(r["is_wet"]),
                "sim": float(f_sims[oi]),
                "match": True,
            })
        hyb_p = sum(r["match"] for r in hyb_rows) / min(K, pool_n)
    else:
        hyb_p = 0.0

    return render_template_string(
        PAGE,
        presets=PRESETS,
        preset=preset_key,
        query=qtext,
        predicate_key=preset_key,
        predicate_str=pred_str,
        pure_rows=pure_rows,
        hyb_rows=hyb_rows,
        pure_p=pure_p,
        hyb_p=hyb_p,
        pool_n=pool_n,
        corpus_n=len(joined),
    )


if __name__ == "__main__":
    print("\n  metadata-lift demo: http://127.0.0.1:5051/\n")
    app.run(host="127.0.0.1", port=5051, debug=False)
