"""
Metadata-lift experiment.

Hypothesis: For needles that require structured filters, adding metadata as a
WHERE-clause in front of vector search dramatically improves precision@K — even
when the encoder is held constant.

Setup: 240 Waymo frames that have BOTH a SigLIP image embedding AND a
Cosmos-Reason-2 caption. Caption regex derives 4 structured fields:
  time_of_day  ∈ {night, day}
  has_pedestrian ∈ {true, false}
  is_wet ∈ {true, false}
  has_construction ∈ {true, false}

For each of 3 needles we compare:
  PURE   : SigLIP text→image vector search over all 240 frames, top-10
  HYBRID : pre-filter by metadata predicate, then SigLIP rank within, top-10

Precision@10 = fraction of top-10 results that satisfy the needle's expected
metadata predicate.
"""

import re
import json
import numpy as np
import lancedb
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import io

import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
LANCE_DIR = os.environ.get("LANCE_DIR", str(_ROOT / "data" / "lance"))
OUT_PATH = str(_ROOT / "eval" / "metadata_lift_results.json")
K = 10

# --- 1) Build the labeled subset ---
db = lancedb.connect(LANCE_DIR)
sig = db.open_table("waymo").to_pandas()           # SigLIP image embeddings
cap = db.open_table("cosmos_aug_r2").to_pandas()   # Cosmos-R2 captions
joined = sig.merge(cap[["frame_id", "description"]], on="frame_id", how="inner")
print(f"[1] labeled subset: {len(joined)} frames")

# --- 2) Derive metadata from captions (regex) ---
desc = joined["description"].str.lower().fillna("")
joined["time_of_day"]      = np.where(desc.str.contains("night|dark|evening|dusk"), "night", "day")
joined["has_pedestrian"]   = desc.str.contains("pedestrian|person walking|people walking|man walking|woman walking|child walking")
joined["is_wet"]           = desc.str.contains(r"\brain\b|\bwet\b|reflection|puddle")
joined["has_construction"] = desc.str.contains("construction|cone|worker|barrier|orange cone")

print(f"[2] metadata-positive counts in the 240-frame subset:")
print(f"    time_of_day=night          {(joined['time_of_day']=='night').sum()}")
print(f"    has_pedestrian             {joined['has_pedestrian'].sum()}")
print(f"    is_wet                     {joined['is_wet'].sum()}")
print(f"    night AND pedestrian       {((joined['time_of_day']=='night') & joined['has_pedestrian']).sum()}")
print(f"    wet AND night              {(joined['is_wet'] & (joined['time_of_day']=='night')).sum()}")

# --- 3) Load SigLIP and embed needle queries ---
print(f"[3] loading SigLIP for text query encoding...")
model_name = "google/siglip-base-patch16-224"
proc = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()

def embed_text(query: str) -> np.ndarray:
    with torch.no_grad():
        ins = proc(text=[query], return_tensors="pt", padding=True, truncation=True)
        feat = model.get_text_features(**ins)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype(np.float32)

# --- 4) Define needles, predicates, and queries ---
NEEDLES = [
    {
        "id": "n06_night_pedestrian",
        "query": "pedestrian at night, poor visibility",
        "predicate": lambda r: (r["time_of_day"] == "night") and r["has_pedestrian"],
        "sql_filter_str": "time_of_day='night' AND has_pedestrian=true",
    },
    {
        "id": "n11_wet_road_reflections",
        "query": "wet road, reflections from streetlights at night",
        "predicate": lambda r: r["is_wet"] and (r["time_of_day"] == "night"),
        "sql_filter_str": "is_wet=true AND time_of_day='night'",
    },
    {
        "id": "n06_night_only",
        "query": "scene at night",
        "predicate": lambda r: r["time_of_day"] == "night",
        "sql_filter_str": "time_of_day='night'",
    },
]

# --- 5) Run pure vs hybrid for each needle ---
results = []
emb_matrix = np.stack(joined["embedding"].values).astype(np.float32)
emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

for n in NEEDLES:
    print(f"\n[5] needle: {n['id']}  query='{n['query']}'")
    qv = embed_text(n["query"])

    # PURE: cosine over all 240
    sims = emb_matrix @ qv
    pure_idx = np.argsort(-sims)[:K]
    pure_rows = joined.iloc[pure_idx]
    pure_hits = pure_rows.apply(n["predicate"], axis=1).sum()
    pure_p_at_k = pure_hits / K

    # HYBRID: filter first, then rank within filtered set
    mask = joined.apply(n["predicate"], axis=1).values
    filtered_pool = mask.sum()
    if filtered_pool == 0:
        hyb_p_at_k = None
        hyb_pool = 0
    else:
        f_idx = np.where(mask)[0]
        f_sims = emb_matrix[f_idx] @ qv
        order = np.argsort(-f_sims)[:K]
        hyb_idx = f_idx[order]
        hyb_rows = joined.iloc[hyb_idx]
        hyb_hits = hyb_rows.apply(n["predicate"], axis=1).sum()
        hyb_p_at_k = hyb_hits / min(K, filtered_pool)
        hyb_pool = filtered_pool

    print(f"  PURE   vector   : precision@{K} = {pure_p_at_k:.2f} ({pure_hits}/{K})")
    print(f"  HYBRID metadata+vec: precision@{K} = {hyb_p_at_k:.2f}  (pool size {hyb_pool})")

    lift_abs = (hyb_p_at_k - pure_p_at_k) if hyb_p_at_k is not None else None
    results.append({
        "needle": n["id"],
        "query": n["query"],
        "sql_filter": n["sql_filter_str"],
        "pure_vector_precision_at_10": float(pure_p_at_k),
        "hybrid_precision_at_10": float(hyb_p_at_k) if hyb_p_at_k is not None else None,
        "lift_absolute": float(lift_abs) if lift_abs is not None else None,
        "lift_multiplier": float(hyb_p_at_k / pure_p_at_k) if (hyb_p_at_k is not None and pure_p_at_k > 0) else None,
        "filtered_pool_size": int(hyb_pool),
        "labeled_corpus_size": len(joined),
    })

# --- 6) Save + summarize ---
with open(OUT_PATH, "w") as f:
    json.dump({
        "experiment": "metadata_lift",
        "corpus": "240 Waymo frames with SigLIP image embedding AND Cosmos-R2 caption",
        "metadata_derivation": "regex on Cosmos-R2 caption text",
        "results": results,
    }, f, indent=2)

print(f"\n[6] wrote {OUT_PATH}")
print("\nSUMMARY:")
print(f"  {'needle':<28s} {'pure':>6s} {'hybrid':>8s} {'lift':>8s}")
for r in results:
    pure = r["pure_vector_precision_at_10"]
    hyb = r["hybrid_precision_at_10"]
    lift = r["lift_multiplier"]
    pure_s = f"{pure:.2f}"
    hyb_s = f"{hyb:.2f}" if hyb is not None else "—"
    lift_s = f"{lift:.1f}x" if lift is not None else "—"
    print(f"  {r['needle']:<28s} {pure_s:>6s} {hyb_s:>8s} {lift_s:>8s}")
