# Pipeline

End-to-end: raw AV datasets → searchable index → visualization-tool integration.

```
┌────────────────────────────────────────────────────────────────────┐
│  RAW DATASETS                                                       │
│  Waymo (BigQuery public) │ BDD100K (HTTP) │ nuScenes │ AV2 (S3)    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  ingest
┌────────────────────────────────────────────────────────────────────┐
│  GCS bucket — raw frames + thumbnails                               │
│  (135 GB steady-state for 3.5M frames)                              │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  extract scalars + labels
┌────────────────────────────────────────────────────────────────────┐
│  BigQuery — METADATA TABLES                                         │
│  frame_id │ ts_ns │ device · scene · city                           │
│  object_tags │ weather │ ego_speed │ gps                            │
│  → "find every left turn in SF with 3+ pedestrians" (SQL)           │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  SigLIP image encoder (GPU)
┌────────────────────────────────────────────────────────────────────┐
│  LanceDB — EMBEDDING TABLES                                         │
│  frame_id │ embedding (768d float16) │ metadata_ref                 │
│  → "find frames that look like THIS scene" (vector similarity)      │
│  → "find every jaywalker at night" (text → vector)                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  JOIN on frame_id
┌────────────────────────────────────────────────────────────────────┐
│  HYBRID QUERY LAYER (DuckDB over LanceDB + BigQuery)                │
│  SQL prefilter → vector re-rank → geospatial filter → top-K          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  results
┌────────────────────────────────────────────────────────────────────┐
│  list of (device_id, ts_ns) tuples                                  │
│  → open in visualization tool at exact MCAP timestamp               │
└────────────────────────────────────────────────────────────────────┘
```

---

## The three data shapes (architectural fork)

| Shape | Tool | Optimized for | Used for |
|---|---|---|---|
| **Row-oriented** | MCAP (rosbag-style) | Small frequent writes; full-record reads | Original recording, replay in viz tool |
| **Column-oriented** | BigQuery / Parquet | Bulk analytical scans on a few columns | "find all left turns" — metadata SQL |
| **Vector** | LanceDB (columnar embeddings + ANN) | Similarity over high-dim space | "find scenes like THIS" — semantic |

The join key is `frame_id` — a deterministic hash of `{dataset, segment, ts_ns, camera}`.

---

## Phase 1 — Waymo via BigQuery

Waymo Open Perception v2 is a BigQuery public dataset. No download needed for metadata; mirror to your own GCS bucket if you want frames for embedding.

```bash
# Authenticate
gcloud auth application-default login

# Pick 150 segments
bq query --use_legacy_sql=false '
  SELECT DISTINCT segment_context_name
  FROM `bigquery-public-data.waymo_open_dataset_v_2_0_0.camera_image`
  WHERE location IN ("location_sf", "location_phx")
  LIMIT 150
' > segments.txt

# Extract camera frames to your bucket
bq extract --destination_format=PARQUET \
  bigquery-public-data:waymo_open_dataset_v_2_0_0.camera_image \
  gs://YOUR_BUCKET/parquet/camera/*.parquet
```

Sampling strategy: **1 Hz × 5 cameras = ~5,000 frames per 20-second segment, ~750K frames across 150 segments.** At higher sampling rates the cost balloons faster than the marginal information.

---

## Phase 2 — BDD100K download + frame extract

```bash
# After academic registration approval
# Filter to videos tagged 'pedestrian', 'night', 'rainy', 'weather:snow'
python src/ingest/bdd100k_filter.py --tags pedestrian,night,rainy --n 10000

# Extract @ 1 Hz
python src/ingest/extract_frames.py --rate 1 --in data/bdd100k/videos --out data/bdd100k/frames
```

---

## Phase 3 — Embedding

```python
# src/embed/siglip_batch.py
import torch
from transformers import AutoModel, AutoProcessor
import lancedb

model = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda().eval()
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

db = lancedb.connect("data/lance")
table = db.create_table("frames", schema=...)

# Batch process — ~200 images/sec on A100
for batch in iter_frames(batch_size=64):
    inputs = processor(images=batch.images, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = model.get_image_features(**inputs).half().cpu().numpy()
    table.add(zip(batch.frame_ids, emb, batch.metadata_refs))
```

Throughput: SigLIP-base on A100 = ~200 img/sec. 1M frames = ~85 min. 3.5M = ~5h pure compute, ~8h wall.

---

## Phase 4 — Hybrid query

```python
# src/query/hybrid.py
def query(text: str, sql_filter: str | None = None, k: int = 10):
    # 1. SQL prefilter on BigQuery metadata
    candidates = bq.query(f"""
        SELECT frame_id FROM metadata WHERE {sql_filter or "TRUE"}
    """) if sql_filter else None
    
    # 2. Text → embedding via SigLIP text encoder
    query_vec = siglip_text_encode(text)
    
    # 3. Vector search in LanceDB (optionally restricted to candidates)
    results = (
        lance.table("frames")
             .search(query_vec)
             .where(f"frame_id IN {candidates}" if candidates else None)
             .limit(k)
             .to_list()
    )
    
    return [(r["device_id"], r["ts_ns"]) for r in results]
```

Example:
```python
query("jaywalker at night",
      sql_filter="city='sf' AND time_of_day='night'",
      k=5)
# → [("waymo-...", 1672531200000000000), ...]
```

---

## Phase 5 — Visualization-tool integration

Each result tuple `(device_id, ts_ns)` opens the original MCAP recording at the exact moment in a visualization tool. Implementation uses the Foxglove Studio extension SDK — the public extension API for displaying custom panels.

See [`src/extension/`](../src/extension/).

---

## Reproducibility commands

```bash
make corpus-tier-a   # Waymo ingest + embed (1.5M frames, ~8h on A100)
make corpus-tier-b   # BDD100K ingest + embed (400K frames, ~3h)
make corpus-tier-c   # nuScenes (optional, ~6h)
make corpus-tier-d   # AV2 (optional, ~2h)
make index           # Build HNSW indexes
make query           # Launch query UI
make demo            # Run all 12 example needle hunts
```

---

## Throughput + scale targets

| Metric | Target |
|---|---|
| Embedding throughput | >200 img/sec on A100 |
| Index build (HNSW, 1M embeddings) | <10 min |
| Hybrid query p95 latency | <1 s (3.5M corpus) |
| Storage overhead vs raw frames | <1% (embeddings + thumbs) |

---

## Known failure modes + fixes

| Failure | Fix |
|---|---|
| SigLIP download fails | Fall back to CLIP ViT-L/14 (`open-clip-torch`) |
| GPU OOM | Halve batch size, use `torch.float16` |
| HNSW build too slow at 3M+ | Switch to IVF, accept ~5% recall hit |
| LanceDB write contention | Single writer process, parallel readers |
| Cross-dataset query returns junk | Inspect: usually a metadata-filter bug, not embedding quality |
