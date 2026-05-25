# src/ingest/

Per-dataset adapters. Each adapter is responsible for:

1. Pulling raw frames + scalar metadata from the source.
2. Computing the deterministic `frame_id` (see [`../frame_id.py`](../frame_id.py)).
3. Writing thumbnails to `gs://$GCS_BUCKET/thumbs/<dataset>/<frame_id>.jpg`.
4. Writing a metadata row to BigQuery in `<project>.avtri.metadata`.

## Modules

| Module | Source | Frames target | Notes |
|---|---|---|---|
| `waymo_segments.py` | BigQuery public dataset `bigquery-public-data.waymo_open_dataset_v_2_0_0` | ~750K (150 segments, 1 Hz × 5 cams) | No download — pulls via BQ extract. |
| `bdd100k_filter.py` | Academic download (`$BDD100K_ROOT`) | ~400K (10K videos, 1 Hz × 40s) | Filters by video-level tags before extract. |
| `nuscenes_ingest.py` | Academic download (`$NUSCENES_ROOT`) | ~1.4M (1000 scenes × 6 cams × 20s × 12Hz, downsampled) | |
| `av2_ingest.py` | S3 public (`s3://argoai-argoverse/`) | ~150K (1000 scenarios × 15s × 7 cams) | |

## Resumability

All adapters skip frames whose `frame_id` is already in the BQ metadata table. To force a re-ingest of a segment, delete those rows first — but **don't** delete the GCS thumbnails (the embeddings table is keyed on them).

## What this stage does NOT do

- No image normalization (SigLIP processor handles that).
- No quality filtering (broken frames stay in; they'll just embed badly).
- No cross-dataset deduplication (each tier is in its own BQ partition).
