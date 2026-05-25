# src/embed/

SigLIP image-encoder pass over ingested frames. Writes 768-dimensional float16 vectors into LanceDB keyed by `frame_id`.

## Module

- `siglip_batch.py` — batched encoder loop. Reads frame paths from a BigQuery metadata table, pulls thumbnails from GCS, runs SigLIP-base on GPU, writes to LanceDB.

## Throughput

| Hardware | Model | Batch | Img/sec | 3.5M frames |
|---|---|---|---|---|
| A100 80GB | siglip-base-patch16-224 | 64 | ~200 | ~5h compute |
| A100 80GB | siglip-large-patch16-384 | 32 | ~70 | ~14h |
| Apple M-series (mps) | siglip-base | 16 | ~25 | smoke-test only |

## Output schema (LanceDB)

```
frame_id      string   (primary key)
dataset       string
segment       string
ts_ns         int64
camera        string
embedding     fixed_size_list<float16, 768>
thumb_uri     string   (gs:// URI)
```

## Recovery

If the process crashes mid-run, restart with the same args — already-embedded `frame_id`s are skipped (LanceDB `frame_id` is the dedupe key). The CLI also writes a `<table>.progress` file with the last successfully committed batch for observability.

## Why SigLIP and not CLIP

SigLIP's sigmoid loss makes its cross-domain similarity scores more linearly comparable across queries — useful when one query box has to rank frames from four different datasets. CLIP works fine if SigLIP downloads fail; fall back via `EMBED_MODEL=openai/clip-vit-large-patch14`.
