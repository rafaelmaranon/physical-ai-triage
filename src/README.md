# src/

Pipeline implementation. Each subdirectory is one pipeline stage; modules are runnable as `python -m src.<stage>.<module>` and wired through the [Makefile](../Makefile).

## Layout

| Dir | Stage | Entrypoint |
|---|---|---|
| [`ingest/`](ingest/) | Pull raw frames + metadata from each dataset into BigQuery + GCS | `python -m src.ingest.{waymo_segments,bdd100k_filter,nuscenes_ingest,av2_ingest}` |
| [`embed/`](embed/) | Run SigLIP over frames → 768d vectors → LanceDB | `python -m src.embed.siglip_batch` |
| [`index/`](index/) | Build HNSW / IVF indexes on the LanceDB tables | `python -m src.index.build_hnsw` |
| [`query/`](query/) | Hybrid SQL+vector query layer + CLI | `python -m src.query.hybrid_cli` |
| [`extension/`](extension/) | Foxglove Studio extension that opens `(device_id, ts_ns)` results in the viz tool | `cd src/extension && npm run build` |

## Shared utilities

- `frame_id.py` — deterministic hash of `{dataset, segment, ts_ns, camera}`. The join key across BigQuery, LanceDB, and MCAP.

## Conventions

- Each module exposes a `typer` CLI. Run with `--help` for flags.
- All paths and credentials come from environment variables (see [`.env.example`](../.env.example)). No hardcoded secrets.
- Reads are idempotent; writes are append-only with deterministic keys. Re-running an ingest will skip already-embedded `frame_id`s.
- Heavy compute (embeddings) is GPU-only by default. Set `EMBED_DEVICE=cpu` for smoke tests on a laptop — expect ~50x slowdown.
