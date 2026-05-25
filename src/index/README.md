# src/index/

Build ANN indexes on the LanceDB embedding tables.

## Module

- `build_hnsw.py` — builds HNSW indexes on each table named `<dataset>`. Falls back to IVF_PQ if HNSW build time exceeds the `--max-build-min` ceiling (default 15 min per table).

## Choice of index

| Index | Build time @ 1M | Recall@10 | Memory | When to use |
|---|---|---|---|---|
| HNSW (M=16, efC=128) | ~3 min | ~0.99 | ~1.5 GB | Default. Up to ~5M vectors. |
| IVF_PQ (nlist=4096, m=8) | ~8 min | ~0.92 | ~200 MB | 10M+ vectors. Memory-constrained. |

LanceDB chooses the implementation; this module just sets parameters.

## Reproducibility note

Index parameters end up in [`docs/cost.md`](../../docs/cost.md) and [`docs/pipeline.md`](../../docs/pipeline.md). If you change them, update both.
