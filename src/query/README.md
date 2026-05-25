# src/query/

The hybrid query layer. SQL prefilter → vector re-rank → top-K, with results returned as `(device_id, ts_ns)` tuples that open in a visualization tool.

## Modules

| Module | Purpose |
|---|---|
| `hybrid.py` | Programmatic API — `query(text, sql_filter, k)` returns ranked rows. |
| `hybrid_cli.py` | Interactive REPL over `hybrid.query` with rich-formatted result tables. |
| `needle_hunt.py` | Runs the 12 documented needle queries and writes timings to [`docs/findings.md`](../../docs/findings.md). |

## The query plan

```
text query  ─►  SigLIP text encoder  ─►  768d vector
                                              │
sql_filter ─►  BigQuery metadata scan  ─►  candidate frame_ids ──┐
                                                                  ▼
                                            LanceDB ANN search restricted to candidates
                                                                  │
                                                                  ▼
                                                    top-K (frame_id, score)
                                                                  │
                                                                  ▼ join metadata for display
                                                    (dataset, segment, device_id, ts_ns, thumb_uri)
```

When `sql_filter` is `None`, the planner runs vector-first on the full table — this is the right default for "looks like THIS frame" queries where structure isn't known up front.

## CLI usage

```bash
$ make query
> jaywalker at night in sf
                                  Top 10
┃ rank  ┃ score   ┃ dataset    ┃ city ┃ ts_ns
┃ 1     ┃ 0.412   ┃ bdd100k    ┃ sf   ┃ 1672531200000000000
...
> open 1
(launches viz tool at the timestamp)
```
