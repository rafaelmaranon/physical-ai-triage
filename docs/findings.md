# Findings

The measured numbers behind the project. Raw data is in `eval/` — this doc summarizes what they show.

---

## The 12 needles

Hand-picked compound queries that an AV/robotics team would realistically want to find in their corpus. Defined in [`eval/needles.json`](../eval/needles.json).

| # | Needle | Filter components |
|---|---|---|
| 1 | Unprotected left turn + pedestrian | turn type · pedestrian presence · location |
| 2 | Near-miss with pedestrian | proximity event · pedestrian |
| 3 | Bumpy ride cause | IMU spike |
| 4 | Cyclist weaving | cyclist presence · lateral motion |
| 5 | Construction zone | scene type |
| 6 | Pedestrian at night | time of day · pedestrian |
| 7 | School bus with stop arm | object class |
| 8 | Emergency vehicle | object class |
| 9 | Jaywalking | pedestrian outside crosswalk |
| 10 | "Looks like THIS frame" | image-seeded similarity |
| 11 | Wet road reflections at night | weather · time of day |
| 12 | Eval set creation (image-seeded) | image-seeded similarity |

---

## Finding 1 — Encoder choice plateaus

All 6 retrieval backbones tested on the same 12 needles. Coverage @ K=10:

| Model | Coverage |
|---|---|
| DINOv2-large (image-only, no text encoder) | 8% |
| Cosmos-Reason-1 (7B) | 25% |
| Cosmos-Reason-2 (8B) | 25% |
| CLIP ViT-L/14 | 33% |
| SigLIP-base | 33% |
| Cosmos Embed1-336p | 83%* |

*\* measured on a different 250-clip subset; not directly comparable to the per-frame encoders.*

**Raw output:** [`eval/results.json`](../eval/results.json) · [`eval/results.parquet`](../eval/results.parquet)

Bigger model did not help (Cosmos-R1 7B → R2 8B was identical). The plateau is structural, not a model-quality issue.

---

## Finding 2 — Vector search leaks structured attributes

Same SigLIP encoder. Same corpus. The difference: a metadata pre-filter derived from Cosmos-Reason-2 captions.

For the query *"pedestrian at night, poor visibility"*:

| Approach | Precision@10 |
|---|---|
| Pure SigLIP (vector only) | 0.20 |
| Hybrid (`time_of_day='night' AND has_pedestrian=true` → SigLIP rank within) | 1.00 |

Pure vector returned 8 of 10 frames that did NOT satisfy the query's stated structured attributes (daytime scenes, scenes without pedestrians).

**Raw output:** [`eval/metadata_lift_results.json`](../eval/metadata_lift_results.json)

**Methodological caveat:** the filter and the predicate share derivation (both come from regex on the same caption text). Hybrid scoring 1.00 on that exact predicate is therefore tautological by construction. The defensible claim is *"pure vector search structurally cannot guarantee a structured attribute is respected; a structured filter mechanically can"* — not *"hybrid is N× smarter at ranking."*

---

## What the two findings imply

Vector search is doing two jobs at once — perception (*"is this at night?"*) and retrieval (*"find similar"*). For compound attribute queries, one cosine number can't do both well. The architectural answer is separation of concerns: do perception ONCE at ingest with a VLM, cache the result as structured columns, reserve vector search for similarity ranking.

---

## When SQL filters beat vector search

Structured attributes that are reliably labeled at ingest — `time_of_day`, `city`, `weather`, `has_pedestrian`, `num_cyclists`, `ego_speed_mps`. Vector search smears these into a single similarity number; a WHERE clause guarantees they're respected.

## When vector search beats SQL

Compositional or unlabeled properties. *"Looks like a near-miss."* *"Scene reminds me of THIS one."* No reasonable label exists at ingest time for these, so similarity in embedding space is the only tool.

## When hybrid is the only option

Almost every real-world query. SQL alone is too broad (returns 10K matching frames, none of them visually right). Vector alone returns too much off-attribute junk (the leak in Finding 2). A query planner that decides which filter to apply first IS the product.

---

## Related work — NVIDIA's two reference implementations

Two NVIDIA-authored reference implementations cover the same retrieval-over-video problem with different architectures. `physical-ai-triage` sits *complementary* to both — different problem shapes, different parts of the same physical-AI retrieval stack.

### NVIDIA Video Search and Summarization (VSS) Blueprint

- **Repo:** [github.com/NVIDIA-AI-Blueprints/video-search-and-summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
- **Stack:** Cosmos-Reason2-8B (VLM) + Nemotron-Nano-9B-v2 (LLM) + Model Context Protocol (MCP) agents + NIM microservices
- **Mode:** hybrid streaming + batch, agentic verification loop
- **Audience:** production deployments — smart-space monitoring, warehouse automation, SOP validation

### NVIDIA Cosmos Dataset Search (community example)

- **Repo:** [github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search)
- **Stack:** Cosmos Embed1 (video embedder, NIM-served) + Qdrant + weighted blending (VIDEO_WEIGHT=0.6 / TEXT_WEIGHT=0.4)
- **Mode:** batch indexing
- **Audience:** developer-friendly entry point to the NVIDIA stack
- **Example dataset:** PhysicalAI-Robotics-GR00T-GR1

### How `physical-ai-triage` differs

|   | VSS Blueprint | Cosmos Dataset Search | physical-ai-triage |
|---|---|---|---|
| Mode | streaming + agentic | batch | batch + hybrid SQL/vector |
| Stack | Cosmos-R2 + Nemotron + MCP | Cosmos Embed1 + Qdrant | 6 backbones tested + LanceDB + DuckDB |
| Audience | production deployment | NVIDIA-stack developer | retrieval architecture decision-makers |
| Verification | agentic LLM check | none (ships one stack) | structured-filter + Cosmos-R2 caption verification |
| Comparative eval | none (ships one stack) | none (ships one stack) | 6-model bake-off published |

The takeaway is not "`physical-ai-triage` is better" — it's that the two NVIDIA implementations each ship ONE recommended path, and teams choosing a retrieval substrate need the comparative data those implementations don't publish. This project fills that gap on AV-specific data.
