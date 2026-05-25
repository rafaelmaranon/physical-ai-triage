# physical-ai-triage

> A 6-model retrieval bake-off + metadata-lift experiment on 95K autonomous-driving frames.
> **Tests the question:** in robotics retrieval, is the bottleneck the encoder — or the structure around it?

**License:** MIT

![physical-ai-triage architecture: 5-stage pipeline showing ingest (Waymo + BDD100K frames) → perception (image encoders + VLMs) → cached perception layer (LanceDB vectors + Parquet columns) → query app (Flask + Foxglove deeplinks) → eval (12-needle harness), with a feedback loop](media/screenshots/architecture.png)

---

## What this is

A reproducible pipeline that:

1. **Ingests** open robot/AV data (Waymo Open · BDD100K · raw MCAP segments) → 95,571 frames
2. **Embeds** each frame with 6 retrieval models (SigLIP · CLIP · DINOv2 · Cosmos-Reason-1 · Cosmos-Reason-2 · Cosmos Embed1) into LanceDB
3. **Searches** them via a Flask UI with three modes: pure vector, structured filter, and hybrid
4. **Measures** retrieval quality against 12 hand-picked needles (jaywalker at night, school bus stop arm, etc.)
5. **Visualizes** results with Foxglove deeplinks that open the exact MCAP at the right timestamp

The point isn't to ship a product. It's to **measure where retrieval over robot logs actually breaks**, with primary sources cited.

---

## Demo

![physical-ai-triage demo: 16-second walkthrough of a 6-model retrieval bake-off (SigLIP, CLIP, DINOv2, Cosmos-Reason 1/2, Cosmos Embed1) across 3 autonomous-driving queries, the system architecture, and metadata-lift findings showing pure vector vs hybrid filter precision](media/video/demo.gif)

*16 seconds, 7 frames. Encoder bake-off across 3 queries → architecture overview → metadata-lift findings on 3 queries.*

---

## Two findings worth your attention

### 1. Encoder choice plateaus

5 of 6 retrieval models capped at **≤33% coverage** on the same 12 needles. The bigger model (Cosmos-Reason-1 7B → Cosmos-Reason-2 8B) did not help. The plateau is structural, not a model-quality problem.

| Model | Coverage @ K=10 |
|---|---|
| DINOv2-large (image-only, no text encoder) | 8% |
| Cosmos-Reason-1 (7B) | 25% |
| Cosmos-Reason-2 (8B) | 25% |
| CLIP ViT-L/14 | 33% |
| SigLIP-base | 33% |
| Cosmos Embed1-336p | 83% * |

*\* Cosmos Embed1 measured on a different 250-clip subset; not directly comparable to the others.*

### 2. The fix is architectural, not a bigger model

Same SigLIP encoder. Same corpus. Add a metadata pre-filter derived from Cosmos-Reason-2 captions (`time_of_day`, `has_pedestrian`). Vector search now ranks within the filtered set.

For the query *"pedestrian at night, poor visibility"*:
- **Pure vector** returned **8 of 10** frames that violated the query's stated structured attributes (daytime scenes, no pedestrians)
- **Hybrid (metadata + vector)** returned frames that respect the filter

**Honest methodological caveat:** the filter and the predicate share derivation (both come from regex on the same caption text), so hybrid scoring 1.00 on those exact predicates is tautological by construction. The defensible claim is therefore *"pure vector search structurally cannot guarantee a structured attribute is respected; a structured filter mechanically can"* — not *"hybrid is N× smarter at ranking."*

**The thesis:** vector search is being asked to do two jobs at once — perception (*"is this at night?"*) and retrieval (*"find similar frames"*). For compound attributes, one cosine number cannot do both well. The fix is separation of concerns: do perception once at ingest with a VLM, cache as structured columns, reserve vector search for similarity.

---

## Architecture

![Detailed physical-ai-triage architecture diagram: five color-coded zones — Sources (Waymo + BDD100K + local MCAPs), Perception Compute (image encoders + VLMs on Mac CPU and Brev L4 GPU), Cached Perception (LanceDB + Parquet + Snowflake — the moat), Query App (Flask + lift demo + Foxglove deeplinks), and Evaluation (12-needle harness + metadata-lift experiment) — connected by data-flow arrows and a feedback loop from eval back to schema](media/screenshots/architecture-detailed.png)

Five stages. Perception happens once at ingest and is cached. Retrieval happens per request using the cached perception as a filter. Eval findings feed the next schema update — that is the flywheel.

---

## Tech stack

Every tool used in this build, grouped by layer:

| Layer | Tools |
|---|---|
| **Models — image encoders** | [SigLIP-base](https://huggingface.co/google/siglip-base-patch16-224) · [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) · [DINOv2-large](https://huggingface.co/facebook/dinov2-large) · [Cosmos Embed1-336p](https://huggingface.co/nvidia/Cosmos-Embed1-336p) |
| **Models — VLMs** | [Cosmos-Reason-1 (7B)](https://huggingface.co/nvidia/Cosmos-Reason1-7B) · [Cosmos-Reason-2 (8B)](https://huggingface.co/nvidia/Cosmos-Reason2-8B) · Anthropic Claude (query router) |
| **Data formats** | [MCAP](https://mcap.dev/) · Apache Parquet · JSON |
| **Storage / index** | [LanceDB](https://lancedb.com/) · [DuckDB](https://duckdb.org/) · AWS S3 · Google Cloud Storage |
| **Datasets** | [Waymo Open Dataset](https://waymo.com/open/) · [BDD100K](https://www.bdd100k.com/) · raw MCAP samples |
| **Compute** | Mac CPU (local) · [Brev](https://brev.dev/) L4 GPU bursts |
| **App / query** | [Flask](https://flask.palletsprojects.com/) · [Anthropic Claude](https://docs.anthropic.com/) (NL→SQL routing) |
| **Visualization / replay** | [Foxglove Studio](https://foxglove.dev/) (desktop + cloud) — opened via deeplinks at exact MCAP timestamps |
| **Observability / BI** | [Snowflake](https://www.snowflake.com/) external tables (BI lane) · [Datadog](https://www.datadoghq.com/) metrics (latency, throughput) |
| **Dev tooling** | Python 3.12 · [uv](https://docs.astral.sh/uv/) · [HuggingFace Transformers](https://huggingface.co/docs/transformers) · [Typer](https://typer.tiangolo.com/) · PyArrow · NumPy · Pillow |
| **Media generation** | [ImageMagick](https://imagemagick.org/) · [FFmpeg](https://ffmpeg.org/) |

---

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/rafaelmaranon/physical-ai-triage.git
cd physical-ai-triage
uv sync

# 2. Configure your own buckets + keys
cp .env.example .env
# edit .env — set BUCKET_URI, WAYMO_SOURCE_URI, AWS creds, ANTHROPIC_API_KEY

# 3. (Optional) Run the metadata-lift experiment standalone
uv run python eval/metadata_lift_experiment.py
# → eval/metadata_lift_results.json

# 4. Launch the live search UI
uv run python -m src.server.search
# → http://127.0.0.1:5050
# Three routes:
#   /         — single-model search
#   /compare  — 6 models side-by-side on the same query
#   /llm      — Claude-routed query (NL → SQL + vector)

# 5. Launch the lift demo (pure vs hybrid side-by-side)
uv run python eval/lift_demo_server.py
# → http://127.0.0.1:5051
```

**Bring your own data:** the corpus is regenerable from public sources. Waymo Open requires a free Google account; BDD100K is an academic download. The pipeline reads from S3/GCS via env vars — no hardcoded bucket names.

---

## Reproducing the experiments

### The 6-model bake-off
```bash
uv run python -m src.eval.run --needles eval/needles.json --k 10
# → eval/results.json
```

### The metadata-lift experiment
```bash
uv run python eval/metadata_lift_experiment.py
# → eval/metadata_lift_results.json
```

Both write structured JSON. Inspect with `jq`, render with the included `dashboard/` HTMLs, or import into a notebook.

---

## Repo structure

```
physical-ai-triage/
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── pyproject.toml + uv.lock   ← dependencies
├── Makefile                   ← shortcuts
├── .env.example               ← config template
├── src/
│   ├── ingest/                ← Waymo / BDD100K / MCAP extraction
│   ├── embed/                 ← per-model batch embedders
│   ├── query/                 ← hybrid SQL + vector query
│   ├── server/                ← Flask UI (search, compare, llm)
│   ├── eval/                  ← 12-needle harness
│   ├── integrations/          ← Snowflake + Datadog
│   └── cloud.py               ← S3/GCS abstraction
├── eval/
│   ├── needles.json           ← 12 needle queries
│   ├── results.json           ← bake-off output
│   ├── metadata_lift_experiment.py
│   ├── metadata_lift_results.json
│   └── lift_demo_server.py    ← live side-by-side demo
├── dashboard/
│   ├── architecture.html      ← system overview
│   └── architecture-detailed.html
├── docs/
│   ├── overview.md            ← design narrative
│   ├── corpus.md              ← dataset breakdown
│   ├── pipeline.md            ← data flow
│   ├── findings.md            ← needle-by-needle results
│   ├── snowflake.md           ← BI integration setup
│   └── datadog.md             ← metrics integration setup
├── scripts/                   ← orchestrators (Brev runs, viz)
└── media/
    ├── screenshots/           ← bake-off + lift-demo PNGs
    └── video/                 ← demo.gif + demo.mp4
```

---

## Reading list

Every VLA breakthrough depends on retrieval over robot logs — to find training hours, debug failures, fine-tune on edge cases. End-to-end models (whether retrieval or policy) struggle to guarantee structural properties; the fix in both cases is an upstream cache of structured state. Better policy models raise the action ceiling. Better retrieval architecture raises the data floor. Both, or neither.

### Policy / action models (VLAs — depend on retrieval for their data work)

- **π₀.₇** (Physical Intelligence, Apr 2026) — 5B-param steerable VLA, conditioned at inference via structured metadata. [arxiv.org/abs/2604.15483](https://arxiv.org/abs/2604.15483)
- **GEN-1** (Generalist AI, Apr 2026) — 99% vs 64% SOTA on cross-embodiment tasks from 1 hr robot data + 500K hours human wearable video. [Official blog](https://generalistai.com/blog/apr-02-2026-GEN-1)
- **OpenVLA** (Stanford + TRI, 2024) — the open VLA baseline most teams fork. [openvla.github.io](https://openvla.github.io)

### Retrieval reference implementations (directly comparable to this work)

NVIDIA shipped two reference implementations for video search over robot data. This repo sits next to them as a *comparative receipt* across 6 retrieval backbones:

- **VSS (Video Search and Summarization) Blueprint** — agentic, streaming, MCP-based. [GitHub](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) · [blueprint card](https://build.nvidia.com/nvidia/video-search-and-summarization/blueprintcard) · [docs](https://docs.nvidia.com/vss/latest/)
- **Cosmos Dataset Search (CDS)** — batch semantic search using Cosmos-Embed1 NIM. [GitHub](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search) · [Cosmos-Embed1 docs](https://docs.nvidia.com/nim/cosmos-embed1/1.0.0/introduction.html)

---

## Limitations

- Coverage metric measures "did the model return any candidate at all," not recall against ground truth. Ground truth labels exist for ~30% of needles; the rest are visual-inspection-only.
- The metadata-lift experiment uses Cosmos-Reason-2 captions as both label source and filter input. A production deployment would use independent labelers per attribute to avoid tautology.
- Cosmos Embed1 was benchmarked on a different 250-clip subset due to its video-temporal architecture. Numbers are not directly comparable to the per-frame encoders.
- This is a research bench, not a production system. No auth, no horizontal scaling, no SLA.

---

## License

[MIT](LICENSE)
