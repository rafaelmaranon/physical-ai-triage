# Corpus Plan

**Question:** How many frames, from which datasets, do you need to make a vector-search demo *actually meaningful* on autonomous-driving data?
**Answer:** ~3.5M frames across 4 datasets covers the demo. Architecture supports 100M+ for "production scale." **Waymo alone is insufficient** — by design, it lacks the long-tail driving events (near-miss, jaywalking, severe weather) that semantic search is most valuable for finding.

---

## The core insight

> **Curated dataset ≠ search-worthy corpus.** Public AV datasets like Waymo are published for ML training and have been filtered to remove disengagements, near-collisions, and edge-case chaos. They are sterile by construction. Semantic similarity search earns its keep where the corpus *contains the long-tail events worth searching for*. To demonstrate the value, you have to mix a "wild" dataset (BDD100K) with a "clean" one (Waymo).

This is the empirical point this project is structured to demonstrate.

---

## Vector DB scale thresholds (image embeddings, ~768-d)

| Corpus size | Why it matters | Tool fit |
|---|---|---|
| 10K | Anything works (brute-force <100 ms) | NumPy / scikit-learn |
| **100K** | ANN starts to matter vs flat scan; demo floor | LanceDB / FAISS flat |
| **1M** | HNSW / IVF mandatory; "real" vector territory | LanceDB / Qdrant / pgvector |
| **10M+** | Production regime for a robotics company | LanceDB / Vertex Vector Search / Milvus |
| **100M–1B** | Petabyte-corpus production scale | Distributed / managed (Pinecone serverless / Vertex) |

For this project: target ~3.5M for the demo (above the 1M "real vector" threshold), with architecture that holds up to 100M.

---

## The 4 corpora

### Tier A — Waymo Open Perception v2

- **Source:** `bigquery-public-data.waymo_open_dataset_v_2_0_0` (BigQuery public dataset)
- **Subset for this project:** ~150 segments out of ~1,950 (≈ 30% of v2 Perception) = **~2 TB raw**
- **Sampling:** 1 Hz × 5 cameras × 150 segments = **~750K frames** + ~150K LiDAR sweeps
- **Sensors:** 5 cameras (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT) + 5 LiDARs
- **Geography:** San Francisco, Phoenix, Mountain View
- **Strengths:** sensor-complete, high-quality labels, multi-modal
- **Gaps:** no near-miss, no disengagement, no jaywalking — filtered for ML training

### Tier B — Berkeley DeepDrive (BDD100K)

- **Source:** `https://bdd-data.berkeley.edu/` (academic registration required)
- **Full size:** 100K videos × 40s × 30 fps ≈ 120M frames
- **Subset for this project:** 10K videos × 1 Hz × 40s = **~400K frames**
- **Sensors:** single front dashcam (no LiDAR)
- **Geography:** NYC, SF, Bay Area, suburban
- **Strengths:** real-world chaos — jaywalking, weather (rain/snow), night, congestion, weaving cyclists
- **Why critical:** Waymo alone returns nothing for "find a near-miss." BDD100K is where the long-tail lives.

### Tier C — nuScenes

- **Source:** `https://www.nuscenes.org/` (academic registration; ~350 GB full)
- **Subset:** 1,000 scenes × 20s = **~1.4M images**, 390K LiDAR sweeps
- **Sensors:** 6 cameras + 1 LiDAR + 5 radars + GPS + IMU
- **Geography:** Boston + Singapore (monsoon, dense urban, right-hand drive)
- **Strengths:** geography diversity, radar fusion, weather

### Tier D — Argoverse 2 Sensor

- **Source:** `https://www.argoverse.org/av2.html` (free, AWS S3 public)
- **Subset:** 1,000 scenarios × 15s = **~150K frames**
- **Geography:** Pittsburgh, Miami, Austin, Detroit, Palo Alto, Washington DC
- **Strengths:** US-city diversity beyond Waymo's 3 cities

### Tier E — Architecture-only (not ingested, on roadmap)

| Dataset | Why interesting |
|---|---|
| Full BDD100K | 120M frame ceiling |
| Comma2k19 | 33h of CA-280 real-driver behavior + corner cases |
| Mapillary Vistas | 25K street-level images, global |
| CARLA synthetic | Unlimited near-miss generation (sim2real) |

---

## Recommended ingest order

| Order | Tier | Frames | Time |
|---|---|---|---|
| 1 | A — Waymo 150 segments | ~750K | 4-8h via BigQuery → parquet + overnight GPU embed |
| 2 | B — BDD100K 10K videos | ~400K | 4h download + 6h frame extract + overnight embed |
| 3 | C — nuScenes 1K scenes | ~1.4M | 3h download + overnight embed |
| 4 | D — AV2 1K scenarios | ~150K | 2h download + 2h embed |

**Demo floor:** A + B (~1.15M frames, supports the near-miss story). **Stretch:** all four (~2.7M frames).

---

## What's worth NOT doing

- **Don't ingest the entire Waymo dataset (~25 TB).** A 2 TB sample is plenty for the architectural story.
- **Don't run embeddings on CPU.** GPU bursts (e.g. cloud L4/A100) are much faster than CPU-bound work.
- **Don't normalize across datasets prematurely.** Each tier in its own table, cross-corpus join is a Day 3 demo, not an ingest dependency.
- **Don't ingest full BDD100K.** 10K videos out of 100K is sufficient — saves significant compute + days of wall time.
