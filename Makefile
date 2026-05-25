# physical-ai-triage — reproducibility targets
# Targets mirror the commands documented in docs/pipeline.md and README.md.

SHELL := /bin/bash
PY := uv run python
ENV_FILE := .env

.PHONY: help setup auth corpus-tier-a corpus-tier-b corpus-tier-c corpus-tier-d \
	index query demo clean-data lint test

help:
	@echo "physical-ai-triage — Make targets"
	@echo ""
	@echo "  setup            Sync dependencies via uv (CPU defaults)"
	@echo "  auth             Show GCP auth + project status"
	@echo ""
	@echo "  corpus-tier-a    Waymo 150 segments → BQ scan + GCS mirror + embed"
	@echo "  corpus-tier-b    BDD100K 10K videos → frames + embed"
	@echo "  corpus-tier-c    nuScenes 1K scenes → frames + embed"
	@echo "  corpus-tier-d    Argoverse 2 1K scenarios → frames + embed"
	@echo ""
	@echo "  index            Build LanceDB ANN indexes on the embeddings"
	@echo "  query            Launch the hybrid query CLI"
	@echo "  demo             Run all 12 example needle hunts and log times"
	@echo ""
	@echo "  lint             ruff check + format"
	@echo "  test             pytest"

setup:
	uv sync

auth:
	@command -v gcloud >/dev/null 2>&1 || { \
	  echo "gcloud not installed."; exit 1; \
	}
	@command -v aws >/dev/null 2>&1 || { \
	  echo "aws CLI not installed."; exit 1; \
	}
	@echo "→ gcloud ADC (BQ source)"
	gcloud auth application-default print-access-token >/dev/null && echo "  ADC ok"
	@gcloud config get-value project 2>/dev/null || echo "  (no project set)"
	@echo "→ AWS identity (S3 sink)"
	aws sts get-caller-identity --output table

corpus-tier-a:
	$(PY) -m src.ingest.waymo_segments
	$(PY) -m src.embed.siglip_batch --table waymo

corpus-tier-b:
	$(PY) -m src.ingest.bdd100k_download --artifact both
	$(PY) -m src.ingest.bdd100k_labels --splits train,val
	$(PY) -m src.embed.siglip_batch --table bdd100k

corpus-tier-c:
	$(PY) -m src.ingest.nuscenes_ingest --scenes 1000 --out data/nuscenes
	$(PY) -m src.embed.siglip_batch --in data/nuscenes/frames --table nuscenes

corpus-tier-d:
	$(PY) -m src.ingest.av2_ingest --scenarios 1000 --out data/av2
	$(PY) -m src.embed.siglip_batch --in data/av2/frames --table av2

index:
	$(PY) -m src.index.build_hnsw --tables waymo,bdd100k,nuscenes,av2

query:
	$(PY) -m src.query.hybrid_cli

demo:
	$(PY) -m src.query.needle_hunt --all --log docs/findings.md

lint:
	uv run ruff check src
	uv run ruff format src

test:
	uv run pytest -q

clean-data:
	@echo "Refuses to delete. Move data/ to an archive folder manually if needed."
	@false
