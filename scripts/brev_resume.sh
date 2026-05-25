#!/usr/bin/env bash
# brev_resume.sh — resume from after Waymo extract.
# Waymo thumbs are already in S3 (49,480 thumbnails). Skip waymo extract;
# do BDD extract + 3 embed × 2 datasets + LanceDB push.

set -euo pipefail
cd "$HOME/physical-ai-triage"

set -a
# shellcheck disable=SC1091
source .env
set +a

echo "================================================================"
echo "[brev_resume] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[brev_resume] skipping Waymo extract (49,480 thumbs already in S3)"
echo "================================================================"

# Ensure deps + GCS creds
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
export PATH="$HOME/.local/bin:$PATH"
uv sync 2>&1 | tail -3

echo "================================================================"
echo "[brev_resume] STAGE 1: BDD100K extract (with fix for labels file selection)"
echo "================================================================"
uv run python -m src.ingest.extract_thumbnails bdd 2>&1 | tail -50

echo "================================================================"
echo "[brev_resume] STAGE 2: SigLIP embed (waymo + bdd100k)"
echo "================================================================"
uv run python -m src.embed.siglip_batch --table waymo \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 64 2>&1 | tail -30
uv run python -m src.embed.siglip_batch --table bdd100k \
  --metadata-uri "$BUCKET_URI/$BDD100K_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 64 2>&1 | tail -30

echo "================================================================"
echo "[brev_resume] STAGE 3: CLIP ViT-L/14 (waymo + bdd100k)"
echo "================================================================"
uv run python -m src.embed.clip_batch --table waymo \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30
uv run python -m src.embed.clip_batch --table bdd100k \
  --metadata-uri "$BUCKET_URI/$BDD100K_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30

echo "================================================================"
echo "[brev_resume] STAGE 4: DINOv2-large (waymo + bdd100k)"
echo "================================================================"
uv run python -m src.embed.dinov2_batch --table waymo \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30
uv run python -m src.embed.dinov2_batch --table bdd100k \
  --metadata-uri "$BUCKET_URI/$BDD100K_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30

echo "================================================================"
echo "[brev_resume] STAGE 5: object-BEV extract + embed (Waymo only — needs LiDAR)"
echo "================================================================"
uv run python -m src.ingest.extract_object_bev 2>&1 | tail -30
uv run python -m src.embed.siglip_batch --table object_bev \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/object_bev_index.parquet" \
  --batch 64 2>&1 | tail -30

echo "================================================================"
echo "[brev_resume] STAGE 6: push LanceDB → S3"
echo "================================================================"
LANCE_DST="${BUCKET_URI#s3://}/${INDEX_PREFIX}/lance"
aws s3 sync data/lance "s3://${LANCE_DST}/" --quiet
echo "[brev_resume] pushed to s3://${LANCE_DST}/"
aws s3 ls "s3://${LANCE_DST}/" --recursive --human-readable --summarize | tail -10

echo "================================================================"
echo "[brev_resume] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
