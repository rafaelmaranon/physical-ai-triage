#!/usr/bin/env bash
# brev_run.sh — runs on Brev L4 instance to do: extract thumbnails + embed 3 models.
# Expects:
#   - Repo at $HOME/physical-ai-triage (copied via `brev copy` before this runs)
#   - ~/.aws/credentials + config copied
#   - ~/.config/gcloud/application_default_credentials.json copied
#   - .env at repo root with all required vars (BUCKET_URI, WAYMO_SOURCE_URI, etc.)

set -euo pipefail
cd "$HOME/physical-ai-triage"

# Load .env into the shell so inline python (uv run python -c ...) sees the vars.
# Scripts that use `from dotenv import load_dotenv` already work — this is for
# the bash-level checks + inline python below.
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "================================================================"
echo "[brev_run] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[brev_run] host: $(hostname) · GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'no GPU')"
echo "[brev_run] BUCKET_URI=$BUCKET_URI · WAYMO_SOURCE_URI=$WAYMO_SOURCE_URI"
echo "================================================================"

# 1. Install uv if not present (Brev base images vary)
if ! command -v uv >/dev/null 2>&1; then
  echo "[brev_run] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Set GOOGLE_APPLICATION_CREDENTIALS so gcsfs can read gs:// (Waymo source)
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "[brev_run] WARN: $GOOGLE_APPLICATION_CREDENTIALS missing — Waymo reads will fail"
fi

# 3. Sync deps (CUDA torch). uv detects GPU and picks the right wheels.
echo "[brev_run] uv sync..."
uv sync 2>&1 | tail -5

# 4. Verify GPU is visible to torch
uv run python -c "import torch; print(f'[brev_run] cuda available: {torch.cuda.is_available()} · device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 5. Verify cloud auth
uv run python -c "
import os
os.environ['EMBED_DEVICE'] = 'cuda'
from src.cloud import get_fs
gfs, gpath = get_fs(os.environ['WAYMO_SOURCE_URI'])
print(f'[brev_run] GCS access: {len(gfs.ls(gpath))} entries at {gpath}')
sfs, spath = get_fs(os.environ['BUCKET_URI'])
print(f'[brev_run] S3 access: bucket {spath} reachable')
" || {
  echo "[brev_run] ERROR: cloud auth check failed — bailing before paying for embed time"
  exit 2
}

# 6. EXTRACT THUMBNAILS — Waymo (full 100 segs) + BDD100K (density-filtered)
echo "================================================================"
echo "[brev_run] STAGE 1: extract thumbnails"
echo "================================================================"
uv run python -m src.ingest.extract_thumbnails waymo 2>&1 | tail -50
echo ""
uv run python -m src.ingest.extract_thumbnails bdd 2>&1 | tail -50

# 7. EMBED 3 models sequentially per Decision 23
echo "================================================================"
echo "[brev_run] STAGE 2: SigLIP (waymo + bdd100k)"
echo "================================================================"
uv run python -m src.embed.siglip_batch --table waymo \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 64 2>&1 | tail -30
uv run python -m src.embed.siglip_batch --table bdd100k \
  --metadata-uri "$BUCKET_URI/$BDD100K_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 64 2>&1 | tail -30

echo "================================================================"
echo "[brev_run] STAGE 3: CLIP ViT-L/14 (waymo + bdd100k)"
echo "================================================================"
uv run python -m src.embed.clip_batch --table waymo \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30
uv run python -m src.embed.clip_batch --table bdd100k \
  --metadata-uri "$BUCKET_URI/$BDD100K_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30

echo "================================================================"
echo "[brev_run] STAGE 4: DINOv2-large (waymo + bdd100k)"
echo "================================================================"
uv run python -m src.embed.dinov2_batch --table waymo \
  --metadata-uri "$BUCKET_URI/$WAYMO_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30
uv run python -m src.embed.dinov2_batch --table bdd100k \
  --metadata-uri "$BUCKET_URI/$BDD100K_PREFIX/metadata/thumbnails_index.parquet" \
  --batch 32 2>&1 | tail -30

# 8. Push LanceDB tables to S3 (each table is a directory)
echo "================================================================"
echo "[brev_run] STAGE 5: push LanceDB → S3"
echo "================================================================"
LANCE_DST="${BUCKET_URI#s3://}/${INDEX_PREFIX}/lance"
aws s3 sync data/lance "s3://${LANCE_DST}/" --quiet
echo "[brev_run] pushed to s3://${LANCE_DST}/"
aws s3 ls "s3://${LANCE_DST}/" --recursive --human-readable --summarize | tail -10

echo "================================================================"
echo "[brev_run] DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
