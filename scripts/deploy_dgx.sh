#!/usr/bin/env bash
# =============================================================================
# deploy_dgx.sh  —  One-command setup for OmniSight + Alpamayo on DGX Spark
# =============================================================================
#
# Usage:
#   bash scripts/deploy_dgx.sh
#   bash scripts/deploy_dgx.sh --skip-model-download   # model already on disk
#   bash scripts/deploy_dgx.sh --skip-data-download    # AV2 data already on disk
#   bash scripts/deploy_dgx.sh --tensor-parallel 2     # use 2 GPUs
#
# What it does:
#   1. Creates a conda environment 'omnisight' with all dependencies
#   2. Downloads the Alpamayo-R1-10B model from HuggingFace (~20 GB)
#   3. Downloads ONE AV2 sensor scene (~1-2 GB) via av2 CLI
#   4. Starts the vLLM inference server     (tmux: alpamayo-server)
#   5. Starts the OmniSight web data viewer (tmux: omnisight-viewer)
#
# After running:
#   • Web viewer:      http://localhost:7860  (SSH tunnel from laptop)
#   • Inference API:   http://localhost:8000/v1
#   • Run inference:   python -m alpamayo.inference --data-dir $AV2_DATA_DIR --task summary
#
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ENV_NAME="omnisight"
MODEL_HF_ID="nvidia/Alpamayo-R1-10B"
MODEL_LOCAL_DIR="${HOME}/models/Alpamayo-R1-10B"
AV2_DATA_DIR="/raid/av2/sensor/val"
VLLM_PORT=8000
VIEWER_PORT=7860
TENSOR_PARALLEL=1
SKIP_MODEL_DOWNLOAD=false
SKIP_DATA_DOWNLOAD=false
TMUX_SERVER="alpamayo-server"
TMUX_VIEWER="omnisight-viewer"

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)           MODEL_LOCAL_DIR="$2"; SKIP_MODEL_DOWNLOAD=true; shift 2 ;;
    --skip-model-download)  SKIP_MODEL_DOWNLOAD=true; shift ;;
    --av2-data-dir)         AV2_DATA_DIR="$2"; SKIP_DATA_DOWNLOAD=true; shift 2 ;;
    --skip-data-download)   SKIP_DATA_DOWNLOAD=true; shift ;;
    --port)                 VLLM_PORT="$2"; shift 2 ;;
    --viewer-port)          VIEWER_PORT="$2"; shift 2 ;;
    --tensor-parallel)      TENSOR_PARALLEL="$2"; shift 2 ;;
    --env)                  ENV_NAME="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  OmniSight + Alpamayo-R1-10B — DGX Spark Setup"
echo "============================================================"
echo "  Conda env      : ${ENV_NAME}"
echo "  Model dir      : ${MODEL_LOCAL_DIR}"
echo "  AV2 data dir   : ${AV2_DATA_DIR}"
echo "  vLLM port      : ${VLLM_PORT}"
echo "  Viewer port    : ${VIEWER_PORT}"
echo "  Tensor parallel: ${TENSOR_PARALLEL}"
echo "============================================================"

# ── Step 1: Conda environment ─────────────────────────────────────────────────
echo ""
echo "[1/5] Setting up conda environment '${ENV_NAME}'..."

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "  -> Environment '${ENV_NAME}' already exists, skipping."
else
  conda create -n "${ENV_NAME}" python=3.11 -y
  echo "  -> Created environment '${ENV_NAME}'"
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Step 2: Install dependencies ──────────────────────────────────────────────
echo ""
echo "[2/5] Installing Python dependencies..."

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "  -> Dependencies installed."

# ── Step 3: Download Alpamayo model ───────────────────────────────────────────
if [ "${SKIP_MODEL_DOWNLOAD}" = false ]; then
  echo ""
  echo "[3/5] Downloading ${MODEL_HF_ID} (~20 GB, may take 20-40 min)..."
  mkdir -p "${MODEL_LOCAL_DIR}"
  huggingface-cli download "${MODEL_HF_ID}" \
    --local-dir "${MODEL_LOCAL_DIR}" \
    --local-dir-use-symlinks False
  echo "  -> Model saved to ${MODEL_LOCAL_DIR}"
else
  echo ""
  echo "[3/5] Skipping model download. Using: ${MODEL_LOCAL_DIR}"
fi

# ── Step 4: Download one AV2 scene ────────────────────────────────────────────
if [ "${SKIP_DATA_DOWNLOAD}" = false ]; then
  echo ""
  echo "[4/5] Downloading one AV2 sensor scene (~1-2 GB)..."
  echo "      This uses the av2 Python package's built-in downloader."
  echo ""

  mkdir -p "${AV2_DATA_DIR}"

  # Download one AV2 scene via s5cmd (public S3 bucket, no credentials needed)
  AV2_LOG_ID="0c6e62d7-bdfa-3061-8d3d-03b13aa21f68"

  if ! command -v s5cmd &>/dev/null; then
    echo "  -> Installing s5cmd..."
    pip install s5cmd -q
  fi

  echo "  -> Downloading scene: ${AV2_LOG_ID}"
  s5cmd --no-sign-request cp \
    "s3://argoai-argoverse2/sensor/val/${AV2_LOG_ID}/*" \
    "${AV2_DATA_DIR}/${AV2_LOG_ID}/"

  echo "  -> Scene saved to ${AV2_DATA_DIR}/${AV2_LOG_ID}/"

else
  echo ""
  echo "[4/5] Skipping data download. Using: ${AV2_DATA_DIR}"
fi

# Check that data dir has at least one scene
if [ ! -d "${AV2_DATA_DIR}" ] || [ -z "$(ls -A "${AV2_DATA_DIR}" 2>/dev/null)" ]; then
  echo ""
  echo "  [WARNING] AV2 data directory is empty or missing: ${AV2_DATA_DIR}"
  echo "  The viewer will start but show no scenes until data is placed there."
fi

# ── Step 5: Start services in tmux ────────────────────────────────────────────
echo ""
echo "[5/5] Starting services in tmux..."

# Kill old sessions if present
tmux kill-session -t "${TMUX_SERVER}" 2>/dev/null || true
tmux kill-session -t "${TMUX_VIEWER}" 2>/dev/null || true

# vLLM inference server
tmux new-session -d -s "${TMUX_SERVER}" \
  "conda activate ${ENV_NAME} && \
   python -m alpamayo.server \
     --model-path ${MODEL_LOCAL_DIR} \
     --port ${VLLM_PORT} \
     --tensor-parallel ${TENSOR_PARALLEL} \
   2>&1 | tee ~/alpamayo-server.log"

echo "  -> vLLM server starting in tmux: ${TMUX_SERVER}"

# Web data viewer
tmux new-session -d -s "${TMUX_VIEWER}" \
  "conda activate ${ENV_NAME} && \
   python viewer.py \
     --data-dir ${AV2_DATA_DIR} \
     --port ${VIEWER_PORT} \
   2>&1 | tee ~/omnisight-viewer.log"

echo "  -> Web viewer starting in tmux: ${TMUX_VIEWER}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Deployment complete!"
echo ""
echo "  Monitor services:"
echo "    tmux attach -t ${TMUX_SERVER}    # vLLM server"
echo "    tmux attach -t ${TMUX_VIEWER}    # web viewer"
echo ""
echo "  Logs:"
echo "    tail -f ~/alpamayo-server.log"
echo "    tail -f ~/omnisight-viewer.log"
echo ""
echo "  Web viewer (from your laptop — SSH tunnel first):"
echo "    ssh -L ${VIEWER_PORT}:localhost:${VIEWER_PORT} user@<dgx-ip>"
echo "    open http://localhost:${VIEWER_PORT}"
echo ""
echo "  Check inference server:"
echo "    curl http://localhost:${VLLM_PORT}/v1/models"
echo ""
echo "  Run inference on a scene:"
echo "    python -m alpamayo.inference \\"
echo "      --data-dir ${AV2_DATA_DIR} \\"
echo "      --task summary"
echo "============================================================"
echo ""
echo "  Data size reference — one AV2 sensor scene:"
echo "    LiDAR sweeps (~150 @ 10Hz):   ~150 MB"
echo "    Camera images (7 cams×150f):  ~300-800 MB"
echo "    Annotations + poses:          ~5 MB"
echo "    Total per scene:              ~500 MB – 1 GB"
echo "============================================================"
