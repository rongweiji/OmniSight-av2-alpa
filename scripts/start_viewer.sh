#!/usr/bin/env bash
# =============================================================================
# start_viewer.sh — Start the OmniSight data API + Next.js frontend on DGX
# =============================================================================
#
# Usage:
#   bash scripts/start_viewer.sh
#   bash scripts/start_viewer.sh --data-dir /raid/av2/sensor/val
#   bash scripts/start_viewer.sh --api-port 8080 --frontend-port 3000
#
# After running:
#   Frontend:  http://<dgx-ip>:3000
#   API:       http://<dgx-ip>:8080
#   API docs:  http://<dgx-ip>:8080/docs
# =============================================================================

set -euo pipefail

DATA_DIR="/raid/av2/sensor/val"
API_PORT=8080
FRONTEND_PORT=3000
ENV_NAME="omnisight"
TMUX_API="omnisight-api"
TMUX_FRONTEND="omnisight-frontend"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    --api-port)       API_PORT="$2"; shift 2 ;;
    --frontend-port)  FRONTEND_PORT="$2"; shift 2 ;;
    --env)            ENV_NAME="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  OmniSight Viewer Startup"
echo "============================================================"
echo "  Data dir      : ${DATA_DIR}"
echo "  API port      : ${API_PORT}"
echo "  Frontend port : ${FRONTEND_PORT}"
echo "============================================================"

# Activate conda — find miniconda/anaconda in common locations
CONDA_BASE=""
for candidate in \
  "$HOME/miniconda3" \
  "$HOME/anaconda3" \
  "/opt/miniconda3" \
  "/opt/anaconda3" \
  "/usr/local/miniconda3" \
  "/usr/local/anaconda3"; do
  if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="${candidate}"
    break
  fi
done

if [ -z "${CONDA_BASE}" ]; then
  echo "[ERROR] Could not find conda installation."
  echo "  Tried: ~/miniconda3, ~/anaconda3, /opt/miniconda3, /opt/anaconda3"
  echo "  Set CONDA_BASE manually or run the API and frontend directly:"
  echo "    python -m api.server --data-dir ${DATA_DIR} --port ${API_PORT}"
  echo "    cd frontend && npm start -- --port ${FRONTEND_PORT}"
  exit 1
fi

echo "  -> Conda base: ${CONDA_BASE}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Kill existing sessions
tmux kill-session -t "${TMUX_API}" 2>/dev/null || true
tmux kill-session -t "${TMUX_FRONTEND}" 2>/dev/null || true

# ── Start Python API ──────────────────────────────────────────────────────────
echo ""
echo "[1/2] Starting Data API (port ${API_PORT})..."

tmux new-session -d -s "${TMUX_API}" \
  "conda activate ${ENV_NAME} && \
   AV2_DATA_DIR=${DATA_DIR} python -m api.server \
     --data-dir '${DATA_DIR}' \
     --port ${API_PORT} \
   2>&1 | tee ~/omnisight-api.log"

echo "  -> API starting in tmux: ${TMUX_API}"

# ── Build + start Next.js frontend ────────────────────────────────────────────
echo ""
echo "[2/2] Starting Next.js frontend (port ${FRONTEND_PORT})..."

cd frontend

# Install Node deps if node_modules is missing
if [ ! -d "node_modules" ]; then
  echo "  -> Installing Node.js dependencies..."
  npm install
fi

# Build for production
# API_URL is server-side only — Next.js proxy rewrites use it, browser never sees it
echo "  -> Building Next.js app..."
API_URL="http://localhost:${API_PORT}" npm run build

cd ..

tmux new-session -d -s "${TMUX_FRONTEND}" \
  "cd $(pwd)/frontend && \
   API_URL=http://localhost:${API_PORT} \
   npm start -- --port ${FRONTEND_PORT} \
   2>&1 | tee ~/omnisight-frontend.log"

echo "  -> Frontend starting in tmux: ${TMUX_FRONTEND}"

# ── Done ──────────────────────────────────────────────────────────────────────
DGX_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "============================================================"
echo "  Viewer ready!"
echo ""
echo "  Open in browser:"
echo "    http://${DGX_IP}:${FRONTEND_PORT}"
echo ""
echo "  Monitor:"
echo "    tmux attach -t ${TMUX_API}       # API server"
echo "    tmux attach -t ${TMUX_FRONTEND}  # Frontend"
echo ""
echo "  Logs:"
echo "    tail -f ~/omnisight-api.log"
echo "    tail -f ~/omnisight-frontend.log"
echo "============================================================"
