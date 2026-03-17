#!/usr/bin/env bash
# =============================================================================
# start.sh — Start OmniSight (API + frontend) with one command
#
# Usage:
#   bash start.sh
#   bash start.sh --data-dir /raid/av2/sensor/val
#   bash start.sh --stop      # stop both services
# =============================================================================

set -euo pipefail

DATA_DIR="${AV2_DATA_DIR:-/raid/av2/sensor/val}"
API_PORT=8080
FRONTEND_PORT=3000
ENV_NAME="omnisight"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
API_PID_FILE="/tmp/omnisight-api.pid"
FRONTEND_PID_FILE="/tmp/omnisight-frontend.pid"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)      DATA_DIR="$2";      shift 2 ;;
    --api-port)      API_PORT="$2";      shift 2 ;;
    --frontend-port) FRONTEND_PORT="$2"; shift 2 ;;
    --stop)
      echo "Stopping OmniSight..."
      [ -f "$API_PID_FILE" ]      && kill "$(cat $API_PID_FILE)"      2>/dev/null && echo "  API stopped"      || true
      [ -f "$FRONTEND_PID_FILE" ] && kill "$(cat $FRONTEND_PID_FILE)" 2>/dev/null && echo "  Frontend stopped" || true
      rm -f "$API_PID_FILE" "$FRONTEND_PID_FILE"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Find conda ────────────────────────────────────────────────────────────────
CONDA_BASE=""
for c in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
  [ -f "${c}/etc/profile.d/conda.sh" ] && CONDA_BASE="$c" && break
done

if [ -z "$CONDA_BASE" ]; then
  echo "[ERROR] conda not found. Activate the omnisight env manually and re-run."
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

DGX_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

# ── Stop any existing instances ───────────────────────────────────────────────
[ -f "$API_PID_FILE" ]      && kill "$(cat $API_PID_FILE)"      2>/dev/null || true
[ -f "$FRONTEND_PID_FILE" ] && kill "$(cat $FRONTEND_PID_FILE)" 2>/dev/null || true
sleep 1

echo "============================================================"
echo "  OmniSight — starting up"
echo "  Data dir : ${DATA_DIR}"
echo "  API      : http://${DGX_IP}:${API_PORT}"
echo "  Viewer   : http://${DGX_IP}:${FRONTEND_PORT}"
echo "============================================================"

# ── Step 1: Start API in background ──────────────────────────────────────────
echo ""
echo "[1/3] Starting data API..."

cd "${REPO_DIR}"
# Use the full conda env Python path — background processes can lose PATH
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
[ ! -f "$PYTHON" ] && PYTHON="$(which python)"  # fallback

AV2_DATA_DIR="${DATA_DIR}" "${PYTHON}" -m api.server \
  --data-dir "${DATA_DIR}" \
  --port "${API_PORT}" \
  > ~/omnisight-api.log 2>&1 &
echo $! > "$API_PID_FILE"
echo "  -> API running (PID $(cat $API_PID_FILE))  log: ~/omnisight-api.log"

# Wait until API is ready (up to 15 s)
echo "  -> Waiting for API to be ready..."
for i in $(seq 1 15); do
  curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1 && break
  sleep 1
done

if ! curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
  echo "  [ERROR] API did not start. Check ~/omnisight-api.log"
  exit 1
fi
echo "  -> API ready."

# ── Step 2: Build frontend (only if needed) ───────────────────────────────────
echo ""
echo "[2/3] Building frontend..."

cd "${REPO_DIR}/frontend"

# Write .env.local so Next.js always has the API URL — more reliable than shell env prefix
echo "API_URL=http://localhost:${API_PORT}" > .env.local

if [ ! -d "node_modules" ]; then
  echo "  -> Installing Node.js dependencies..."
  npm install
fi

npm run build
echo "  -> Frontend built."

# ── Step 3: Start frontend in background ─────────────────────────────────────
echo ""
echo "[3/3] Starting frontend..."

npm start -- --port "${FRONTEND_PORT}" \
  > ~/omnisight-frontend.log 2>&1 &
echo $! > "$FRONTEND_PID_FILE"
echo "  -> Frontend running (PID $(cat $FRONTEND_PID_FILE))  log: ~/omnisight-frontend.log"

# Wait until frontend responds
for i in $(seq 1 20); do
  curl -sf "http://localhost:${FRONTEND_PORT}" > /dev/null 2>&1 && break
  sleep 1
done

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  OmniSight is running!"
echo ""
echo "  Open:  http://${DGX_IP}:${FRONTEND_PORT}"
echo ""
echo "  Logs:"
echo "    tail -f ~/omnisight-api.log"
echo "    tail -f ~/omnisight-frontend.log"
echo ""
echo "  Stop:  bash start.sh --stop"
echo "============================================================"
