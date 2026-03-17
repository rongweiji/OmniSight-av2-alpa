#!/usr/bin/env bash
# Quick dev mode — hot-reload on both API and frontend.
# Run two terminals, or use tmux split.
#
# Terminal 1 (API):
#   bash scripts/start_dev.sh api
#
# Terminal 2 (Frontend):
#   bash scripts/start_dev.sh frontend
#
# Or both in one tmux split:
#   bash scripts/start_dev.sh all

set -euo pipefail

DATA_DIR="${AV2_DATA_DIR:-/raid/av2/sensor/val}"
API_PORT="${API_PORT:-8080}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
DGX_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

MODE="${1:-all}"

_activate_conda() {
  for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
      source "${candidate}/etc/profile.d/conda.sh"
      conda activate omnisight 2>/dev/null || true
      return
    fi
  done
}

run_api() {
  echo "Starting API on :${API_PORT} → data: ${DATA_DIR}"
  _activate_conda
  AV2_DATA_DIR="${DATA_DIR}" python -m api.server --data-dir "${DATA_DIR}" --port "${API_PORT}"
}

run_frontend() {
  echo "Starting Next.js dev on :${FRONTEND_PORT}"
  cd "$(dirname "$0")/../frontend"
  export API_URL="http://127.0.0.1:${API_PORT}"
  export NEXT_PUBLIC_API_URL="${API_URL}"
  npm run dev -- --port "${FRONTEND_PORT}"
}

case "$MODE" in
  api)      run_api ;;
  frontend) run_frontend ;;
  all)
    # Requires tmux
    tmux new-session -d -s omnisight-dev-api "bash $0 api"
    tmux split-window -h "bash $0 frontend"
    tmux attach -t omnisight-dev-api
    ;;
  *)
    echo "Usage: $0 [api|frontend|all]"
    exit 1
    ;;
esac
