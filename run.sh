#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cd "$ROOT_DIR"

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

echo "Starting FastAPI backend on http://localhost:${BACKEND_PORT}"
uvicorn backend.app:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

echo "Starting React frontend on http://localhost:${FRONTEND_PORT}"
(cd frontend && VITE_API_BASE_URL="http://localhost:${BACKEND_PORT}" npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT") &
FRONTEND_PID=$!

cleanup() {
  echo "Stopping services..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT
wait
