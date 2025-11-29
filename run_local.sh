#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON:-python3}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"

echo "Запускаю backend на порту ${BACKEND_PORT}..."
"${PYTHON_BIN}" -m uvicorn backend.app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" --reload &
BACK_PID=$!
echo "Backend PID: ${BACK_PID}"

echo "Запускаю frontend (статический сервер) на порту ${FRONTEND_PORT}..."
"${PYTHON_BIN}" -m http.server "${FRONTEND_PORT}" --directory "${ROOT_DIR}/frontend" &
FRONT_PID=$!
echo "Frontend PID: ${FRONT_PID}"

trap 'echo "Останавливаю..."; kill "${BACK_PID}" "${FRONT_PID}"' INT TERM
wait
