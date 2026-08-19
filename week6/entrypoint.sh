#!/usr/bin/env bash
# Week-6 single-container entrypoint (DEPLOYMENT_MODE=single).
#
#   1. Frozen-artifact SHA-256 integrity check — REFUSE TO SERVE on any
#      missing/mismatched artifact. Never regenerate artifacts to pass this.
#   2. Start FastAPI (week6.orchestrator:app, DEPLOYMENT_MODE=single) on
#      $API_PORT. Tier-1 scoring runs IN-PROCESS with the frozen models —
#      no auxiliary agent processes, no ports 8001-8004, so nothing external
#      can die and break the dashboard.
#   3. Start the Streamlit dashboard (week6/dashboard/app.py) on
#      $DASHBOARD_PORT.
#
# The six-service microservice architecture (docker-compose.yml) overrides
# this entrypoint per service with DEPLOYMENT_MODE=microservice on the
# orchestrator-api.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
export DEPLOYMENT_MODE=single

echo "[week6] frozen-artifact integrity verification ..."
python -m week6.artifact_integrity
echo "[week6] integrity OK — starting services (DEPLOYMENT_MODE=single)"

python -m uvicorn week6.orchestrator:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

# Give the API time to load the Tier-1 models + frozen data before the
# dashboard comes up (single mode loads everything eagerly at startup).
for _ in $(seq 1 90); do
    if python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:${API_PORT}/health', timeout=5).status_code==200 else 1)" 2>/dev/null; then
        break
    fi
    sleep 2
done

python -m streamlit run week6/dashboard/app.py \
    --server.port "${DASHBOARD_PORT}" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false &
DASH_PID=$!

trap 'kill "${API_PID}" "${DASH_PID}" 2>/dev/null || true' TERM INT

# Lifecycle guard: if either process dies, stop the other one too — the
# container then exits and Docker's restart policy brings the whole,
# consistent stack back instead of serving a half-dead demo.
while kill -0 "${API_PID}" 2>/dev/null && kill -0 "${DASH_PID}" 2>/dev/null; do
    sleep 5 &
    wait $!
done
echo "[week6] a service exited — shutting the container down"
kill "${API_PID}" "${DASH_PID}" 2>/dev/null || true
wait || true
exit 1
