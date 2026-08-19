# Week-6 deployment (shared image for all services).
#
# ONE image (`insider-threat-w6`) is reused by every compose service:
#   http-agent / file-device-agent / logon-agent / email-agent (Tier-1 MLPs),
#   orchestrator-api (frozen Tier-2 + public API) and dashboard (Streamlit).
# Each service overrides the ENTRYPOINT in docker-compose.yml
# (DEPLOYMENT_MODE=microservice on the orchestrator). The default
# entrypoint.sh below runs the RELIABLE SINGLE-CONTAINER demo:
# FastAPI orchestrator with DEPLOYMENT_MODE=single (in-process Tier-1
# scoring — no auxiliary agent processes, no ports 8001-8004) + Streamlit.
#
# The image contains CODE + dependencies only. Frozen research artifacts are
# mounted READ-ONLY at their original relative paths (see docker-compose.yml)
# and SHA-256-verified by week6/artifact_integrity.py before the services
# start. On mismatch the container refuses to serve.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    API_PORT=8000 \
    DASHBOARD_PORT=8501 \
    SHOW_GROUND_TRUTH=false

# Dependency install (exact locked set + the week6 group = streamlit, the only
# NEW Week-6 dependency). Project itself is not installed (--no-install-project).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-default-groups --group week6

# Research/deployment code (frozen artifacts excluded via .dockerignore).
COPY week4/ week4/
COPY week5/ week5/
COPY week6/ week6/

# Windows checkouts may bring CRLF line endings — normalize for bash.
RUN sed -i 's/\r$//' /app/week6/entrypoint.sh \
    && chmod +x /app/week6/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501

ENTRYPOINT ["/app/week6/entrypoint.sh"]
