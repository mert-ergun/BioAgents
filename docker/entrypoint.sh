#!/usr/bin/env bash
# BioAgents container entrypoint.
#
# Responsibilities:
#   * Ensure runtime directories exist and are writable (volume mounts can
#     override ownership set at build time).
#   * Print a one-line runtime summary that helps users spot config issues.
#   * Dispatch to one of the supported commands:
#       serve     — run the FastAPI web UI (default)
#       worker    — same image, used by docker-compose for the sandbox worker
#       shell     — drop into a bash prompt with the venv active
#       test      — run the pytest suite
#       exec …    — run an arbitrary command inside the venv

set -euo pipefail

log() { printf '[bioagents] %s\n' "$*"; }

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-1}"

ensure_dirs() {
    for d in generated_artifacts uploads experiment_runs sandbox_workdir logs .bioagents_cache; do
        mkdir -p "/app/${d}" 2>/dev/null || true
    done
}

print_banner() {
    log "BioAgents container starting"
    log "  python  : $(python --version 2>&1)"
    log "  node    : $(node --version 2>&1 || echo 'not installed')"
    log "  rdkit-agent: $(rdkit-agent --version 2>&1 || echo 'not installed')"
    log "  LLM_PROVIDER=${LLM_PROVIDER:-<unset>}  USE_LOCAL_EXECUTOR=${USE_LOCAL_EXECUTOR:-false}"
    if [ -S /var/run/docker.sock ]; then
        log "  docker socket detected — nested DockerExecutor is available"
    fi
}

ensure_dirs
print_banner

cmd="${1:-serve}"
shift || true

case "$cmd" in
    serve)
        log "launching uvicorn on ${HOST}:${PORT} (workers=${WORKERS})"
        exec uvicorn frontend.server:app \
            --host "${HOST}" \
            --port "${PORT}" \
            --workers "${WORKERS}" \
            --proxy-headers \
            --forwarded-allow-ips='*'
        ;;
    worker)
        log "worker mode — idling so agents can exec() into this container"
        exec sleep infinity
        ;;
    shell|bash)
        exec bash "$@"
        ;;
    test)
        exec pytest "$@"
        ;;
    exec)
        exec "$@"
        ;;
    *)
        exec "$cmd" "$@"
        ;;
esac
