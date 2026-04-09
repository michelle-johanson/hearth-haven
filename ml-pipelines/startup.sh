#!/bin/bash
#
# Azure App Service startup command for the Hearth Haven ML inference server.
#
# This script is designed to be committed to the repo as a single source of
# truth for how the FastAPI server is launched in production. Point Azure at
# it via:
#
#   az webapp config set \
#     --resource-group <rg> \
#     --name hearth-haven-ml \
#     --startup-file "bash /home/site/wwwroot/startup.sh"
#
# It assumes the deploy root (wwwroot) contains this script alongside the
# `inference/` and `models/` directories from ml-pipelines/.
#
# Workers:
#   - 2 workers is a safe default for the smallest App Service tiers (B1/S1).
#     Bump if you move to a larger plan and see request queueing.
# Timeout:
#   - 120s accommodates cold-start sklearn model loads on the first request
#     after a restart.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

exec gunicorn \
  --chdir "$SCRIPT_DIR/inference" \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT:-8000}" \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  server:app
