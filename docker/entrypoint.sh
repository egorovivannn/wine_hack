#!/usr/bin/env bash
# Container start: on the first run prepare the catalogue, download the pinned models and build
# the index into the mounted ./data, then serve the API. Later starts go straight to the server.
set -euo pipefail
cd /app

step() { echo "==> $*"; uv run --locked --no-sync python -m "$@"; }

if [ ! -f data/.setup-complete ] || [ "${WINE_SETUP:-auto}" = "force" ]; then
  if [ ! -f "data/source/Датасет.zip" ]; then
    echo "Put the organizers' archive at ./data/source/Датасет.zip and mount ./data to /app/data." >&2
    exit 1
  fi
  echo "First start: preparing data, models and index in ./data (about 20 minutes)."
  step scanner.prepare
  step scanner.reference_fixes
  step scanner.model_fetch
  step scanner.verifier download
  step scanner.vision build --batch-size 16
  step scanner.verifier thumbnails
  touch data/.setup-complete
fi

exec uv run --locked --no-sync uvicorn scanner.server:app --host 0.0.0.0 --port 8088 --workers 1
