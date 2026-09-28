#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export GRAPHIFY_MAX_WORKERS="${GRAPHIFY_MAX_WORKERS:-2}"
graphify update .
# Refresh clustering sidecars before wiki export; update can leave them stale.
graphify cluster-only . --no-label
graphify export wiki
graphify tree --label JBDtection
graphify export callflow-html --lang en
