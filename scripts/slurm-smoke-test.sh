#!/usr/bin/env bash
# Small end-to-end check: one coarse CFD case plus automatic HDF5 validation.
set -euo pipefail
source "$(dirname "$0")/cluster-common.sh"
cluster_prepare
smoke_root="$RUN_ROOT/smoke"
manifest="$MANIFEST_ROOT/smoke.json"
cluster_python "$PROJECT_ROOT/scripts/experiments.py" manifest --config "$PROJECT_ROOT/configs/cluster-smoke.json" \
  --manifest "$manifest" --output-root "$smoke_root"
if [[ "${1:-}" == --dry-run ]]; then
  cluster_python "$PROJECT_ROOT/scripts/experiments.py" case --manifest "$manifest" --index 0 --dry-run
else
  cluster_python "$PROJECT_ROOT/scripts/experiments.py" case --manifest "$manifest" --index 0
  cluster_python "$PROJECT_ROOT/scripts/experiments.py" status --manifest "$manifest"
fi
