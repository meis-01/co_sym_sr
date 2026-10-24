#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/cluster-common.sh"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$MANIFEST_ROOT"
if [[ "$CLUSTER_EXECUTION_MODE" == modules ]]; then
  cluster_load_modules
  command -v uv >/dev/null || { echo "Install/provide uv on the login node first" >&2; exit 2; }
  uv venv --clear --python "$CLUSTER_PYTHON" --system-site-packages
  uv sync --locked \
    --no-install-package numpy --no-install-package scipy --no-install-package h5py \
    --no-install-package matplotlib
  uv run --no-sync python -c 'import dolfin, gmsh, h5py, meshio, numpy, petsc4py, scipy; print("CFD imports ready", dolfin.__version__)'
elif [[ "$CLUSTER_EXECUTION_MODE" == apptainer ]]; then
  [[ -n "${APPTAINER_IMAGE:-}" && -f "$APPTAINER_IMAGE" ]] || {
    echo "Set APPTAINER_IMAGE to an existing .sif file" >&2; exit 2;
  }
else
  echo "CLUSTER_EXECUTION_MODE must be modules or apptainer" >&2
  exit 2
fi
echo "Cluster environment setup complete ($CLUSTER_EXECUTION_MODE)."
