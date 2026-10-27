#!/usr/bin/env bash
#SBATCH --job-name=sr-stage
#SBATCH --ntasks=1
set -euo pipefail
source "${CFD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/cluster-common.sh"
cluster_prepare
"$CLUSTER_PYTHON" "$PROJECT_ROOT/scripts/sr_workflow.py" stage --state "${1:?workflow.json required}" "${2:?stage required}"
