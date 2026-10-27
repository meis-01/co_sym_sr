#!/usr/bin/env bash
#SBATCH --job-name=cardiac-cfd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=cardiac-cfd-%j.out
#SBATCH --error=cardiac-cfd-%j.err
set -euo pipefail
source "${CFD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/cluster-common.sh"
cluster_prepare
manifest="${1:?usage: run-case.sbatch MANIFEST.json CASE_INDEX}"
index="${2:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "$index" ]] || { echo "A case index is required" >&2; exit 2; }
if [[ -n "${3:-}" ]]; then
  "$CLUSTER_PYTHON" "$PROJECT_ROOT/scripts/sr_workflow.py" verify --state "$3"
fi
cluster_python "$PROJECT_ROOT/scripts/experiments.py" case --manifest "$manifest" --index "$index"
