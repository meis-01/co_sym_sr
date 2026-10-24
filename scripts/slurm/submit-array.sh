#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../cluster-common.sh"
study="${1:?usage: submit-array.sh sweep|convergence|rotation}"
case "$study" in sweep|convergence|rotation) ;; *) echo "Unknown study: $study" >&2; exit 2 ;; esac
cluster_prepare
manifest="$MANIFEST_ROOT/$study.json"
output_root="$RUN_ROOT/$study"
cluster_python "$PROJECT_ROOT/scripts/experiments.py" manifest --study "$study" \
  --manifest "$manifest" --output-root "$output_root"
count="$(cluster_python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["cases"]))' "$manifest")"
((count > 0)) || { echo "Manifest contains no cases" >&2; exit 2; }
cluster_sbatch_options
mkdir -p "$LOG_ROOT"
array="0-$((count-1))%$SLURM_ARRAY_CONCURRENCY"
submit=(sbatch "${SBATCH_OPTIONS[@]}" --array="$array" --output="$LOG_ROOT/%x-%A_%a.out"
  --error="$LOG_ROOT/%x-%A_%a.err" "$(dirname "$0")/run-case.sbatch" "$manifest")
if [[ "${SLURM_DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY RUN:'; printf ' %q' "${submit[@]}"; printf '\n'
else
  "${submit[@]}"
fi
