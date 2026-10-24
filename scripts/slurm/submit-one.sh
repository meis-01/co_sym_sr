#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../cluster-common.sh"
config="${1:?usage: submit-one.sh CONFIG.json OUTPUT_DIRECTORY}"
output="${2:?usage: submit-one.sh CONFIG.json OUTPUT_DIRECTORY}"
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 2; }
cluster_prepare
mkdir -p "$MANIFEST_ROOT" "$(dirname "$output")"
config="$(realpath "$config")"
output="$(realpath -m "$output")"
config_hash="$(sha256sum "$config" | cut -c1-12)"
output_hash="$(printf '%s' "$output" | sha256sum | cut -c1-12)"
manifest="$MANIFEST_ROOT/single-$(basename "${config%.json}")-$config_hash-$output_hash.json"
qualification=()
case_study="$(cluster_python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("_sr", {}).get("study", ""))' "$config")"
if [[ "$case_study" == sr_sweep ]]; then
  if [[ -n "${SR_QUALIFICATION:-}" ]]; then
    qualification=(--qualification "$SR_QUALIFICATION")
  elif [[ "${SLURM_DRY_RUN:-0}" != 1 ]]; then
    echo "Set SR_QUALIFICATION for sr_sweep configurations" >&2; exit 2
  fi
fi
cluster_python "$PROJECT_ROOT/scripts/experiments.py" manifest --config "$config" \
  --manifest "$manifest" --output-root "$(dirname "$output")" --single-output "$output" "${qualification[@]}"
cluster_sbatch_options
mkdir -p "$LOG_ROOT"
submit=(sbatch "${SBATCH_OPTIONS[@]}" --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err"
  "$(dirname "$0")/run-case.sbatch" "$manifest" 0)
if [[ "${SLURM_DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY RUN:'; printf ' %q' "${submit[@]}"; printf '\n'
else
  "${submit[@]}"
fi
