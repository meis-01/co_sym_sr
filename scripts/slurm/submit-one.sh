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
manifest="$MANIFEST_ROOT/single-$(basename "${config%.json}")-$config_hash.json"
cluster_python "$PROJECT_ROOT/scripts/experiments.py" manifest --config "$config" \
  --manifest "$manifest" --output-root "$(dirname "$output")"
# Single-case output path is explicit, so update the manifest output before submission.
cluster_python - "$manifest" "$output" <<'PY'
import json, sys
path, output = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
data["cases"][0]["output"] = output
data["output_root"] = __import__("os").path.dirname(output)
open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
cluster_sbatch_options
mkdir -p "$LOG_ROOT"
submit=(sbatch "${SBATCH_OPTIONS[@]}" --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err"
  "$(dirname "$0")/run-case.sbatch" "$manifest" 0)
if [[ "${SLURM_DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY RUN:'; printf ' %q' "${submit[@]}"; printf '\n'
else
  "${submit[@]}"
fi
