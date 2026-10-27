#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../cluster-common.sh"
action="${1:?usage: submit-sr.sh submit|status NEW_CAMPAIGN_ROOT [--dry-run]}"
shift
case "$action" in submit|status) ;; *) echo "Unknown action: $action" >&2; exit 2 ;; esac
if [[ "$action" == status ]]; then
  # Status remains usable when the native CFD runtime/container is unavailable.
  "$CLUSTER_PYTHON" "$PROJECT_ROOT/scripts/sr_workflow.py" status "$@"
  exit $?
fi
if [[ "$action" == submit ]]; then
  campaign_root="$(realpath -m "${1:?new campaign root required}")"
  run_root="$(realpath -m "$RUN_ROOT")"
  case "$campaign_root" in "$run_root"/*) ;; *)
    echo "Campaign root must be inside RUN_ROOT=$run_root (shared/container-visible storage)." >&2
    exit 2 ;;
  esac
fi
cluster_prepare
cluster_sbatch_options # export project/config/environment for spool copies
options=()
if [[ "$action" == submit ]]; then
  options=(--cpus "$SLURM_CPUS" --memory "$SLURM_MEMORY" --time "$SLURM_TIME"
    --partition "$SLURM_PARTITION" --account "$SLURM_ACCOUNT" --concurrency "$SLURM_ARRAY_CONCURRENCY")
  [[ "${SLURM_DRY_RUN:-0}" != 1 ]] || options+=(--dry-run)
fi
if [[ "$action" == submit ]]; then
  case "$CLUSTER_EXECUTION_MODE" in
    modules) science=(uv run --no-sync python) ;;
    micromamba) mapfile -d '' -t science < <(cluster_micromamba_args) ;;
    apptainer) mapfile -d '' -t science < <(cluster_apptainer_args) ;;
  esac
  science_json="$(printf '%s\0' "${science[@]}" | "$CLUSTER_PYTHON" -c 'import json,sys; print(json.dumps(sys.stdin.read().split("\0")[:-1]))')"
  options+=(--python-command-json "$science_json")
fi
# sbatch/sacct belong to the host; scientific subprocesses use the selected runtime.
"$CLUSTER_PYTHON" "$PROJECT_ROOT/scripts/sr_workflow.py" "$action" "$@" "${options[@]}"
