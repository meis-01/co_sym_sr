#!/usr/bin/env bash
set -euo pipefail

CLUSTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CLUSTER_SCRIPT_DIR/.." && pwd)"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-$CLUSTER_SCRIPT_DIR/cluster.env}"
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
  echo "Missing $CLUSTER_CONFIG; copy scripts/cluster.env.example and configure the cluster." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$CLUSTER_CONFIG"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$CLUSTER_SCRIPT_DIR/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/cluster-runs}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/cluster-logs}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RUN_ROOT/manifests}"
SLURM_CPUS="${SLURM_CPUS:-1}"
if [[ ! "$SLURM_CPUS" =~ ^[1-8]$ ]]; then
  echo "SLURM_CPUS must be an integer from 1 to 8 (requested: $SLURM_CPUS)." >&2
  exit 2
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && ! "$SLURM_CPUS_PER_TASK" =~ ^[1-8]$ ]]; then
  echo "Refusing job allocation with SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK; maximum is 8." >&2
  exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" && "$SLURM_NTASKS" != 1 ]]; then
  echo "Refusing SLURM_NTASKS=$SLURM_NTASKS; CFD jobs must run as one process." >&2
  exit 2
fi
SLURM_MEMORY="${SLURM_MEMORY:-8G}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_ARRAY_CONCURRENCY="${SLURM_ARRAY_CONCURRENCY:-4}"
CLUSTER_EXECUTION_MODE="${CLUSTER_EXECUTION_MODE:-modules}"
CLUSTER_PYTHON="${CLUSTER_PYTHON:-python3}"
declare -p CLUSTER_MODULES >/dev/null 2>&1 || CLUSTER_MODULES=()

cluster_load_modules() {
  if ((${#CLUSTER_MODULES[@]})); then
    command -v module >/dev/null || { echo "Environment Modules is unavailable" >&2; exit 2; }
    module load "${CLUSTER_MODULES[@]}"
    export CLUSTER_MODULES_LOADED="${CLUSTER_MODULES[*]}"
  fi
}

cluster_apptainer_args() {
  local binds=("$PROJECT_ROOT:$PROJECT_ROOT" "$RUN_ROOT:$RUN_ROOT" "$LOG_ROOT:$LOG_ROOT")
  local bind_list
  bind_list="$(IFS=,; echo "${binds[*]}")"
  printf '%s\0' apptainer exec --bind "$bind_list" --pwd "$PROJECT_ROOT" "$APPTAINER_IMAGE" "$APPTAINER_PYTHON"
}

cluster_limit_threads() {
  # DOLFIN currently enforces a single MPI rank and the CFD path is serial.
  # Keep common native BLAS/OpenMP runtimes from silently oversubscribing it.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1 TBB_NUM_THREADS=1
}

cluster_prepare() {
  mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$MANIFEST_ROOT"
  export CLUSTER_EXECUTION_MODE
  case "$CLUSTER_EXECUTION_MODE" in
    modules)
      cluster_load_modules
      # Apply these after module loading in case site modules set thread counts.
      cluster_limit_threads
      command -v uv >/dev/null || { echo "uv is required on the login and compute nodes" >&2; exit 2; }
      if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        echo "Missing $PROJECT_ROOT/.venv; run scripts/cluster-setup.sh after loading modules" >&2
        exit 2
      fi
      source "$PROJECT_ROOT/.venv/bin/activate"
      uv run --no-sync python -c 'import dolfin, gmsh, h5py, meshio, numpy, petsc4py, scipy; print("CFD imports ready", dolfin.__version__)'
      ;;
    apptainer)
      cluster_limit_threads
      [[ -n "${APPTAINER_IMAGE:-}" && -f "$APPTAINER_IMAGE" ]] || {
        echo "Set APPTAINER_IMAGE to an existing .sif file" >&2; exit 2;
      }
      export APPTAINERENV_CLUSTER_EXECUTION_MODE=apptainer APPTAINERENV_CFD_IN_CONTAINER=1
      export APPTAINERENV_APPTAINER_IMAGE="$APPTAINER_IMAGE"
      export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
      export APPTAINERENV_MKL_NUM_THREADS=1 APPTAINERENV_NUMEXPR_NUM_THREADS=1
      export APPTAINERENV_BLIS_NUM_THREADS=1 APPTAINERENV_GOTO_NUM_THREADS=1
      export APPTAINERENV_VECLIB_MAXIMUM_THREADS=1 APPTAINERENV_TBB_NUM_THREADS=1
      mapfile -d '' -t check_command < <(cluster_apptainer_args)
      "${check_command[@]}" -c 'import dolfin, gmsh, h5py, meshio, numpy, petsc4py, scipy; print("CFD imports ready", dolfin.__version__)'
      ;;
    *) echo "CLUSTER_EXECUTION_MODE must be modules or apptainer" >&2; exit 2 ;;
  esac
}

cluster_python() {
  if [[ "$CLUSTER_EXECUTION_MODE" == apptainer ]]; then
    # NUL-delimited arguments avoid shell re-parsing of user paths.
    mapfile -d '' -t command < <(cluster_apptainer_args)
    "${command[@]}" "$@"
  else
    uv run --no-sync python "$@"
  fi
}

cluster_sbatch_options() {
  SBATCH_OPTIONS=(--ntasks=1 "--cpus-per-task=$SLURM_CPUS" "--mem=$SLURM_MEMORY" "--time=$SLURM_TIME")
  [[ -z "$SLURM_PARTITION" ]] || SBATCH_OPTIONS+=("--partition=$SLURM_PARTITION")
  [[ -z "$SLURM_ACCOUNT" ]] || SBATCH_OPTIONS+=("--account=$SLURM_ACCOUNT")
}
