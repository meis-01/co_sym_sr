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
CLUSTER_PROJECT_PARENT="${CLUSTER_PROJECT_PARENT:-$(cd "$PROJECT_ROOT/.." && pwd)}"
CLUSTER_EXECUTION_MODE="${CLUSTER_EXECUTION_MODE:-modules}"
if [[ "$CLUSTER_EXECUTION_MODE" == micromamba ]]; then
  RUN_ROOT="${RUN_ROOT:-$CLUSTER_PROJECT_PARENT/cluster-runs}"
  LOG_ROOT="${LOG_ROOT:-$CLUSTER_PROJECT_PARENT/cluster-logs}"
else
  RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/cluster-runs}"
  LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/cluster-logs}"
fi
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
if [[ ! "$SLURM_ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_ARRAY_CONCURRENCY must be a positive integer." >&2
  exit 2
fi
CLUSTER_PYTHON="${CLUSTER_PYTHON:-python3}"
MICROMAMBA_EXE="${MICROMAMBA_EXE:-$CLUSTER_PROJECT_PARENT/.tools/micromamba}"
MICROMAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX:-$CLUSTER_PROJECT_PARENT/.micromamba}"
MICROMAMBA_ENV_PREFIX="${MICROMAMBA_ENV_PREFIX:-$CLUSTER_PROJECT_PARENT/.cfd-env}"
MICROMAMBA_PYTHON="${MICROMAMBA_PYTHON:-python}"
declare -p CLUSTER_MODULES >/dev/null 2>&1 || CLUSTER_MODULES=()

cluster_load_modules() {
  if ((${#CLUSTER_MODULES[@]})); then
    command -v module >/dev/null || { echo "Environment Modules is unavailable" >&2; exit 2; }
    module load "${CLUSTER_MODULES[@]}"
    export CLUSTER_MODULES_LOADED="${CLUSTER_MODULES[*]}"
  fi
}

cluster_apptainer_args() {
  local binds=("$PROJECT_ROOT:$PROJECT_ROOT" "$RUN_ROOT:$RUN_ROOT" "$LOG_ROOT:$LOG_ROOT"
    "$MANIFEST_ROOT:$MANIFEST_ROOT")
  local bind_list
  bind_list="$(IFS=,; echo "${binds[*]}")"
  printf '%s\0' apptainer exec --bind "$bind_list" --pwd "$PROJECT_ROOT" "$APPTAINER_IMAGE" "$APPTAINER_PYTHON"
}

cluster_micromamba_args() {
  # Use `run` instead of activation so batch jobs do not modify shell startup
  # files or the installed environment.  NUL delimiters preserve paths safely.
  printf '%s\0' "$MICROMAMBA_EXE" run --root-prefix "$MICROMAMBA_ROOT_PREFIX" \
    --prefix "$MICROMAMBA_ENV_PREFIX" "$MICROMAMBA_PYTHON"
}

cluster_limit_threads() {
  # DOLFIN currently enforces a single MPI rank and the CFD path is serial.
  # Keep common native BLAS/OpenMP runtimes from silently oversubscribing it.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1 TBB_NUM_THREADS=1
}

cluster_verify_git() {
  # Probe Git through the selected CFD Python so a host-only Git installation
  # cannot mask a missing executable in the job environment.
  if ! cluster_python -c 'import subprocess, sys
repo = sys.argv[1]
subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL)
subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "HEAD"],
               check=True, stdout=subprocess.DEVNULL)' "$PROJECT_ROOT"; then
    echo "Git is unavailable in the selected CFD environment or cannot access: $PROJECT_ROOT" >&2
    exit 2
  fi
}

cluster_verify_imports() {
  cluster_python -c 'import cardiac_flow, dolfin, gmsh, h5py, meshio, numpy, petsc4py, scipy; print("CFD imports ready", dolfin.__version__)'
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
      export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      cluster_verify_imports
      cluster_verify_git
      ;;
    micromamba)
      cluster_limit_threads
      [[ -x "$MICROMAMBA_EXE" ]] || {
        echo "Micromamba executable is not executable: $MICROMAMBA_EXE" >&2; exit 2;
      }
      [[ -d "$MICROMAMBA_ROOT_PREFIX" ]] || {
        echo "Micromamba root prefix does not exist: $MICROMAMBA_ROOT_PREFIX" >&2; exit 2;
      }
      [[ -d "$MICROMAMBA_ENV_PREFIX" ]] || {
        echo "Micromamba environment prefix does not exist: $MICROMAMBA_ENV_PREFIX" >&2; exit 2;
      }
      export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      cluster_verify_imports
      cluster_verify_git
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
      export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      mapfile -d '' -t check_command < <(cluster_apptainer_args)
      "${check_command[@]}" -c 'import cardiac_flow, dolfin, gmsh, h5py, meshio, numpy, petsc4py, scipy; print("CFD imports ready", dolfin.__version__)'
      cluster_verify_git
      ;;
    *) echo "CLUSTER_EXECUTION_MODE must be modules, micromamba, or apptainer" >&2; exit 2 ;;
  esac
}

cluster_python() {
  if [[ "$CLUSTER_EXECUTION_MODE" == apptainer ]]; then
    # NUL-delimited arguments avoid shell re-parsing of user paths.
    mapfile -d '' -t command < <(cluster_apptainer_args)
    "${command[@]}" "$@"
  elif [[ "$CLUSTER_EXECUTION_MODE" == micromamba ]]; then
    mapfile -d '' -t command < <(cluster_micromamba_args)
    "${command[@]}" "$@"
  elif [[ "$CLUSTER_EXECUTION_MODE" == modules ]]; then
    uv run --no-sync python "$@"
  else
    echo "CLUSTER_EXECUTION_MODE must be modules, micromamba, or apptainer" >&2
    return 2
  fi
}

cluster_sbatch_options() {
  # SLURM executes a spool copy of the batch file; do not resolve code via $0.
  export CFD_PROJECT_ROOT="$PROJECT_ROOT"
  export CLUSTER_CONFIG="$(realpath "$CLUSTER_CONFIG")"
  export CLUSTER_PROJECT_PARENT MICROMAMBA_EXE MICROMAMBA_ROOT_PREFIX MICROMAMBA_ENV_PREFIX MICROMAMBA_PYTHON
  SBATCH_OPTIONS=(--ntasks=1 "--cpus-per-task=$SLURM_CPUS" "--mem=$SLURM_MEMORY" "--time=$SLURM_TIME"
    "--chdir=$PROJECT_ROOT" --export=ALL)
  [[ -z "$SLURM_PARTITION" ]] || SBATCH_OPTIONS+=("--partition=$SLURM_PARTITION")
  [[ -z "$SLURM_ACCOUNT" ]] || SBATCH_OPTIONS+=("--account=$SLURM_ACCOUNT")
}
