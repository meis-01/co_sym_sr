# SLURM cluster runs

The solver remains a serial DOLFIN/PETSc/MUMPS application. Each SLURM job requests
one task and defaults to one CPU. The solver checks that it is running with one
MPI rank, and the cluster runner limits common native math/thread runtimes to
one thread, including after site modules load. Keep `SLURM_CPUS=1` for this
serial implementation. Increasing array concurrency runs more independent
simulations at once; it does not make an individual simulation use multiple
CPUs.

## Checkout and environment

Clone the `exp` branch and configure the site-specific paths and resources:

```bash
git clone --branch exp --single-branch https://github.com/meis-01/co_sym_sr.git
cd co_sym_sr
cp scripts/cluster.env.example scripts/cluster.env
${EDITOR:-vi} scripts/cluster.env
```

Set `PROJECT_ROOT`, `RUN_ROOT`, `LOG_ROOT`, and `MANIFEST_ROOT` to paths visible
from both login and compute nodes. Prefer fast shared storage for the checkout
and scratch storage with enough capacity for outputs. Set the SLURM CPU, memory,
time, partition, account, and array-concurrency values for the site. The
`scripts/cluster.env` file is ignored by Git and contains local settings.
`SLURM_CPUS` is checked centrally and must be an integer from 1 through 8;
larger values are rejected before either submitter reaches `sbatch`. The batch
file itself defaults to one task and one CPU for direct `sbatch` use. The batch
runner refuses to start CFD if an explicitly overridden allocation has more
than 8 `SLURM_CPUS_PER_TASK` or more than one task; use the helper scripts to
ensure the reservation is checked before SLURM accepts the job.

The native-module mode is the first choice when the cluster supplies a
compatible stack. Put the exact module names in `CLUSTER_MODULES`, including one
consistent compiler/MPI/PETSc/FEniCS stack, and provide Python 3.12 plus Gmsh,
MUMPS, and the Python bindings for DOLFIN, PETSc, NumPy, SciPy, h5py, and
meshio. The Python ABI and native libraries must agree. The existing lockfile
pins Python package versions; `uv sync` installs the lockfile's pure-Python
packages into `.venv` while exposing the module-provided compiled packages via
system site packages. It deliberately does not replace NumPy, SciPy, h5py, or
Matplotlib with wheels. The setup and each job check key imports, but a site
stack still needs the smoke simulation below.

If the site does not provide uv as a module, install it in your user account on
a node with network access and make its bin directory available on login and
compute nodes:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

The upstream [uv installer](https://docs.astral.sh/uv/reference/installer/)
documents alternate install paths for shared cluster software locations.

```bash
bash scripts/cluster-setup.sh
```

If no compatible native stack is available, use Apptainer where the site allows
image builds. The included definition isolates the known Ubuntu 24.04 legacy
FEniCS/PETSc package stack inside the image; the host operating system need not
be Ubuntu. This is an optional image recipe, not a claim that the host's native
packages are portable across distributions.

```bash
apptainer build "$SCRATCH/cardiac-flow.sif" containers/cardiac-flow.def
# In scripts/cluster.env:
CLUSTER_EXECUTION_MODE=apptainer
APPTAINER_IMAGE="$SCRATCH/cardiac-flow.sif"
bash scripts/cluster-setup.sh
```

The cluster must provide Apptainer and access to the image, project, run, and
log paths. Native-module mode instead requires a usable `uv` command on login
and compute nodes. No Conda or Poetry environment is used.

`environment-ubuntu24.04.lock.txt` records the project's existing WSL apt
baseline; it is not a requirement for the cluster host. The Apptainer recipe
uses that known Ubuntu package family inside the container when native modules
are unavailable.

## One simulation

Submit a single config and choose its output directory explicitly:

```bash
bash scripts/slurm/submit-one.sh configs/cluster-smoke.json "$RUN_ROOT/smoke/first"
```

The submitter creates a one-case manifest, validates the config, and sends one
task to SLURM. It stores scheduler stdout and stderr separately in `LOG_ROOT`.
For a safe configuration/submission preview, set `SLURM_DRY_RUN=1`; no job is
submitted:

```bash
SLURM_DRY_RUN=1 bash scripts/slurm/submit-one.sh configs/cluster-smoke.json "$RUN_ROOT/smoke/first"
```

## Parameter sweeps

The array generator reuses the cases from `scripts/study.py`. It creates a
manifest with stable zero-based array indices, config copies, parameter hashes,
and per-case output paths:

```bash
SLURM_DRY_RUN=1 bash scripts/slurm/submit-array.sh sweep
bash scripts/slurm/submit-array.sh sweep
```

The array range is `0..case_count-1`; its maximum simultaneous jobs comes from
`SLURM_ARRAY_CONCURRENCY`. Logs are named with the array and task IDs. The
available studies are `sweep`, `convergence`, and `rotation`. The generated
manifest is kept under `MANIFEST_ROOT`, and `RUN_ROOT/<study>/<case-id>` holds
each simulation independently.

Before submitting a full array, verify one generated case without running CFD:

```bash
uv run --no-sync python scripts/experiments.py case \
  --manifest "$MANIFEST_ROOT/sweep.json" --index 0 --dry-run
```

## Monitoring, retries, and results

Use the usual SLURM tools:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
uv run --no-sync python scripts/experiments.py status --manifest "$MANIFEST_ROOT/sweep.json"
```

Successful outputs are skipped on resubmission only after `fields.h5` has a
passed `validation.json`. Failed or interrupted output directories are moved
into `RUN_ROOT/.attempts/<case-id>/<timestamp>` before retry, preserving their
diagnostics and failure report. Resubmit the same array command to retry those
cases; validated successful results are never overwritten, and a successful
directory made from a different config is rejected. Each output contains the
solver's `config.json`, diagnostics, report, unchanged HDF5 dataset, validation
report, and `execution.json` with Git commit/dirty state, config and manifest
hashes, parameters, Python/native package versions, and execution mode.

Validation runs automatically immediately after each successful solve. It can
also be run again explicitly:

```bash
uv run --no-sync python -m cardiac_flow validate "$RUN_ROOT/sweep/case_000/fields.h5"
```

Copy results and logs back to another machine with, for example:

```bash
rsync -av cluster-user@cluster:/scratch/USER/cardiac-flow-runs/sweep/ ./results/sweep/
rsync -av cluster-user@cluster:/scratch/USER/cardiac-flow-logs/ ./logs/
```

## Smoke test

`configs/cluster-smoke.json` is a deliberately coarse, short run. First verify
the environment and submission command, then submit it as a normal SLURM job:

```bash
SLURM_DRY_RUN=1 bash scripts/slurm/submit-one.sh configs/cluster-smoke.json "$RUN_ROOT/smoke/cluster-smoke"
bash scripts/slurm/submit-one.sh configs/cluster-smoke.json "$RUN_ROOT/smoke/cluster-smoke"
```

Check the job log and `validation.json` before using the site for larger runs.
This smoke test checks basic execution and output integrity; it does not measure
parallel scaling or establish production numerical accuracy.
