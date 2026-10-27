# One-session SR campaign

Use branch `exp` with the CFD environment already provisioned as described in
[cluster.md](cluster.md). Configure `scripts/cluster.env` with the site's partition,
account, scratch paths, memory and finite wall time. The login environment and
compute nodes must share this checkout and the campaign directory. SLURM must
allow `sbatch` from compute nodes: the assessment job submits the qualified sweep
and its two completion jobs. No persistent SSH connection is required.
The campaign directory must be inside the configured `RUN_ROOT`, which is bound
into Apptainer jobs. The host needs Python 3 and SLURM commands; NumPy/HDF5 and
CFD dependencies are used through the selected modules/micromamba/Apptainer runtime.

Submit the complete campaign with one command, using a new absolute directory:

```bash
bash scripts/slurm/submit-sr.sh submit "$RUN_ROOT/sr-20261027"
```

Here `$RUN_ROOT` must be set in the calling shell (or replace it with your scratch
path); configuring it only inside `cluster.env` does not expand it in the caller.
Check the whole campaign with:

```bash
bash scripts/slurm/submit-sr.sh status "$RUN_ROOT/sr-20261027"
```

Status prints JSON with the overall state, job IDs, `sacct` array/job states and
exit codes, per-study case counts, assessment result, stage failures, and final
dataset path. Failed workflows return exit code 1. If accounting is unavailable,
the output says so; absent accounting alone is never treated as completion.
File counts describe validated outputs; scheduler records distinguish pending,
running, and killed jobs, even when Python could not write a stage report.

## Dry run

Preview all five submission commands and generate/check the 32 qualification
configs and 36 draft sweep configs without submitting any jobs:

```bash
bash scripts/slurm/submit-sr.sh submit "$RUN_ROOT/sr-preview" --dry-run
```

This uses the configured environment and its import checks, but executes no CFD
and never calls `sbatch`. For a local preview requiring only the normal Python
array/HDF5 dependencies, use:

```bash
python scripts/sr_workflow.py submit studies/sr-workflow-preview --dry-run
```

The dry run validates case counts, solver settings, unique output paths, resource
syntax, and the full dependency plan. `plan.json` displays symbolic predecessor
IDs. It cannot validate site quotas, partition access, runtime, or numerical
qualification. A draft sweep has no qualification and cannot execute. Use a
different directory for the real submission; existing campaign directories are
always refused to protect queued jobs and prior evidence.

## Dependencies and numerical gates

| Job | Runs after | Action |
| --- | --- | --- |
| `sr-accuracy` | Submission | Existing `run-case.sbatch`, array 0–31 |
| `sr-assess` | `afterany` qualification array | `scripts/sr.py assess`; preserve diagnostics even if cases failed |
| `sr-sweep` | `afterok` assessment | Generate only after qualification passes; existing runner, array 0–35 |
| `sr-certify` | `afterany` sweep array | `scripts/sr.py certify` for every case, including missing/failed outputs |
| `sr-export` | `afterok` certification | Recheck certification, then `scripts/sr.py export` |

Only the first two jobs are submitted initially. A passing assessment generates
the qualified sweep manifest and submits the last three jobs. A failed/blocked
assessment submits none of them. The sweep also depends on successful completion
of the assessment controller, so downstream submission errors prevent CFD from
starting. Invalid dependencies use `--kill-on-invalid-dep=yes`.

The campaign uses the existing PETSc FieldSplit implementation and grad-div ratio
10 throughout qualification and production. General production configurations
still default to MUMPS. Mesh/time refinement and tighter-periodicity cases retain
their original settings; production uses the qualified finest profile. Counts,
physics, scientific thresholds, sampling seed 20261024, 2048 samples/phase, and
24 train / 6 validation / 6 test configuration splits are preserved. `--samples`,
`--seed`, and optional `--pressure` affect export only. Pressure retains its own
qualification gate; an unsuccessful pressure export fails the final stage.

Source scripts, solver/assessment code, JSON configurations and the cluster env
file are hash-bound at submission and checked before each array task and stage.
Keep the checkout, cluster config, and CFD environment fixed until completion.
The existing manifests/config hashes and qualification provenance provide the
remaining run checks. Commit the implementation before transferring it to HPC;
the manifests also retain Git provenance.

## Resources

All jobs inherit the same explicit allocation from `cluster.env`, including the
partition and account. Export/assessment are serial Python/HDF5 work and require
the CFD environment too. Defaults are one task, one CPU, 8 GB, 12 hours and four
concurrent array tasks. CPU requests outside 1–8, unlimited/zero memory or wall
time, and nonpositive concurrency are rejected. Native threads remain capped at
one, as in the existing runner. The CFD solver is serial; more CPUs have no
established benefit.

For example, with site-approved memory and wall time already configured:

```bash
SLURM_ARRAY_CONCURRENCY=2 bash scripts/slurm/submit-sr.sh submit "$RUN_ROOT/sr-production"
```

At concurrency C, each active CFD array requests at most C times its per-job
CPUs and memory (bounded by 32 or 36 tasks). Arrays do not overlap; short controller
completion can overlap dependent job startup. At the defaults, an active array
requests at most 4 CPUs and 32 GB. There are 68 CFD simulations plus three serial
assessment/certification/export jobs; no extra pilot simulations are added.

These defaults are requests, not measured requirements or runtime guarantees.
Choose values within local partition/QOS limits using existing pilot/benchmark
measurements and `sacct` peak RSS; do not assume the default 12-hour request is
enough for fine runs. See [sr_campaign.md](sr_campaign.md#resource-planning-and-checks)
for the existing estimates and their limits. Allow storage for all 68 outputs,
visualization files, logs, and failed attempts. The previous sweep-only estimate
was about 4 GB, excluding qualification and retries; it is not a storage bound.
The workflow does not change partitions, split allocations, or retry timed-out
simulations automatically.

## Artifacts and failures

Each campaign directory contains:

* `workflow.json`, `plan.json`, `jobs/*.json`: frozen inputs, commands, dependencies,
  and submitted IDs, including IDs retained if a later submission fails.
* `manifests/`: copied qualification configs, draft sweep preview, and the
  qualified sweep manifest/configs created only after a passing assessment.
* `sr_accuracy/`, `sr_sweep/`: unchanged executor outputs, per-attempt logs,
  validation and execution provenance; certification writes per-case `accuracy.json`.
* `qualification.json`, `reports/*.out`, `reports/*.err`, `stages/*.json`,
  `logs/`: scientific diagnostics, stage outcomes and SLURM logs.
* `sr_dataset.h5`: published only after the full campaign certifies and export
  reports a ready, complete dataset with the exact 24/6/6 splits.

Qualification failure preserves all available results and reports. Sweep failure
still runs certification to record rejected/missing cases; any rejection blocks
export. Export uses `sr_dataset.inprogress.h5` and publishes the final path only
after verification. A handled export failure marks the temporary file uncertified
and failed. A hard kill can leave an unfinished temporary file; it is never the
final artifact. No partial campaign is published by this workflow, although the
generic exporter supports partial datasets in manual workflows.

Submission/stage failures are not silently retried. Inspect recorded IDs and
`sacct`; cancel any remaining recorded jobs before deliberately abandoning a
campaign. Preserve the directory and resolve the cause before starting a new
campaign root. Existing manual retry tools remain available, but they do not
resume this controller automatically. No dataset from failed qualification is
labelled certified.

## Local verification

```bash
python -m pytest -q tests/test_sr_workflow.py tests/test_sr_pipeline.py
bash -n scripts/slurm/submit-sr.sh scripts/slurm/sr-stage.sbatch scripts/slurm/run-case.sbatch
```

Tests use fake SLURM submissions, synthetic successful gates, and real assessment
of missing evidence. They exercise the dependency plan, resource validation,
failure preservation, profile consistency, complete-export enforcement and status
without HPC access or CFD execution. They do not establish numerical qualification
or resource sufficiency on a particular cluster.
