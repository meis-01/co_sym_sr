# CFD data for future symbolic regression

All work uses `exp`, the existing serial Navier–Stokes solver, and uv. No SR
algorithm is included. Generating a manifest does not execute CFD. The current
mesh/time settings are **candidates**, not a selected production resolution.
No new SR simulation or production SLURM job has been run during implementation.
The existing 184-cell smoke case completed four cycles, passed automatic output
validation, and was skipped on repeat. An earlier failed attempt exposed an HDF5
wheel/native ABI mismatch; rebuilding with `scripts/install-wsl-uv.sh` restored
the compatible native environment. Use `uv run --no-sync` for this environment.

## Design and splits

`configs/sr_campaign.json` is the versioned design and acceptance policy.
`scripts/study.py sr_sweep` generates the Cartesian product:

| Re | alpha | eccentricity | configurations |
| --- | --- | --- | --- |
| 60, 150, 300 | 12, 16, 20 | 0, 0.1, 0.2, 0.3 | 36 |

Every case uses a rigid straight pipe (R=0.012 m, L=0.06 m), rho=1060 kg/m³,
mu=0.0035 Pa s, zero expansion, orientation zero, and the **parabolic** inlet,
including e=0. Harmonics are always `[[0.5,0.0],[0.15,-0.5]]`. The inversion of
`model.dimensionless` is implemented by `model.flow_parameters`:
Qmean=Re*pi*R*mu/(2*rho); T=2*pi*rho*R²/(mu*alpha²).
Qmean is approximately 3.734, 9.336, 18.672 mL/s; T is approximately 1.903,
1.070, 0.685 s. A fully developed Womersley solution is not an exact reference
for these parabolic inlet conditions; `benchmark` stays false.

The design fixes 24 training, 6 validation and 6 test combinations. Validation
and test tuples are explicit in the policy, with training their complement.
Every split contains all individual Re, alpha and eccentricity levels, while
test **combinations** are unseen in training. All phases/samples of a physical
configuration share its assignment; there is no point-level split. Refinement,
pilot, and independently rotated outputs never enter the training split.
Numerical qualification can inspect test combinations without using them to fit
an SR model. Case IDs, configuration hashes, and array ordering are deterministic.

## Generate and inspect, without running CFD

Run from the repository root in the configured uv environment:

```bash
uv run --no-sync python scripts/experiments.py manifest --study sr_sweep \
  --manifest studies/manifests/sr_sweep.json --output-root studies/sr_sweep
uv run --no-sync python scripts/experiments.py case \
  --manifest studies/manifests/sr_sweep.json --index 0 --dry-run
uv run --no-sync python scripts/study.py sr_pilot --dry-run
```

Generated manifests contain copied configs, independent output paths, Git SHA,
and config hashes. They embed a passed qualification when supplied. Regenerate
manifests on the cluster: absolute output paths belong to the target filesystem.
Local artifacts are under ignored `studies/`; regenerate them from a Git clone.
Existing SR mappings are immutable. An unrun draft may be upgraded with passed
qualification; later previews must supply that same qualification. Changed
designs require fresh manifest/run roots so pending jobs retain their mappings.

## Pilot and qualification

`sr_pilot` selects (Re,alpha,e)=(60,12,0), (150,16,0.1), (300,20,0.3), with
h=4.5 mm and 100 steps/cycle. It is a preliminary performance/environment check,
not accuracy evidence for the fine mesh. It uses the unchanged physical checks;
a coarse pilot may fail them. Do not loosen thresholds to rescue it.

Generate a pilot manifest and run just its first case locally when ready:

```bash
uv run --no-sync python scripts/experiments.py manifest --study sr_pilot \
  --manifest studies/manifests/sr_pilot.json --output-root studies/sr_pilot
uv run --no-sync python scripts/experiments.py case \
  --manifest studies/manifests/sr_pilot.json --index 0
```

For SLURM, first configure site modules or Apptainer and paths as described in
[cluster.md](cluster.md). The scripts request one task and default to one CPU;
requests over **8 CPUs/job** fail before submission. Keep one CPU for the serial
implementation. The following first previews, then submits only the pilot:

```bash
SLURM_DRY_RUN=1 bash scripts/slurm/submit-array.sh sr_pilot
SLURM_ARRAY_CONCURRENCY=2 bash scripts/slurm/submit-array.sh sr_pilot
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

`sr_accuracy` has 32 independently executed configurations. Representatives are
(60,12,0), (150,16,0.1), (300,12,0.3), and (300,20,0.3):

* Three mesh levels, h=4.5, 3, 2 mm, at 400 steps/cycle.
* Three time levels, 100, 200, 400 steps/cycle, at h=2 mm. The finest run is shared
  with the mesh study, giving five distinct mesh/time runs per representative.
* A tighter periodicity run per representative: tolerance 5e-5 and at least six
  cycles, compared with baseline 5e-4 and at least three cycles.
* Separate 45° and 90° eccentric runs at medium/fine mesh for both the middle and
  high-Re/high-alpha eccentric representatives (eight additional runs).

Once the pilot has established suitable wall time and memory, run qualification
separately from the production sweep:

```bash
SLURM_DRY_RUN=1 bash scripts/slurm/submit-array.sh sr_accuracy
# Execute only when resources have been calibrated:
SLURM_ARRAY_CONCURRENCY=2 bash scripts/slurm/submit-array.sh sr_accuracy
source scripts/cluster.env
uv run --no-sync python scripts/sr.py assess \
  --manifest "$MANIFEST_ROOT/sr_accuracy.json" --output "$RUN_ROOT/sr_qualification.json"
```

Alternatively `scripts/study.py sr_accuracy --run` uses the existing manifest
executor locally. These can be long serial runs. Missing evidence produces a
`blocked` assessment and nonzero exit; failed criteria produce `failed`.
Only a passed assessment supplies `selected_numerics`. If candidates fail,
investigate and refine the three-level design; retain old evidence under a
different run root. Changing the policy or core source invalidates qualification.

## Acceptance policy and its limits

These are predeclared project error budgets, **not measured accuracy claims**.
Actual convergence evidence is required before they can qualify a resolution.
The report keeps `numerical_convergence`, `conservation`, and `symmetry` separate.

| Domain | Requirement |
| --- | --- |
| Spatial/temporal velocity | Each finest-grid sampled L2 GCI estimate <=1% |
| Optional pressure | Corresponding estimates <=2% |
| Refinement evidence | Three levels; observed order >=0.5; decreasing, aligned differences (cosine >=0.9) |
| Comparisons | 4,096 reproducible volume samples, all 25 phases; >=99% common mesh coverage; no extrapolation |
| Periodic convergence | Both original solver norms <5e-4; changing to 5e-5 changes velocity <=0.1% (pressure separately) |
| Local divergence | max phase volume RMS(div u)*R/U <0.1 and volume-weighted 95th percentile cell RMS <0.1; cell maxima also reported |
| Mass conservation | max phase abs(Qin-Qout)/Qmean <1%; existing solver checks at every step remain active |
| Inlet fidelity | max phase abs(Qactual-Qprescribed)/Qmean <0.5% |
| Faceted geometry | relative mesh-volume error against pi*R²*L <0.5% |
| Symmetry | Independent rotated velocity L2 error <=1%, pressure separately <=2%, and decreases with refinement or is <=1e-4 |

GCI uses Richardson differences, safety factor 1.25, and conservative effective
order caps of two for space and one for backward Euler. Three-grid evidence and
the safety factor follow the [NASA spatial convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html).
The alignment check rejects inconsistent error directions. This is a sampled
field estimate conditional on the observed refinement regime, not a rigorous
error bound throughout the domain. Representative evidence covers this specified
design; every production case still requires its own conservation/periodicity
checks. Pressure has its own acceptance flag and is omitted unless requested.

HDF5 structural validation retains all previous assertions and continues to say
`accuracy_certified=false`. A separate hash-bound `accuracy.json` records SR
acceptance. Existing benchmarks do not qualify the new campaign: the coarse
Poiseuille example has scaled divergence about 0.166 and prescribed inlet error
about 3.77%, despite passing structural validation. No existing threshold or
solver algorithm has been changed. If new studies reveal a numerical defect,
document it independently before changing the solver.

## Eventual production, retries and acceptance

These commands are for **after** qualification; no production campaign was
launched here. Set partition, account, scratch paths and calibrated resources in
`scripts/cluster.env`. The template allows environment overrides.

```bash
source scripts/cluster.env
export SR_QUALIFICATION="$RUN_ROOT/sr_qualification.json"
SLURM_DRY_RUN=1 bash scripts/slurm/submit-array.sh sr_sweep
bash scripts/slurm/submit-array.sh sr_sweep
uv run --no-sync python scripts/experiments.py status --manifest "$MANIFEST_ROOT/sr_sweep.json"
# Run after results arrive; missing/failed cases receive no accepted certificate:
uv run --no-sync python scripts/sr.py certify \
  --manifest "$MANIFEST_ROOT/sr_sweep.json" --qualification "$SR_QUALIFICATION"
uv run --no-sync python scripts/sr.py export --manifest "$MANIFEST_ROOT/sr_sweep.json" \
  --output "$RUN_ROOT/sr_dataset.h5" --samples 2048 --seed 20261024
```

Without qualification, `SLURM_DRY_RUN=1 ... sr_sweep` still previews the 36-case
array; execution refuses the draft. Locally, generate a qualified manifest using
`scripts/experiments.py manifest --study sr_sweep --qualification PATH ...`,
then execute individual `case --index N` commands. `scripts/study.py sr_sweep
--run --qualification PATH` executes that manifest sequentially.

Resubmitting the same array skips structurally completed outputs, archives failed
or incomplete attempts, and reruns those attempts safely. This restarts a
simulation; it is not a checkpoint resume. Numerical rejection requires resolving
its cause and requalifying changed settings, rather than overwriting successful
files. Per-attempt CFD stdout/stderr are under `<output-root>/.logs/<case-id>/`;
SLURM wrapper logs remain under `LOG_ROOT`. `execution.json` preserves Git and
native/Python environment metadata. Retrieve runs, logs, qualification and manifests
together using the rsync commands in [cluster.md](cluster.md).

## Solver-independent data interface

`cardiac_flow.sr_data` imports only NumPy, HDF5 and the native-array reader.
Each exported simulation owns its full original mesh connectivity and sampled
arrays; different mesh sizes/node counts need no common mesh. Source HDF5 files
are read-only and SHA-256 checked. Complete original attributes, config, reports,
execution provenance and acceptance records are retained.

Sampling is uniform in tetrahedral **volume**, using seeded Dirichlet barycentric
coordinates and exact native P2 velocity/P1 pressure interpolation. The same
spatial samples are used at all 25 phases. Seeds depend on the global seed and
stable case ID, so filtering/reordering does not change another case's samples.
The HDF5 schema `cardiac-flow-sr-1` stores, per simulation:

* SI coordinates, phase=(t/T) mod 1, time in seconds, Re, alpha, eccentricity,
  waveform amplitudes/phases, SI velocity and optional gauge pressure in Pa.
* Coordinates (x/R,y/R,z/L), velocity u/Umean where Umean=Qmean/(pi R²), and optional
  pressure (p-pout)/(rho Umean²). It uses prescribed Qmean, not renormalized actual flow.
* Native `mesh/` datasets, sampling cell indices/barycentric coordinates, units,
  source hash, configuration split, and complete metadata.

```python
from cardiac_flow.sr_data import iter_batches, iter_simulations

for batch in iter_batches("sr_dataset.h5", split="train"):
    X = batch["normalized_inputs"]  # xyz, phase, Re, alpha, e, waveform parameters
    y = batch["dimensionless_velocity"]
    names = batch["normalized_feature_names"]
```

The exporter excludes missing, rejected and stale certificates, records reasons,
and refuses to create an empty export. Partial accepted campaigns are explicitly
marked `complete_campaign=false`. Existing destinations are never overwritten.
Use `--pressure` only when pressure passed its own qualification.

For a format demonstration using existing unverified data:

```bash
uv run --no-sync python scripts/sr.py export --preliminary \
  --dataset results/poiseuille/fields.h5 --dataset results/expansion_strict/fields.h5 \
  --output studies/sr_preliminary_demo.h5 --samples 256 --pressure
```

Such exports are marked preliminary, have no training/test assignment, and are
rejected by readers unless `allow_preliminary=True` is explicitly provided.
They demonstrate the interface, not the SR campaign's accuracy.

## Resource planning and checks

```bash
uv run --no-sync python scripts/sr.py estimate --output studies/sr_resource_estimate.json
PYTHONPATH=. uv run --no-sync python -m pytest -q
bash -n scripts/cluster-common.sh scripts/slurm/*.sh scripts/slurm/*.sbatch
```

The available finest straight-pipe benchmark has 1,250 cells, five cycles at 100
steps/cycle, and took 706.7 seconds on the local machine. h^-3 extrapolation to
the **candidate** 2 mm mesh suggests ~19,500 cells, ~30 MB compressed fields/case,
and ~4 GB for the 36 complete output directories including visualization files.
Two sampled velocity representations at 2,048 points/phase add ~88.5 MB before
compression, plus meshes, metadata and optional pressure. Keep space for the
accuracy study, failed attempts and storage variation; these are planning values.

At an illustrative 40 cycles, coarse pilots extrapolate to ~2.2–2.5 hours each.
The fine candidate extrapolates to ~98–388 hours/case using linear versus
superlinear work models. These are same-machine scenarios, **not cluster timing
predictions**: the benchmark used analytical initialization, whereas SR runs start
from zero; nonlinear work, MUMPS fill-in and required cycles can differ. The
estimator also reports 3- and 160-cycle scenarios. Do not use the default 12-hour
request blindly for fine runs. Measure pilot and refinement Elapsed/MaxRSS, then
choose a site-supported wall time and memory before production. No measured peak
RSS or tested MPI speedup is available. Start pilots with one CPU, 8 GB and 12 h,
and modest array concurrency; the eight-CPU ceiling remains enforced.
On a fresh clone without benchmark outputs, the estimator reports no measured
timings; generate a pilot measurement before making resource projections.

Automated tests exercise the real submitters with fake sbatch/Apptainer commands,
including CPU limits and execution of a spool copy. They do not claim a cluster
submission, container build, or numerical qualification run.
