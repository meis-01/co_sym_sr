# Optional FieldSplit: five-cycle baseline comparison

The integrated optional FieldSplit solver passed five-cycle numerical equivalence
on the **184-cell / 1,364-DOF baseline**. MUMPS remains the default. Measured total
runtime improved modestly (1.10x); isolated solver time improved 1.37x. This does
not establish production suitability for finer meshes or other flow regimes.

## Integration and reproduction

Set `"linear_solver": "fieldsplit"` in a configuration to select right FGMRES,
restart 100, full velocity/pressure Schur factorization, and `selfp` Schur
preconditioning. Velocity uses ILU(0); the approximate pressure Schur matrix
uses MUMPS LU. Tolerances are fixed at `rtol=1e-14`, `atol=0`, maximum 300
iterations. Each solve starts from zero and builds a fresh preconditioner.
Explicit DOLFIN mixed-space index sets identify the two fields. The operator,
row Dirichlet conditions, pressure reference, P2/P1 elements, Picard criterion
(`<1e-7`, maximum 20 iterations), and physical parameters are unchanged.

Omitting `linear_solver`, or using `"mumps"`, retains the original high-level
`df.solve(..., solver_parameters={'linear_solver': 'mumps'})` path. FieldSplit
checks PETSc's convergence reason, finite values, and independently computed
`||Ax-b||2/||b||2 <= 1e-12` on every solve. Failures stop the run and write the
existing failure report; there is no silent fallback. Temporary PETSc options
are restored. Reports now include solver selection/options, cumulative runtime,
solve/iteration counts, true residual maximum (FieldSplit), per-cycle periodicity,
and inlet-flow error. FieldSplit remains marked experimental in metadata.

Reproduce locally from WSL using a new output directory:

```bash
.venv/bin/python scripts/validate_linear_solvers.py studies/new_linear_comparison
.venv/bin/python -m pytest -q
```

The comparison script uses `configs/solver_baseline.json`: the previously
benchmarked Re=60, alpha=12, zero-initialized parabolic-inlet case, 10 mm mesh,
100 time steps per cycle, and unchanged periodic tolerance `0.0005`. Both cycle
limits are five. A bounded warm-up verifies 184 cells before any complete run
and excludes first-call FFC/JIT and solver-library initialization from timing.
Both methods run sequentially on the same machine with one MPI rank and one
OpenMP/BLAS thread. No 4.5 mm or 2 mm full-cycle simulation or HPC access occurred.

## Executed results

Executed 2026-10-25 on Intel Core Ultra 7 258V / WSL Ubuntu, DOLFIN
2019.2.0.13.dev0, PETSc 3.19.6, Python 3.12.3. Both completed **500 time steps**
and **1,995 linear solves**, with identical Picard iteration counts at every
step. The new MUMPS mesh, times, velocity, and pressure arrays were also
**bitwise identical** to the existing pre-integration five-cycle baseline.

| Quantity | MUMPS | FieldSplit |
| --- | ---: | ---: |
| Total `run()` runtime, including mesh/setup and export | 54.67 s | 49.86 s |
| Simulation loop, existing `wall_seconds` | 49.14 s | 46.81 s |
| All linear-system calls, including assembly | 44.82 s | 40.26 s |
| Solver runtime, including factorization/setup | 22.00 s | 16.07 s |
| Final velocity periodic change | 3.869532332e-4 | 3.869532332e-4 |
| Final pressure periodic change | 3.901637236e-5 | 3.901638503e-5 |
| Maximum mass imbalance over 500 steps, divided by mean flow | 1.81e-15 | 1.91e-11 |
| Maximum scaled divergence over 500 steps | 1.09927887414 | 1.09927887413 |
| Final-cycle phase maximum scaled divergence | 0.473016001911 | 0.473016001904 |
| Maximum inlet-flow error over 500 steps, divided by mean flow | 0.104423561347 | 0.104423561347 |
| Final-cycle phase maximum inlet-flow error | 0.103968554409 | 0.103968554409 |

MUMPS solver time comes from DOLFIN's cumulative `LU solver` wall timer, which
excludes assembly; FieldSplit uses `perf_counter` around KSP creation,
configuration, setup/factorization and solve. The MUMPS path was kept intact,
so separate assembly time is not reported for it. FieldSplit assembly/BC time
was 23.17 s. Total timing excludes the subsequent independent validation and
comparison. These are one sequential pair of runs; setup/I/O and system load
affect the small total-runtime difference. The startup microbenchmark's 1.56x
solver speedup on this mesh should not be substituted for this measured result.

The final-cycle exported fields were compared at all 25 phases on the identical
mesh, using FE volume L2 integration (exact for these polynomial squared fields),
without pressure gauge shifts or interpolation:

| Relative error against MUMPS | Velocity | Pressure |
| --- | ---: | ---: |
| Space–time FE L2 | 1.22e-11 | 1.24e-10 |
| Maximum per-phase FE L2 | 1.90e-11 | 5.48e-9 |
| Maximum absolute nodal difference | 6.93e-13 m/s | 3.10e-10 Pa |

All per-phase relative field errors passed the preselected `1e-8` equivalence
gate. Maximum differences across the cycle-periodicity histories were 1.72e-12
(velocity) and 1.41e-11 (pressure). Across all 500 time-step diagnostics, maximum
differences were 2.85e-11 in scaled divergence, 1.91e-11 in scaled flux imbalance,
and 4.08e-15 in scaled inlet-flow error. Every FieldSplit solve converged in at
most 22 iterations; the maximum true relative residual was 1.01e-14.

Both exported datasets passed the existing independent validator, including
flux, quadratic wall nodes, pressure basis, and reconstructed divergence.
This verifies equivalence of the two linear-solver choices; it does **not**
certify physical resolution. The coarse mesh still has about 10.4% inlet-flow
error and final-cycle scaled divergence 0.473, above the existing 0.1 screening
level. Changing the linear solver does not remedy those discretization errors.

## Verification and recommendation

The full local suite passed **42 tests in 20.02 s**. Added regression coverage
checks the actual integrated FieldSplit branch against MUMPS on three evolving
Oseen systems, preservation of the default MUMPS call, restoration of PETSc
options, explicit convergence-failure behavior, solver-key validation, and FE
field-comparison accuracy. Existing tests also passed. No dependencies were added.

Keep FieldSplit available as an **optional experimental solver**, with MUMPS as
default. Five-cycle baseline equivalence passed, but the modest total speedup on
this tiny mesh and the remaining untested flow/mesh regimes do not justify
changing the default or claiming broad production validation. Finer full-cycle
validation requires a separate authorized task.

Full metadata, options, cycle histories, field errors and diagnostics comparisons:
[saved results](benchmarks/linear_solver_five_cycles.json). Local datasets and
per-time-step diagnostics are in `studies/linear_solver_five_cycles/{mumps,fieldsplit}`.
