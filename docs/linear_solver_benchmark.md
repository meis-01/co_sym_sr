# Local FEniCS linear-solver benchmark

This records the initial standalone benchmark. The subsequent optional integration
and five-cycle validation are documented [separately](linear_solver_five_cycles.md).

PETSc FieldSplit is feasible and promising: velocity ILU plus a pressure Schur
approximation solved the tested 9,257-DOF systems **7.29 times faster** than
MUMPS, including fresh preconditioner setup. MUMPS remains the production default.
These are bounded startup-system benchmarks, not validation across cardiac cycles.

Executed locally on 2026-10-25, branch `exp` (base `bdd9ff5`), WSL Ubuntu,
Intel Core Ultra 7 258V; Python 3.12.3, DOLFIN 2019.2.0.13.dev0,
PETSc/petsc4py 3.19.6. MUMPS and HYPRE were already available. All methods used
one MPI rank and one OpenMP/BLAS thread. No dependencies were installed.

## Formulation and method

The production operator is the nonsymmetric mixed P2/P1 backward-Euler Oseen
system with Picard convection, optional grad-div, and the existing outlet
backflow term. In block notation it is `[F G; B 0]`. Only inlet and wall velocity
Dirichlet conditions are applied; the natural outlet fixes the pressure reference.
No pressure pinning, nullspace removal, or equation changes were introduced.

`scripts/benchmark_linear_solvers.py` temporarily intercepts `df.solve` inside
the existing `solver.run`, in its own process. It assembles the actual production
forms and applies the same row Dirichlet conditions. Each method receives the
same PETSc matrix and RHS; CSR/RHS hashes verify neither was modified. The first
system on each mesh was also solved by the original high-level `df.solve`:
the relative coefficient difference from the benchmark MUMPS path was **zero**.
Only MUMPS solutions advance the original Picard loop. Its velocity L2 update
criterion `<1e-7`, 20-iteration limit, physical checks, and all parameters remain
unchanged. The experiment stops after eight systems, before completing a cycle.

Explicit index sets come from `W.sub(0/1).dofmap().dofs()`, with checks that they
partition the interleaved mixed space. The zero pressure block is also checked.
FieldSplit uses full block factorization and PETSc's `selfp` approximation
`S_p = -B diag(F)^(-1) G` to the Schur complement `S = -B F^(-1) G`.
The default `a11` approximation would use the zero pressure block and is unsuitable
for LU/ILU here. See the official [Schur preconditioner documentation](https://petsc.org/main/manualpages/PC/PCFieldSplitSetSchurPre/)
and [FieldSplit documentation](https://petsc.org/main/manualpages/PC/PCFIELDSPLIT/).

Configurations tested:

| Method | PETSc configuration |
| --- | --- |
| MUMPS | `ksp_type=preonly`, `pc_type=lu`, `pc_factor_mat_solver_type=mumps` |
| Monolithic ILU control | right GMRES, restart 100, `pc_type=ilu` (ILU(0)) |
| Schur ILU/LU | right FGMRES, restart 100; FieldSplit Schur/full/selfp; velocity `preonly+ilu`; pressure `preonly+lu`, MUMPS |
| Schur ILU/ILU | same, pressure `preonly+ilu` |
| Schur AMG/LU | same, velocity `preonly+hypre/boomeramg`, one cycle (`max_iter=1`, `tol=0`); pressure LU/MUMPS |

All iterative runs use an unpreconditioned outer residual norm, `rtol=1e-14`,
`atol=0`, at most 300 iterations, and a zero initial guess. Both field subsolves
are `preonly`; their approximate inverses form the preconditioner for the outer
iteration. All detailed option names are saved in the raw results.

## Workload, timing, and numerical checks

The physical case is the existing Re=60, alpha=12 straight parabolic-inlet pilot:
radius 0.012 m, length 0.06 m, density 1060 kg/m3, viscosity 0.0035 Pa s,
mean flow 3.7343459844557917e-6 m3/s, period 1.902907550174389 s,
100 steps/cycle, zero initialization, zero grad-div and outlet pressure, and
the original two harmonics. The meshes use 10 mm and 4.5 mm respectively;
all other physical and time-discretization parameters agree.

Each mesh supplies eight distinct systems: all five Picard iterations at
`t=0.0190290755 s`, then the first three at `t=0.0380581510 s`.
Each system/method has five measured repetitions (40 solves per method/mesh).
Form assembly/BC application and each solver are warmed outside timing;
first-call FFC JIT and library initialization are excluded. Timings use fresh
matrices for assembly and a fresh KSP/preconditioner for every solver trial.
No factorization reuse is credited. Method order rotates between repetitions.
Solver time includes KSP configuration, setup/factorization, and solve, but
excludes residual/error checks. PETSc may defer some setup into `solve`, so
**total solver time**, rather than its setup/solve subdivisions, is the comparison.

Assembly (including matrix, RHS, allocation, and BC application), median:

| Cells | Mixed DOFs (velocity + pressure) | Assembly |
| --- | --- | --- |
| 184 | 1,364 (1,281 + 83) | 11.35 ms |
| 1,647 | 9,257 (8,790 + 467) | 95.97 ms |

Linear-solver medians, including setup/factorization:

| Method | 184 cells | 1,647 cells | Validated trials, 184 / 1,647 |
| --- | --- | --- | --- |
| MUMPS | 10.31 ms | 403.56 ms | 40/40 / 40/40 |
| Schur ILU/LU | 6.62 ms | **55.33 ms** | 40/40 / 40/40 |
| Schur ILU/ILU | 5.18 ms | 58.56 ms | 40/40 / **30/40** |
| Schur AMG/LU | 8.84 ms | 82.33 ms | 40/40 / 40/40 |
| Monolithic GMRES+ILU | failed | failed | 0/40 / 0/40 |

For the 1,647-cell case, the MUMPS timing interquartile range is 394.52–418.57 ms;
Schur ILU/LU is 53.73–57.15 ms. Schur ILU/LU gives a 1.56x linear-solve speedup
on 184 cells and 7.29x on 1,647 cells. Adding the measured assembly medians
gives estimated assembly-plus-solve speedups of **1.21x and 3.30x** respectively.
Those are component estimates, not measured full-simulation speedups.

Acceptance requires a positive PETSc convergence reason, independently computed
`||Ax-b||2 / ||b||2 <= 1e-12`, and separate relative velocity and pressure
coefficient errors against MUMPS `<1e-8`. No pressure gauge shift is applied.
The errors below are coefficient-vector Euclidean norms, not FE L2 errors.

Worst numerical results over all measured trials:

| Cells / method | Iterations | True relative residual | Velocity difference | Pressure difference |
| --- | --- | --- | --- | --- |
| 184 / MUMPS | 1 | 3.95e-15 | 0 | 0 |
| 184 / Schur ILU/LU | 21–22 | 1.00e-14 | 2.43e-11 | 4.08e-9 |
| 184 / Schur ILU/ILU | 23 | 4.93e-15 | 2.26e-11 | 3.36e-9 |
| 184 / Schur AMG/LU | 25 | 6.95e-15 | 1.97e-11 | 3.38e-9 |
| 1,647 / MUMPS | 1 | 1.05e-16 | 0 | 0 |
| 1,647 / Schur ILU/LU | 21–22 | 4.96e-15 | 2.47e-11 | 2.50e-9 |
| 1,647 / Schur ILU/ILU | 28–29 | 9.09e-15 | 1.18e-10 | **1.21e-8** |
| 1,647 / Schur AMG/LU | 28 | 6.21e-15 | 1.18e-11 | 9.09e-10 |

All Schur trials returned `CONVERGED_RTOL` (2). The pressure ILU variant failed
the strict pressure-equivalence gate on ten fine-mesh trials despite convergence;
it is not the selected candidate. Monolithic ILU reproduced `DIVERGED_PC_FAILED`
(-11) at iteration zero on 184 cells. On 1,647 cells it instead stagnated or
broke down: reasons -3/-5, 100–300 iterations, worst true residual 3.62.

A preliminary 184-cell run at outer `rtol=1e-12` achieved true residuals below
1e-12 but pressure differences as large as 1.40e-6 for Schur ILU/ILU. Small
pressure fields and the mixed units/Dirichlet rows make a global residual alone
insufficient. Tightening to 1e-14, with independent field-error gates, produced
the final results above. No numerical tolerance was relaxed to improve timings.

## Changes, verification, and recommendation

Added the standalone benchmark, two native-stack regression tests, this report,
and raw results: [184 cells](benchmarks/linear_solver_184.json) and
[1,647 cells](benchmarks/linear_solver_1647.json). Raw data include every trial,
assembly samples, full configuration, PETSc options, source hashes, and system
hashes. `cardiac_flow/solver.py` and production configuration files are unchanged.

Verification: `.venv/bin/python -m pytest -q` passed **37 tests in 37.68 s**
(the original suite passed 35 tests before additions). New tests cover a real
production Oseen assembly/MUMPS/Schur comparison and rejection of meshes finer
than 4.5 mm before output creation. FFC emitted 1,742 NumPy `product`
deprecation warnings during test JIT compilation; there were no test failures.

**Recommendation:** retain MUMPS as default. Schur ILU/LU is a validated and
substantially faster experimental candidate for these startup systems. Production
integration still needs bounded comparisons at other waveform phases, flow
regimes, and active backflow/grad-div settings, followed by nonlinear/periodic
trajectory checks. This experiment does not establish mesh-independent scaling,
whole-cycle numerical equivalence, or performance on 2 mm meshes. No full cardiac
cycle, 2 mm CFD run, or HPC access occurred during this task.

To reproduce from the saved parameter snapshots in the local WSL environment
(use new output directories):

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
Path('studies').mkdir(exist_ok=True)
for cells in (184, 1647):
    data = json.loads(Path(f'docs/benchmarks/linear_solver_{cells}.json').read_text())
    Path(f'studies/linear_solver_{cells}_config.json').write_text(json.dumps(data['config']))
PY
.venv/bin/python scripts/benchmark_linear_solvers.py studies/linear_solver_184_config.json studies/linear_recheck_184 --expected-cells 184
.venv/bin/python scripts/benchmark_linear_solvers.py studies/linear_solver_1647_config.json studies/linear_recheck_1647 --expected-cells 1647
.venv/bin/python -m pytest -q
```
