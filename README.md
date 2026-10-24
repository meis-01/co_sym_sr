# Pulsatile cardiac outlet CFD

This repository solves the full three-dimensional incompressible Newtonian
Navier–Stokes equations on a rigid straight tube or smooth axisymmetric expansion.
It exports independent, SI-unit HDF5 datasets with 25 equally spaced phases of a
numerically converged periodic cycle. No MRI processing or symbolic regression is included.

The CPU backend is **legacy FEniCS/DOLFIN 2019.2**, using the maintained Ubuntu
24.04 binary package. DOLFINx was the preferred backend, but this environment's
Ubuntu repositories do not offer an installable `python3-dolfinx` package. The
implementation uses DOLFIN's mixed finite elements and PETSc/MUMPS, not a reduced
axisymmetric model or analytical fields presented as CFD.

Python packages and the project virtual environment are managed by **uv**.
The WSL CFD environment uses uv's `.venv` with system site packages so it can
load the ABI-matched Ubuntu FEniCS/PETSc libraries. NumPy and other Python
dependencies are pinned in `pyproject.toml` and `uv.lock`. The native CFD stack
stays in the Ubuntu package layer; pip wheels cannot replace those compiled
libraries safely. `environment-ubuntu24.04.lock.txt` records that OS layer.

See [RESULTS.md](RESULTS.md) for executed runs and measured errors.
Configuration-only sweep cases are not counted as simulations.

## Run on Windows through WSL

From Windows, open the Ubuntu WSL shell and run:

```bash
cd /mnt/c/Users/meisa/Projects/co_sym_sr
bash scripts/install-wsl-uv.sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv run --no-sync python -m cardiac_flow run configs/straight.json results/new_poiseuille
uv run --no-sync python -m cardiac_flow run configs/womersley.json results/new_womersley
uv run --no-sync python -m cardiac_flow run configs/expansion.json results/new_expansion
uv run --no-sync python -m cardiac_flow validate results/new_expansion/fields.h5
uv run --no-sync python -m cardiac_flow plot results/new_expansion/fields.h5
uv run --no-sync pytest -q
```

The first setup needs `sudo` only if Ubuntu's native packages are missing. It
creates `.venv`, resolves the Python lockfile, and checks imports for the CFD
libraries. The scientific Python and compiled CFD packages come from Ubuntu;
the uv environment exposes them with system site packages to preserve HDF5,
PETSc, and DOLFIN compatibility. `--no-sync` prevents `uv run` from replacing
those ABI-sensitive system packages with PyPI wheels. Existing nonempty run
directories are never overwritten. First use compiles finite-element kernels
and can take several minutes. Run one benchmark first to populate the compiler
cache before launching additional runs.

Dataset analysis can use the same uv environment; on Windows without WSL, `uv sync`
installs the platform-independent data and analysis dependencies, while CFD runs
still require the Linux FEniCS/PETSc system libraries. Use
`from cardiac_flow.storage import read` for a solver-independent NumPy reader.
Run project commands from the repository root.

## Model and boundary conditions

All quantities are in SI units. The equations are

\[
\rho(\partial_t u+(u\cdot\nabla)u)=-\nabla p+\mu\Delta u,\qquad\nabla\cdot u=0.
\]

Default density is 1060 kg/m³ and dynamic viscosity is 0.0035 Pa·s. The baseline
radius is 12 mm. The inexpensive examples use deliberately low flow, 2 mL/s,
for verification; they are **not representative normal adult aortic flow**.
`configs/physiological.json` provides a higher-flow, higher-resolution starting
configuration (80 mL/s mean), whose numerical adequacy must be established by
the convergence studies before use. A laminar Newtonian rigid-wall idealization
does not establish clinical fidelity at high Reynolds number.
The executed 13-dataset inventory includes coarse benchmarks, three expansion
runs (unstabilized, ratio 1000 and ratio 10000), an independent eccentric-jet
rotation pair, a five-run refinement study and the low-alpha benchmark.
The 48-case configuration sweep and `physiological.json` have **not** been run.

Geometry A has constant radius. Geometry B is a smooth expansion and return:

\[
R(z)=R_0[1+e\sin^2(\pi z/L)].
\]

A Gmsh cylinder is tetrahedralized, then coordinates are mapped radially to this
radius. Both ends have radius R0 and zero radius slope. The continuum geometry is
rotationally symmetric; an unstructured, piecewise planar mesh has discretization
anisotropy. Mesh refinement and rotation studies quantify its effect.

Walls impose u=0. The inlet has prescribed volume flow

\[
Q(t)=\bar Q[1+\sum_{k=1}^K a_k\cos(2\pi kt/T+\phi_k)].
\]

`harmonics` stores `[relative_amplitude, phase_radians]` for successive harmonics.
An empty list gives steady flow. Parabolic inlet profiles have optional tilt:

\[
u_z=\frac{2Q(t)}{\pi R_0^2}(1-r^2/R_0^2)
[1+\epsilon(x\cos\theta+y\sin\theta)/R_0],\quad u_x=u_y=0.
\]

This makes an eccentric axial jet without changing disk-integrated Q or wall
no-slip. Eccentricity is a dimensionless tilt strength in [0,1), not a literal
centerline displacement; orientation is in radians. Faceting introduces a small
inlet-flow error relative to the requested continuum-disk flow; the dataset
records measured inlet flow, which is the correct comparator for mass balance.

The outlet uses the Laplacian-form natural pseudo-traction

\[
\mu\nabla u\,n-pn+\tfrac12\rho\max(-u_{adv}\cdot n,0)u=0.
\]

This is a zero-gauge open outlet with backflow stabilization, consistent with the
chosen weak form. It is **not** symmetric Cauchy stress traction or a Windkessel
model. No extra pressure Dirichlet condition is imposed. For fully developed
straight flow, grad(u)n=0 and p(L)=0, so the analytical benchmark is compatible.
The outlet should be moved farther downstream in applications with persistent
recirculation at the exit; test sensitivity to length and backflow.
Set optional `outlet_pressure` (Pa) to change the constant reference pressure;
the right-hand side traction then becomes -p_out*n. Its default is zero.

## Numerical method and acceptance

Velocity is continuous quadratic P2 and pressure continuous linear P1 on 3D
tetrahedra (inf-sup-stable Taylor–Hood). Backward Euler is first order in time.
Each step solves the full coupled Oseen problem with Picard iteration, using
PETSc MUMPS direct LU on one CPU process. Picard stops at relative velocity
L2 update <1e-7, or the run fails after 20 iterations. No turbulence model,
pressure projection, or artificial compressibility is used. Viscosity is implicit.
An optional consistent grad-div term, `grad_div_ratio * mu * div(u)*div(v)`,
reduces strong divergence while retaining the mixed continuity constraint.
`expansion_stabilized.json` uses a finer 6 mm mesh and ratio 1000 to demonstrate
this control. Large penalties can reduce velocity accuracy on coarse meshes;
compare stabilized solutions under mesh refinement before treating them as
resolution independent.
The executed stabilization comparison is in `studies/stabilization/comparison.json`.
Its substantial velocity and pressure changes are a reason to refine aortic meshes,
even when the local-divergence screening threshold passes. Current datasets are
appropriate for pipeline/format and numerical-method development; they are not
yet resolution-independent quantitative training data.
The stricter follow-up `expansion_strict.json` uses ratio 10000 and initializes
from phase zero of `results/expansion_stabilized/fields.h5`. Run that parent first.
This continuation still solves every subsequent time step and rechecks periodicity.
Dataset initialization requires identical geometry, fluid, waveform and orientation;
it refuses target nodes outside the parent's faceted domain. Other configurations
are independent of previous datasets. `womersley_low_alpha.json` changes the
period to 2.74 s (alpha approximately 10) as a dimensionless benchmark variation;
this long period is not a normal adult resting cardiac period.

Time steps per cycle must be a multiple of 25 and at least 50; all exported
phases coincide with solved time steps. At least two cycles are required. The
velocity **and pressure**, in space-time FE L2 norms over 25 phases, must change
by less than `periodic_tolerance` between consecutive cycles. Pressure is fixed
by the outlet convention. The endpoint is excluded: phase = 0, 1/25, ..., 24/25.

Every time step checks finite solutions, nonlinear convergence, integrated
inlet/outlet mismatch <1% of mean flow, and wall RMS <1e-8 m/s. Reports also record
the strong divergence RMS scaled by R0/Umean. P2/P1 is weakly incompressible, not
pointwise divergence-free: small global flux error alone does not certify small
local divergence. Inspect this diagnostic and its refinement trend. Periodic
convergence alone does not establish mesh or time-step independence.

If any solve fails or periodicity is not reached by `max_cycles`, the run raises
an error and writes `failure.json` and diagnostics; it does not export a final
dataset. Coarse cases are for pipeline verification, not resolution-certified
training data. All time-step diagnostics are retained in `diagnostics.json`.

## Analytical verification

`straight.json` uses steady Poiseuille inlet flow. `womersley.json` uses the exact
fully developed pulsatile inlet for the prescribed flow harmonics, avoiding
comparison of a parabolic entrance flow to an incompatible Womersley solution.
The complex formula uses exp(+iωt), λ = i^(3/2)α√k, and

\[
\hat u_k(r)=\frac{\hat Q_k}{\pi R_0^2}
\frac{1-J_0(\lambda r/R_0)/J_0(\lambda)}
{1-2J_1(\lambda)/(\lambda J_0(\lambda))},\quad
\alpha=R_0\sqrt{2\pi\rho/(T\mu)}.
\]

Benchmark initialization interpolates the analytical field, reducing startup
time. All subsequent fields come from numerical PDE solves; periodic and
analytical errors are measured after evolution. Tests independently verify
the reference PDE residual and its integrated flux. The faceted wall does not
exactly coincide with the mathematical cylinder between boundary vertices;
benchmark error includes this geometry error and must decrease with refinement.
Analytical errors use independent vectorized volume quadrature (125 Duffy
Gauss points per tetrahedron), evaluating the analytical solution directly.
`pressure_error_scaled` divides pressure RMS error by
max(rho*Umean², mu*Umean*L/R0²); it is not relative to the instantaneous analytical
pressure, which can cross zero during the cycle.

## Parameter sweeps and verification studies

```bash
/usr/bin/python3 scripts/study.py sweep
/usr/bin/python3 scripts/study.py rotation --run
/usr/bin/python3 scripts/study.py convergence --run
/usr/bin/python3 scripts/plot_study.py
/usr/bin/python3 scripts/summarize.py
```

Without `--run`, scripts write inspectable JSON configurations only. The sweep
varies Re, α, waveforms, expansion, eccentricity and orientation. Re and α are
controlled independently through mean flow and period:
Re = 2ρQmean/(πR0μ), T = 2πρR0²/(μα²). The default sweep is a coarse screening
grid; refine accepted cases before scientific use. Execution is serial and
resumes only runs that already contain final datasets; investigate failures
before removing or renaming their directories.

The convergence study uses three mesh sizes and three time-step counts. It
uses h = 8, 6, 5 mm and 100, 150, 200 steps per cycle, with a tighter periodic
tolerance of 0.0005 than the smoke cases. These remain inexpensive verification
levels rather than universal aortic resolution recommendations. It
reports analytical errors, changes relative to the finest time step, and
interpolated field differences. It fails if spatial analytical errors or
temporal differences do not decrease. It is not a proof of asymptotic order
or a prescribed error bound. The rotation study compares independently solved
eccentric inlets 90° apart, rotating both coordinates and vectors. Its coarse
acceptance tolerance is 15% in a node-weighted interior velocity norm. Neither
configuration generation nor an unexecuted study is reported as passed.

## Files and downstream data contract

`cardiac_flow/model.py` defines parameters and references; `mesh.py` constructs
geometry; `solver.py` implements CFD; `storage.py` and `analysis.py` are independent
downstream interfaces. `scripts/study.py` orchestrates studies. Configurations
are ordinary JSON; physical parameters, geometry, inlet waveform, initialization,
mesh size, time resolution, duration limit and periodic tolerance are configurable.

Each completed run contains:

| File | Content |
|---|---|
| `fields.h5` | Full 3D velocity, pressure, time, mesh, metadata |
| `phase_00.vtu` … `phase_24.vtu` | Quadratic tetrahedral fields for ParaView |
| `config.json` | Exact input parameters |
| `report.json` | Periodicity, physical diagnostics, solver/provenance metadata |
| `diagnostics.json` | Every time-step diagnostic |
| `validation.json` | Independent exported-data checks, after validation command |
| `overview.png`, `velocity_3d.png` | Plots, after plot command |

HDF5 schema `cardiac-flow-1.0`:

| Dataset | Shape | Meaning |
|---|---|---|
| `mesh/coordinates` | (N2,3) | P2 node coordinates in metres |
| `mesh/tetra10` | (Nc,10) | Zero-based quadratic tetrahedral connectivity |
| `mesh/tetra4` | (Nc,4) | Corner connectivity; corners index the coordinate array |
| `mesh/boundary_triangles` | (Nf,3) | Boundary corner indices |
| `mesh/boundary_tags` | (Nf,) | 1 inlet, 2 outlet, 3 wall |
| `time` | (25,) | Absolute physical seconds in the exported cycle |
| `phase` | (25,) | Dimensionless cardiac phase in [0,1) |
| `velocity` | (25,N2,3) | ux,uy,uz in m/s |
| `pressure` | (25,N2) | Gauge Pa; P1 pressure evaluated at P2 nodes |

Every dataset has a units attribute. Root attributes contain configuration,
report and conventions. `tetra10` follows VTK/meshio order: four corners, then
edge midpoints (0,1), (1,2), (2,0), (0,3), (1,3), (2,3). On barycentric coordinates
λ, P2 corner bases are λi(2λi−1) and edge bases 4λiλj. These allow consistent
velocity derivatives without CFD internals. Pressure remains linear despite
being sampled at quadratic nodes; use the four corner values for derivatives.
The actual mesh is piecewise affine, and boundary triangle orientation is not
guaranteed; reconstruct outward normals from adjacent cells when needed.

The independent validator integrates quadratic inlet/outlet flux exactly and
checks schema, time phases, finite values, connectivity, nondegenerate cells,
wall vertices and reported periodicity. The solver separately checks wall RMS
using FE boundary integrals. Unit tests do not substitute for the executed CFD
benchmarks and refinement studies.
The validator also reconstructs P2 velocity derivatives and independently
integrates squared divergence, checking it against the FE solver's diagnostic.
`status: passed` covers these stated checks. `accuracy_certified` stays false;
review `local_divergence_below_0_1` and the separate convergence study before use.

## References

- [DOLFIN mixed Taylor–Hood formulation](https://olddocs.fenicsproject.org/dolfin/2016.2.0/python/demo/documented/stokes-taylor-hood/python/documentation.html)
- [FEniCSx platform and installation options](https://github.com/FEniCS/dolfinx)
- [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html)
- Womersley, J. R. (1955), Method for the calculation of velocity, rate of flow
  and viscous drag in arteries when the pressure gradient is known.
  [Journal of Physiology, DOI](https://doi.org/10.1113/jphysiol.1955.sp005276).
