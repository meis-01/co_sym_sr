"""Bounded, serial benchmark of the *production* Oseen systems (not a CFD run).

Intercepts solver.run's variational solve in this process only. MUMPS advances
the unchanged Picard trajectory; every candidate sees the very same A and b.
No solver.py changes, copied weak forms, extra pressure BCs or new dependencies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

# Set before importing numerical libraries; equal thread budget for all methods.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import dolfin as df
import numpy as np
from petsc4py import PETSc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cardiac_flow import solver
from cardiac_flow.model import load

METHODS = ('mumps', 'gmres_ilu', 'schur_ilu_lu', 'schur_ilu_ilu', 'schur_hypre_lu')
RTOL = 1e-14
MAX_IT = 300
EQUIVALENCE_TOL = 1e-8
TRUE_RESIDUAL_TOL = 1e-12


class _Finished(BaseException):
    """Exit the bounded experiment without writing a production failure report."""


def assemble(equation, bcs):
    # DOLFIN's default nonsymmetric LinearVariationalSolver uses row BCs.
    A, b = df.assemble(equation.lhs), df.assemble(equation.rhs)
    for bc in bcs:
        bc.apply(A, b)
    return A, b


def indices(W):
    fields = [np.asarray(W.sub(i).dofmap().dofs(), dtype=PETSc.IntType) for i in (0, 1)]
    joined = np.concatenate(fields)
    if not np.array_equal(np.sort(joined), np.arange(W.dim())):
        raise RuntimeError('Velocity/pressure indices must partition the mixed space')
    return fields, [PETSc.IS().createGeneral(f, comm=PETSc.COMM_SELF) for f in fields]


def options(method):
    if method == 'mumps':
        return dict(ksp_type='preonly', pc_type='lu', pc_factor_mat_solver_type='mumps')
    if method == 'gmres_ilu':
        return dict(ksp_type='gmres', ksp_pc_side='right', pc_type='ilu',
                    ksp_gmres_restart=100)
    velocity, pressure = method.removeprefix('schur_').split('_')
    result = dict(ksp_type='fgmres', ksp_pc_side='right', ksp_gmres_restart=100,
                  pc_type='fieldsplit', pc_fieldsplit_type='schur',
                  pc_fieldsplit_schur_fact_type='full', pc_fieldsplit_schur_precondition='selfp',
                  fieldsplit_u_ksp_type='preonly', fieldsplit_u_pc_type=velocity,
                  fieldsplit_p_ksp_type='preonly', fieldsplit_p_pc_type=pressure)
    if velocity == 'hypre':
        result.update(fieldsplit_u_pc_hypre_type='boomeramg',
                      fieldsplit_u_pc_hypre_boomeramg_max_iter=1,
                      fieldsplit_u_pc_hypre_boomeramg_tol=0.0)
    if pressure == 'lu':
        result['fieldsplit_p_pc_factor_mat_solver_type'] = 'mumps'
    return result


def system_digest(A, b):
    digest = hashlib.sha256()
    for array in (*A.getValuesCSR(), b.getArray(readonly=True)):
        digest.update(array.tobytes())
    return digest.hexdigest()


def solve_once(A, b, split_is, method):
    """Fresh PC including setup/factorization, zero initial guess, no reuse."""
    prefix = 'linear_bench_'
    opts = PETSc.Options()
    settings = options(method)
    start = time.perf_counter()
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    x = A.createVecRight()
    x.set(0)
    record = dict(method=method)
    try:
        for key, value in settings.items():
            opts[prefix + key] = value
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A)
        if method.startswith('schur_'):
            ksp.getPC().setType('fieldsplit')
            ksp.getPC().setFieldSplitIS(('u', split_is[0]), ('p', split_is[1]))
        ksp.setFromOptions()
        ksp.setTolerances(rtol=RTOL, atol=0.0, divtol=1e8, max_it=MAX_IT)
        if method != 'mumps':
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setInitialGuessNonzero(False)
        ksp.setUp()
        setup_end = time.perf_counter()
        ksp.solve(b, x)
        end = time.perf_counter()
        record.update(setup_seconds=setup_end-start, solve_seconds=end-setup_end,
                      total_seconds=end-start, reason=int(ksp.getConvergedReason()),
                      iterations=ksp.getIterationNumber())
        residual = b.duplicate()
        A.mult(x, residual)
        residual.axpy(-1, b)
        relative_residual = residual.norm()/b.norm()
        record['true_relative_residual'] = relative_residual if np.isfinite(relative_residual) else None
        residual.destroy()
        values = x.getArray().copy()
        return values if np.all(np.isfinite(values)) else None, record
    except PETSc.Error as error:
        record.update(total_seconds=time.perf_counter()-start,
                      reason=int(ksp.getConvergedReason()),
                      iterations=ksp.getIterationNumber(), error=str(error))
        return None, record
    finally:
        ksp.destroy()
        x.destroy()
        for key in settings:
            del opts[prefix + key]


def benchmark(config, output, *, systems=8, repeats=5, methods=METHODS, expected_cells=None):
    if df.MPI.size(df.MPI.comm_world) != 1:
        raise ValueError('Benchmark requires serial execution')
    if not 1 <= systems <= 30 or not 1 <= repeats <= 20:
        raise ValueError('Use 1..30 systems and 1..20 repeats')
    c = load(config)
    if c['mesh_size'] < 0.0045:
        raise ValueError('This bounded benchmark forbids meshes finer than 4.5 mm')
    if 'mumps' not in methods or any(m not in METHODS for m in methods):
        raise ValueError('Select known methods and include mumps as reference')
    report = dict(schema='cardiac-flow-linear-benchmark-1', config=c,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), dolfin=df.__version__,
                                   petsc=PETSc.Sys.getVersion(), platform=platform.platform(),
                                   cpu=next((line.split(':', 1)[1].strip() for line in
                                             Path('/proc/cpuinfo').read_text().splitlines()
                                             if line.startswith('model name')), 'unknown'),
                                   threads={v: os.environ[v] for v in
                                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                                   packages={p: PETSc.Sys.hasExternalPackage(p) for p in ('mumps', 'hypre')}),
                  source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__).resolve(), REPO/'cardiac_flow/solver.py',
                                           REPO/'cardiac_flow/mesh.py', REPO/'cardiac_flow/model.py')},
                  systems_requested=systems, repeats=repeats, rtol=RTOL, atol=0.0,
                  equivalence_tolerance=EQUIVALENCE_TOL,
                  true_residual_tolerance=TRUE_RESIDUAL_TOL,
                  max_iterations=MAX_IT, options={m: options(m) for m in methods}, systems=[])
    original_solve = df.solve
    fields = split_is = None

    def intercept(equation, result, bcs, **kwargs):
        nonlocal fields, split_is
        if len(report['systems']) == systems:
            raise _Finished()
        W = result.function_space()
        if fields is None:
            cells = W.mesh().num_cells()
            if expected_cells is not None and cells != expected_cells:
                raise ValueError(f'Expected {expected_cells} cells, got {cells}')
            fields, split_is = indices(W)
            report.update(cells=cells, mixed_dofs=W.dim(),
                          velocity_dofs=len(fields[0]), pressure_dofs=len(fields[1]))
        # Warm form compilation/BC application before measuring assembly.
        assemble(equation, bcs)
        assembly_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            A_df, b_df = assemble(equation, bcs)
            assembly_times.append(time.perf_counter()-start)
        A, b = df.as_backend_type(A_df).mat(), df.as_backend_type(b_df).vec()
        before = system_digest(A, b)
        pressure_block = A.createSubMatrix(split_is[1], split_is[1])
        pressure_norm = pressure_block.norm()
        pressure_block.destroy()
        if pressure_norm != 0:
            raise RuntimeError('Expected zero pressure-pressure block')
        reference, direct = solve_once(A, b, split_is, 'mumps')
        if direct.get('reason', 0) <= 0 or direct['true_relative_residual'] > TRUE_RESIDUAL_TOL:
            raise RuntimeError(f'MUMPS reference failed: {direct}')
        # Read-only metadata from the production caller; never alter its state.
        state = sys._getframe(1).f_locals
        row = dict(index=len(report['systems']), time=state['t0']+state['dt'],
                   picard_iteration=state['iteration']+1, cycle=state['cycle'],
                   system_sha256=before, assembly_seconds=assembly_times, trials=[])
        del state
        if not report['systems']:
            # Independently check the intercept assembly and direct PETSc path
            # against the original high-level variational solve (outside timing).
            original_solve(equation, result, bcs, **kwargs)
            row['original_df_solve_relative_difference'] = float(
                np.linalg.norm(result.vector().get_local()-reference)/np.linalg.norm(reference))
            if row['original_df_solve_relative_difference'] > 1e-12:
                raise RuntimeError('Assembly differs from the production variational solve')
        # First invocation of each PC/library is never part of the timings.
        for method in methods:
            solve_once(A, b, split_is, method)
        for repeat in range(repeats):
            # Rotate order to reduce systematic warm-cache / timing-order bias.
            order = list(methods)[repeat % len(methods):] + list(methods)[:repeat % len(methods)]
            for method in order:
                values, trial = solve_once(A, b, split_is, method)
                trial['repeat'] = repeat
                if values is not None:
                    for name, ix in zip(('velocity', 'pressure', 'mixed'),
                                        (*fields, np.arange(W.dim()))):
                        trial[f'{name}_relative_difference'] = float(
                            np.linalg.norm(values[ix]-reference[ix])/max(np.linalg.norm(reference[ix]), 1e-300))
                trial['validated'] = bool(trial.get('reason', 0) > 0
                    and trial.get('true_relative_residual') is not None
                    and trial['true_relative_residual'] <= TRUE_RESIDUAL_TOL
                    and trial.get('velocity_relative_difference', float('inf')) < EQUIVALENCE_TOL
                    and trial.get('pressure_relative_difference', float('inf')) < EQUIVALENCE_TOL)
                row['trials'].append(trial)
        if system_digest(A, b) != before:
            raise RuntimeError('A solver modified the shared matrix or RHS')
        # Candidates never influence the next Picard matrix or stopping decision.
        result.vector().set_local(reference)
        result.vector().apply('insert')
        report['systems'].append(row)
        print(f"system={row['index']} cells={report['cells']} "
              + ' '.join(f"{m}={np.median([t['total_seconds'] for t in row['trials'] if t['method'] == m]):.4f}s"
                         for m in methods), flush=True)

    try:
        with patch.object(df, 'solve', intercept):
            solver.run(c, output)
    except _Finished:
        pass
    finally:
        if split_is:
            for index_set in split_is:
                index_set.destroy()
    if len(report['systems']) != systems:
        raise RuntimeError('Production run ended before all benchmark systems were captured')
    summary = {}
    for method in methods:
        trials = [t for s in report['systems'] for t in s['trials'] if t['method'] == method]
        summary[method] = dict(trials=len(trials), validated=all(t['validated'] for t in trials),
                               median_total_seconds=float(np.median([t['total_seconds'] for t in trials])),
                               reasons=sorted(set(t['reason'] for t in trials)),
                               max_iterations=max(t['iterations'] for t in trials))
        for key in ('true_relative_residual', 'velocity_relative_difference', 'pressure_relative_difference'):
            measured = [t[key] for t in trials if t.get(key) is not None]
            summary[method]['max_'+key] = max(measured) if measured else None
    report['summary'] = summary
    report['median_assembly_seconds'] = float(np.median([t for s in report['systems'] for t in s['assembly_seconds']]))
    Path(output, 'linear_benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--systems', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--expected-cells', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    args = parser.parse_args()
    benchmark(args.config, args.output, systems=args.systems, repeats=args.repeats,
              methods=args.methods, expected_cells=args.expected_cells)


if __name__ == '__main__':
    main()
