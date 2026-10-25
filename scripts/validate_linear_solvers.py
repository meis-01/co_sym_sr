"""Five-cycle comparison restricted to the 184-cell baseline; no fine-mesh runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from petsc4py import PETSc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cardiac_flow.analysis import validate
from cardiac_flow.model import load
from cardiac_flow.reference import ReferenceQuadrature
from cardiac_flow.solver import run
from cardiac_flow.storage import read
from scripts.benchmark_linear_solvers import benchmark


def field_errors(reference, candidate):
    """Exact FE volume L2 norms on the identical mesh; no pressure gauge shift."""
    for key in ('coordinates', 'tetra10', 'tetra4', 'time'):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f'Comparison requires identical {key}')
    quad = ReferenceQuadrature(reference['coordinates'], reference['tetra4'])
    result = {}
    for field in ('velocity', 'pressure'):
        a, b = reference[field], candidate[field]
        if field == 'pressure':
            a, b = a[..., None], b[..., None]
        aq = np.einsum('qi,tcij->tcqj', quad.basis, a[:, reference['tetra10']])
        dq = np.einsum('qi,tcij->tcqj', quad.basis, (b-a)[:, reference['tetra10']])
        norm2 = np.einsum('tcqj,cq->t', aq*aq, quad.weights)
        diff2 = np.einsum('tcqj,cq->t', dq*dq, quad.weights)
        result[field] = dict(relative_space_time_l2=float(np.sqrt(diff2.sum()/norm2.sum())),
                             max_phase_relative_l2=float(np.sqrt(diff2/norm2).max()),
                             phase_relative_l2=np.sqrt(diff2/norm2).tolist(),
                             max_absolute_nodal_difference=float(np.max(np.abs(b-a))))
    return result


def compare_runs(output):
    output = Path(output)
    fields = field_errors(read(output/'mumps/fields.h5'), read(output/'fieldsplit/fields.h5'))
    reports = {m: json.loads((output/m/'report.json').read_text()) for m in ('mumps', 'fieldsplit')}
    diagnostics = {m: json.loads((output/m/'diagnostics.json').read_text()) for m in reports}
    a, b = diagnostics['mumps'], diagnostics['fieldsplit']
    if len(a) != 500 or len(b) != 500:
        raise ValueError('Expected 500 completed time steps per run')
    np.testing.assert_array_equal([r['time'] for r in a], [r['time'] for r in b])
    differences = {key: max(abs(x[key]-y[key]) for x, y in zip(a, b))
                   for key in ('inlet_flow', 'outlet_flow', 'flux_error', 'divergence_scaled',
                               'inlet_flow_error', 'wall_rms', 'max_speed')}
    qmean = read(output/'mumps/fields.h5')['config']['mean_flow']
    differences['inlet_flow_scaled'] = differences['inlet_flow']/qmean
    differences['outlet_flow_scaled'] = differences['outlet_flow']/qmean
    periodic_differences = {field: max(abs(x[f'periodic_{field}']-y[f'periodic_{field}'])
        for x, y in zip(reports['mumps']['cycle_history'][1:], reports['fieldsplit']['cycle_history'][1:]))
        for field in ('velocity', 'pressure')}
    equivalent = (all(x['max_phase_relative_l2'] < 1e-8 for x in fields.values())
        and all(r['cycles'] == 5 and r['status'] == 'periodic_converged' for r in reports.values())
        and max(periodic_differences.values()) < 1e-8
        and max(differences[k] for k in ('inlet_flow_scaled', 'outlet_flow_scaled',
                                       'flux_error', 'divergence_scaled', 'inlet_flow_error')) < 1e-8)
    return dict(numerically_equivalent=bool(equivalent), field_errors=fields,
                max_time_step_diagnostic_differences=differences,
                max_cycle_periodicity_differences=periodic_differences,
                identical_picard_iteration_counts=all(x['picard_iterations'] == y['picard_iterations']
                                                       for x, y in zip(a, b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Use a new comparison directory: {args.output}')
    config = REPO/'configs/solver_baseline.json'
    c = load(config)
    if c['mesh_size'] != 0.01 or c['min_cycles'] != 5 or c['max_cycles'] != 5:
        raise ValueError('Only the 10 mm, five-cycle baseline is authorized by this script')
    # Warm the actual forms, DOLFIN and both PETSc backends outside total timing.
    # The mesh count is checked before the first solve or any complete run.
    with tempfile.TemporaryDirectory(prefix='cfd-linear-warm-') as temporary:
        benchmark(config, Path(temporary)/'warm', systems=1, repeats=1,
                  methods=('mumps', 'schur_ilu_lu'), expected_cells=184)
    args.output.mkdir(parents=True)
    runs = {}
    for method in ('mumps', 'fieldsplit'):
        parameters = dict(c)
        if method == 'fieldsplit':
            parameters['linear_solver'] = method
        start = time.perf_counter()
        report = run(parameters, args.output/method)
        total_seconds = time.perf_counter()-start
        if report['cells'] != 184 or report['cycles'] != 5:
            raise RuntimeError('Unexpected workload')
        validation = validate(args.output/method/'fields.h5')
        runs[method] = dict(total_seconds=total_seconds, report=report, validation=validation)
    result = dict(schema='cardiac-flow-five-cycle-linear-comparison-1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), petsc=PETSc.Sys.getVersion(),
                      platform=platform.platform(),
                      cpu=next(line.split(':', 1)[1].strip() for line in
                               Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                      threads={v: os.environ[v] for v in
                               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
                  comparison=compare_runs(args.output), runs=runs,
                  source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__).resolve(), config)})
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(comparison=result['comparison'], runtimes={m: dict(
        total_seconds=r['total_seconds'], **r['report']['linear_solver_statistics'])
        for m, r in runs.items()}), indent=2), flush=True)
    if not result['comparison']['numerically_equivalent']:
        raise SystemExit('Numerical equivalence failed: inspect comparison.json')


if __name__ == '__main__':
    main()
