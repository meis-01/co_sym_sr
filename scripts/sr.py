"""Qualify CFD numerics, certify cases, prepare SR data, and estimate resources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def estimate_resources(roots):
    from cardiac_flow.sr_campaign import specification
    benchmarks = []
    if isinstance(roots, (str, Path)):
        roots = [roots]
    reports = {path.resolve() for root in roots for path in Path(root).rglob('report.json')}
    for path in sorted(reports):
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            c = json.loads(path.with_name('config.json').read_text(encoding='utf-8'))
            fields = path.with_name('fields.h5')
            if not fields.is_file() or report['status'] != 'periodic_converged':
                continue
            benchmarks.append(dict(directory=str(path.parent), cells=report['cells'],
                mesh_size=c['mesh_size'], geometry=c['geometry'], initial=c['initial'], cycles=report['cycles'],
                steps_per_cycle=c['steps_per_cycle'], wall_seconds=report['wall_seconds'],
                fields_bytes=fields.stat().st_size,
                total_output_bytes=sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())))
        except (OSError, KeyError, ValueError):
            continue
    s = specification()
    result = dict(benchmarks=benchmarks, cpus_per_job=1, maximum_cpus_per_job=8,
                  initial_memory_request='8G', initial_wall_time_request='12:00:00',
                  memory_estimate=None, memory_note='No measured peak RSS is available; size from pilot sacct MaxRSS.',
                  simulations=36, accuracy_simulations=32, pilot_simulations=3,
                  sr_velocity_arrays_uncompressed_bytes=36*25*2048*3*8*2,
                  assumptions='Same-machine planning extrapolations only; excludes JIT/setup and queue time. '
                  'MUMPS fill-in, nonlinear iterations, and cycles can change substantially; no HPC runtime guarantee.')
    candidates = [b for b in benchmarks if b['geometry'] == 'straight']
    if candidates:
        anchor = max(candidates, key=lambda b: b['cells'])
        ratio = (anchor['mesh_size']/s['mesh_sizes'][-1])**3
        scenarios = []
        for cycles in (s['min_cycles'], 40, s['max_cycles']):
            step_ratio = cycles*s['steps_per_cycle'][-1]/(anchor['cycles']*anchor['steps_per_cycle'])
            scenarios.append(dict(cycles=cycles,
                wall_seconds_linear_work=anchor['wall_seconds']*step_ratio*ratio,
                wall_seconds_superlinear_work=anchor['wall_seconds']*step_ratio*ratio**1.5))
        result.update(anchor=anchor, estimated_fine_cells=round(anchor['cells']*ratio),
                      estimated_fields_bytes_per_case=round(anchor['fields_bytes']*ratio),
                      estimated_campaign_outputs_bytes=round(36*anchor['total_output_bytes']*ratio),
                      runtime_scenarios=scenarios)
        pilot_ratio = (anchor['mesh_size']/s['mesh_sizes'][0])**3
        result['pilot_40_cycle_runtime_seconds'] = {
            'linear_work': anchor['wall_seconds']*40*s['steps_per_cycle'][0]/(
                anchor['cycles']*anchor['steps_per_cycle'])*pilot_ratio,
            'superlinear_work': anchor['wall_seconds']*40*s['steps_per_cycle'][0]/(
                anchor['cycles']*anchor['steps_per_cycle'])*pilot_ratio**1.5}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    assess = sub.add_parser('assess', help='assess completed sr_accuracy refinement/rotation evidence')
    assess.add_argument('--manifest', type=Path, required=True)
    assess.add_argument('--output', type=Path, required=True)
    certify = sub.add_parser('certify', help='accept/reject each sr_sweep case using qualified numerics')
    certify.add_argument('--manifest', type=Path, required=True)
    certify.add_argument('--qualification', type=Path, required=True)
    export = sub.add_parser('export', help='export certified data or an explicit preliminary demonstration')
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path)
    source.add_argument('--dataset', type=Path, action='append')
    export.add_argument('--preliminary', action='store_true')
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--samples', type=int, default=2048)
    export.add_argument('--seed', type=int, default=20261024)
    export.add_argument('--pressure', action='store_true')
    estimate = sub.add_parser('estimate', help='planning estimates from available completed benchmarks')
    estimate.add_argument('--benchmark-root', type=Path, action='append', help='repeatable; defaults to results and studies')
    estimate.add_argument('--output', type=Path)
    args = parser.parse_args()
    exit_code = 0
    if args.command == 'assess':
        from cardiac_flow.sr_accuracy import assess
        result = assess(args.manifest, args.output)
        print(json.dumps(dict(status=result['status'], output=str(args.output), errors=result['errors']), indent=2))
        exit_code = int(result['status'] != 'passed')
    elif args.command == 'certify':
        from cardiac_flow.sr_accuracy import certify
        result = certify(args.manifest, args.qualification)
        print(json.dumps(result, indent=2))
        exit_code = int(any(status != 'accepted' for status in result.values()))
    elif args.command == 'export':
        if bool(args.dataset) != args.preliminary:
            parser.error('--dataset requires --preliminary; certified exports use --manifest')
        from cardiac_flow.sr_data import export_dataset
        print(json.dumps(export_dataset(args.output, manifest_path=args.manifest,
            preliminary_paths=args.dataset, samples=args.samples, seed=args.seed, pressure=args.pressure), indent=2))
    else:
        result = estimate_resources(args.benchmark_root or [Path('results'), Path('studies')])
        if args.output:
            from cardiac_flow.sr_accuracy import write_report
            write_report(args.output, result)
        print(json.dumps(result, indent=2))
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
