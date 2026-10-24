"""Generate reproducible parameter grids and optionally execute them serially.

Run from the repository root: /usr/bin/python3 scripts/study.py sweep --run
"""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac_flow.model import load
from cardiac_flow.compare import compare
from cardiac_flow.sr_campaign import STUDIES, configurations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', choices=['sweep', 'convergence', 'rotation', *STUDIES])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='inspect manifest cases without simulation')
    parser.add_argument('--qualification', type=Path, help='passed SR accuracy qualification')
    parser.add_argument('--root', default='studies')
    args = parser.parse_args()
    root = Path(args.root)/args.study
    root.mkdir(parents=True, exist_ok=True)
    cases = {}
    if args.study in STUDIES:
        cases = configurations(args.study, args.qualification)
    elif args.study == 'sweep':
        base = load('configs/expansion.json')
        for re, alpha, expansion, waveform, eccentricity, angle in itertools.product(
                [60., 300.], [10., 18.], [0., .3], [0, 1], [0., .5], [0., 1.5707963267948966]):
            if eccentricity == 0 and angle != 0:
                continue
            c = copy.deepcopy(base)
            c.update(mean_flow=re*np.pi*c['radius']*c['mu']/(2*c['rho']),
                     period=2*np.pi*c['rho']*c['radius']**2/(c['mu']*alpha**2),
                     expansion=expansion, geometry='expansion' if expansion else 'straight',
                     eccentricity=eccentricity, orientation=angle,
                     harmonics=[[.5, 0.]] if waveform == 0 else [[.6, 0.], [.2, -.5]],
                     max_cycles=80)
            cases[f'case_{len(cases):03d}'] = c
    elif args.study == 'rotation':
        base = load('configs/expansion.json')
        # Faster convective flushing than the very low-flow smoke case.
        base.update(mean_flow=1e-5, eccentricity=.5, max_cycles=40)
        cases = {'orientation_0': dict(base, orientation=0.),
                 'orientation_90': dict(base, orientation=1.5707963267948966)}
    else:
        base = load('configs/womersley.json')
        base.update(max_cycles=50, periodic_tolerance=.0005)
        cases = {'base': base, 'mesh_medium': dict(base, mesh_size=.006),
                 'mesh_fine': dict(base, mesh_size=.005),
                 'dt_medium': dict(base, steps_per_cycle=150),
                 'dt_fine': dict(base, steps_per_cycle=200)}
    for name, config in cases.items():
        config_path = root/f'{name}.json'
        config_path.write_text(json.dumps(config, indent=2))
        if args.run and not args.dry_run and args.study not in STUDIES and not (root/name/'fields.h5').exists():
            subprocess.run([sys.executable, '-m', 'cardiac_flow', 'run', str(config_path), str(root/name)],
                           check=True, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    if args.study in STUDIES and (args.run or args.dry_run):
        from scripts.experiments import build_manifest, run_case, REPO
        path = root.parent / f'{args.study}.manifest.json'
        manifest = build_manifest(args.study, None, path.resolve(), root.resolve(), REPO,
                                  args.qualification)
        for entry in manifest['cases']:
            run_case(path, entry['index'], dry_run=args.dry_run)
    if args.run and not args.dry_run and args.study in ('rotation', 'convergence'):
        if args.study == 'rotation':
            result = compare(root/'orientation_0/fields.h5', root/'orientation_90/fields.h5',
                             angle=1.5707963267948966)
            result['passed'] = result['relative_velocity_l2_nodes'] < .15
        else:
            result = {name: compare(root/'base/fields.h5', root/name/'fields.h5')
                      for name in cases if name != 'base'}
            # Record analytical errors as well; differences alone do not establish convergence.
            for name in cases:
                report = json.loads((root/name/'report.json').read_text())
                result[name] = dict(result.get(name, {}),
                                    analytical_velocity_error=max(r['velocity_relative_l2'] for r in report['phases']))
            errors = [result[x]['analytical_velocity_error'] for x in ('base', 'mesh_medium', 'mesh_fine')]
            result['mesh_error_decreases'] = bool(errors[2] < errors[1] < errors[0])
            tc = compare(root/'dt_fine/fields.h5', root/'base/fields.h5')
            tm = compare(root/'dt_fine/fields.h5', root/'dt_medium/fields.h5')
            temporal_coarse = tc['relative_velocity_l2_nodes']
            temporal_medium = tm['relative_velocity_l2_nodes']
            result['temporal_to_finest'] = dict(coarse=temporal_coarse, medium=temporal_medium,
                error_decreases=temporal_medium < temporal_coarse,
                pressure_coarse=tc['relative_pressure_l2_nodes'], pressure_medium=tm['relative_pressure_l2_nodes'],
                pressure_error_decreases=tm['relative_pressure_l2_nodes'] < tc['relative_pressure_l2_nodes'])
            result['observed_mesh_error_slope'] = float(np.polyfit(
                np.log([cases[x]['mesh_size'] for x in ('base', 'mesh_medium', 'mesh_fine')]),
                np.log(errors), 1)[0])
            result['passed'] = (result['mesh_error_decreases'] and result['temporal_to_finest']['error_decreases']
                                and result['temporal_to_finest']['pressure_error_decreases'])
        (root/'comparison.json').write_text(json.dumps(result, indent=2))
        if not result['passed']:
            raise RuntimeError('Study did not meet acceptance criterion; inspect comparison.json')
    print(f'{len(cases)} configurations written to {root}')


if __name__ == '__main__':
    main()
