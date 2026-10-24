"""SR experiment design. Mesh/time settings are candidates until qualified."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from .model import flow_parameters

SPEC_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'sr_campaign.json'
STUDIES = ('sr_sweep', 'sr_accuracy', 'sr_pilot')
SOLVER_FILES = ('solver.py', 'mesh.py', 'model.py', 'storage.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def specification():
    return json.loads(SPEC_PATH.read_text(encoding='utf-8'))


def group_id(re, alpha, eccentricity):
    return f're{int(re):03d}_a{int(alpha):02d}_e{round(100*eccentricity):02d}'


def split_for(parameters, spec):
    values = list(parameters)
    if values in spec['test_combinations']:
        return 'test'
    if values in spec['validation_combinations']:
        return 'validation'
    return 'train'


def solver_signature(report=None):
    if report is not None:
        return {name: report['source_sha256'][name] for name in SOLVER_FILES}
    root = Path(__file__).parent
    return {name: digest(root / name) for name in SOLVER_FILES}


def assessment_signature():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in (
        'sr_campaign.py', 'sr_accuracy.py', 'sr_data.py', 'compare.py', 'analysis.py')}


def load_qualification(path):
    q = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_qualification(q)
    return q


def verify_qualification(q):
    if (q.get('schema') != 'cardiac-flow-sr-qualification-1' or q.get('status') != 'passed'
            or q.get('campaign_sha256') != digest(SPEC_PATH)
            or q.get('solver_signature') != solver_signature()
            or q.get('assessment_signature') != assessment_signature()):
        raise ValueError('qualification is missing, failed, stale, or belongs to another solver/campaign')


def configurations(study, qualification=None):
    s = specification()
    q = load_qualification(qualification) if qualification else None
    if study not in STUDIES:
        raise ValueError(f'unknown SR study: {study}')
    if study != 'sr_sweep' and q:
        raise ValueError('only sr_sweep consumes qualification')

    def case(parameters, h=2, t=2, angle=0., tight=False):
        re, alpha, eccentricity = parameters
        c = dict(geometry='straight', expansion=0., radius=s['radius'], length=s['length'],
                 rho=s['rho'], mu=s['mu'], harmonics=s['harmonics'], inlet='parabolic',
                 eccentricity=eccentricity, orientation=angle, initial='zero', benchmark=False,
                 mesh_size=s['mesh_sizes'][h], steps_per_cycle=s['steps_per_cycle'][t],
                 min_cycles=6 if tight else s['min_cycles'], max_cycles=s['max_cycles'],
                 periodic_tolerance=s['tight_periodic_tolerance'] if tight else s['periodic_tolerance'],
                 grad_div_ratio=0., outlet_pressure=0.)
        c.update(flow_parameters(re, alpha, radius=c['radius'], rho=c['rho'], mu=c['mu']))
        c['_sr'] = dict(study=study, configuration_group=group_id(*parameters),
                        campaign_sha256=digest(SPEC_PATH), reynolds=re, alpha=alpha,
                        split=split_for(parameters, s) if study == 'sr_sweep' else (
                            'pilot_only' if study == 'sr_pilot' else 'accuracy_only'),
                        numerics_status='qualified' if q else 'candidate')
        if q:
            c.update(q['selected_numerics'])
            c['_sr']['qualification_sha256'] = digest(qualification)
        return c

    if study == 'sr_sweep':
        return {group_id(*p): case(p) for p in itertools.product(
            s['reynolds'], s['alpha'], s['eccentricity'])}
    if study == 'sr_pilot':
        return {group_id(*p): case(p, h=0, t=0) for p in (
            s['representatives'][0], s['representatives'][1], s['representatives'][-1])}
    result = {}
    for p in s['representatives']:
        prefix = group_id(*p)
        for h, t in ((0, 2), (1, 2), (2, 2), (2, 0), (2, 1)):
            result[f'{prefix}__h{h}_t{t}'] = case(p, h, t)
        result[f'{prefix}__periodic_tight'] = case(p, tight=True)
    for p in s['rotation_representatives']:
        for i, angle in enumerate(s['rotation_angles']):
            for h in (1, 2):
                result[f'{group_id(*p)}__rot{i}_h{h}'] = case(p, h, angle=angle)
    return result
